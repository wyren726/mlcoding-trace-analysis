from __future__ import annotations

import json
import hashlib
import re
from difflib import SequenceMatcher
from typing import Any

from .common import redact, stable_id


NON_SEMANTIC_KEYS = {"cache_control", "cache_creation", "cache_read", "metadata"}
PLATFORM_PREFIXES = (
    "<system-reminder>", "[system]", "Continue from where you left off.",
    "This session is being continued", "Resume from the paused agent session.",
    "CRITICAL: Respond with TEXT ONLY.", "Generate a concise chat-conversation title from the user's first message.",
)
UPLOAD_MARKER = "The user just uploaded"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in sorted(value.items()) if key not in NON_SEMANTIC_KEYS}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _canonical_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": "" if content is None else str(content)}]
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append({"type": "text", "text": str(block)})
            continue
        kind = str(block.get("type") or "unknown")
        if kind == "text":
            blocks.append({"type": "text", "text": str(block.get("text") or "")})
        elif kind == "thinking":
            blocks.append({"type": "thinking", "thinking": str(block.get("thinking") or "")})
        elif kind == "tool_use":
            blocks.append({"type": "tool_use", "id": block.get("id"), "name": block.get("name"),
                           "input": _clean(block.get("input"))})
        elif kind == "tool_result":
            blocks.append({"type": "tool_result", "tool_use_id": block.get("tool_use_id"),
                           "content": _clean(block.get("content")), "is_error": block.get("is_error")})
        else:
            blocks.append(_clean(block))
    return blocks


def _message_fingerprint(message: dict[str, Any]) -> str:
    canonical = {"role": str(message.get("role") or "unknown"),
                 "content": _canonical_blocks(message.get("content"))}
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def _new_message_start(previous: list[str], current: list[str]) -> int:
    """Return the first genuinely new message in a cumulative/trimmed snapshot."""
    if not previous:
        return 0
    if current[:len(previous)] == previous:
        return len(previous)
    prefix = 0
    for old_hash, new_hash in zip(previous, current):
        if old_hash != new_hash:
            break
        prefix += 1
    # Handles history-window trimming: anchor on an exact suffix of the last
    # processed conversation and an exact prefix of the current snapshot.
    maximum = min(len(previous), len(current))
    for size in range(maximum, 0, -1):
        if previous[-size:] == current[:size]:
            return size
    # Last-resort alignment for snapshots with non-semantic insertions/removals.
    matcher = SequenceMatcher(a=previous, b=current, autojunk=False)
    suffix_blocks = [block for block in matcher.get_matching_blocks()
                     if block.size and block.a + block.size == len(previous)]
    return max([prefix, *(block.b + block.size for block in suffix_blocks)])


def _is_previous_response(message: dict[str, Any], state: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    blocks = _canonical_blocks(message.get("content"))
    text = "\n".join(str(block.get("text") or "") for block in blocks if block.get("type") == "text").strip()
    thinking = "\n".join(str(block.get("thinking") or "") for block in blocks
                          if block.get("type") == "thinking").strip()
    expected_text = str(state.get("last_response") or "").strip()
    expected_thinking = str(state.get("last_reasoning") or "").strip()
    return bool(expected_text and text == expected_text and (not expected_thinking or thinking == expected_thinking))


def _split_user_text(value: str) -> tuple[str, str | None]:
    marker_index = value.find(UPLOAD_MARKER)
    if marker_index < 0:
        return value.strip(), None
    user_text = value[:marker_index].strip()
    platform_text = value[marker_index:].strip()
    return user_text, platform_text


def _uploaded_files(value: str) -> list[dict[str, str]]:
    files = []
    pattern = re.compile(r"^-\s+(.+?):\s+local path\s+(.+?)\s*$", re.MULTILINE)
    for match in pattern.finditer(value):
        files.append({"name": match.group(1).strip(), "path": match.group(2).strip()})
    return files


class SLSProxyAdapter:
    name = "sls_proxy"

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def detect(self, record: dict[str, Any]) -> bool:
        if not isinstance(record.get("content"), str):
            return False
        try:
            return isinstance(json.loads(record["content"]), dict)
        except json.JSONDecodeError:
            return False

    def convert(self, record: dict[str, Any], *, batch_id: str, source_file: str, record_index: int) -> list[dict[str, Any]]:
        payload = json.loads(record["content"])
        headers = ((payload.get("request_metadata") or {}).get("headers") or {})
        native_session = headers.get("x-claude-code-session-id")
        request_id = str(payload.get("id") or stable_id("req", source_file, record_index))
        session_id = str(native_session or stable_id("ses", source_file, request_id))
        timestamp = payload.get("timestamp") or record.get("_time_") or record.get("__time__")
        state = self._states.setdefault(session_id, {"fingerprints": [], "turn_index": 0, "turn_id": None,
                                                     "sequence": 0, "last_response": None,
                                                     "last_reasoning": None})
        events: list[dict[str, Any]] = []

        def add(event_type: str, data: Any, raw: Any, *, turn_id: str | None = None) -> None:
            state["sequence"] += 1
            sequence = state["sequence"]
            events.append({
                "schema_version": "v1.0", "batch_id": batch_id,
                "event_id": stable_id("evt", session_id, request_id, sequence, event_type, data),
                "session_id": session_id, "trace_id": None, "turn_id": turn_id if turn_id is not None else state["turn_id"],
                "sequence": sequence, "timestamp": timestamp, "type": event_type,
                "data": redact(data), "raw": redact(raw),
                "source": {"input_file": source_file, "record_index": record_index, "adapter": self.name, "request_id": request_id},
            })

        user_agent = str(headers.get("user-agent") or headers.get("User-Agent") or "")
        harness = "claude_code" if re.search(r"claude[-_ ]?(?:cli|code)", user_agent, re.I) else ("codex" if "codex" in user_agent.lower() else "unknown")
        add("system_event", {"kind": "request_snapshot", "request_id": request_id, "model": payload.get("model"),
                             "provider": payload.get("provider"), "harness": harness}, record)
        messages = [message for message in payload.get("messages") or [] if isinstance(message, dict)]
        fingerprints = [_message_fingerprint(message) for message in messages]
        previous = state["fingerprints"]
        new_start = _new_message_start(previous, fingerprints)
        if new_start < len(messages) and _is_previous_response(messages[new_start], state):
            new_start += 1
        new_messages = messages[new_start:]
        state["fingerprints"] = fingerprints
        for message in new_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            blocks = message.get("content") if isinstance(message.get("content"), list) else [{"type": "text", "text": message.get("content")}]
            has_tool_result = any(isinstance(block, dict) and block.get("type") == "tool_result" for block in blocks)
            raw_user_text = "\n".join(str(block.get("text") or "") for block in blocks
                                      if isinstance(block, dict) and block.get("type") == "text").strip()
            user_text, upload_text = _split_user_text(raw_user_text) if role == "user" else (raw_user_text, None)
            is_platform = user_text.startswith(PLATFORM_PREFIXES)
            if role == "user" and user_text and not has_tool_result and not is_platform:
                state["turn_index"] += 1
                state["turn_id"] = stable_id("turn", session_id, state["turn_index"], user_text)
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_use":
                    add("tool_call", {"call_id": block.get("id"), "tool_name": block.get("name"), "arguments": block.get("input")}, block)
                elif kind == "tool_result":
                    add("tool_result", {"call_id": block.get("tool_use_id"), "content": block.get("content"), "is_error": block.get("is_error")}, block)
                elif kind == "thinking":
                    add("reasoning", {"content": block.get("thinking")}, block)
                elif kind == "text":
                    if role == "user":
                        block_user_text, block_upload_text = _split_user_text(str(block.get("text") or ""))
                        if block_user_text:
                            event_type = "system_event" if block_user_text.startswith(PLATFORM_PREFIXES) else "user_message"
                            add(event_type, {"kind": "platform_instruction", "content": block_user_text}
                                if event_type == "system_event" else {"content": block_user_text}, block)
                        if block_upload_text:
                            files = _uploaded_files(block_upload_text)
                            add("system_event", {"kind": "uploaded_files" if files else "platform_instruction",
                                                 "content": block_upload_text, "files": files}, block)
                    else:
                        add("assistant_message", {"content": block.get("text")}, block)
                else:
                    add("unknown", block, block)
        add("model_call", {"request_id": request_id, "model": payload.get("model"),
                           "provider": payload.get("provider")}, {"id": request_id, "model": payload.get("model")})
        if payload.get("reasoning"):
            add("reasoning", {"content": payload.get("reasoning")}, payload.get("reasoning"))
        if payload.get("response"):
            add("model_result", {"request_id": request_id, "model": payload.get("model"),
                                 "status_code": payload.get("status_code")}, {"id": request_id})
            add("assistant_message", {"content": payload.get("response")}, payload.get("response"))
        state["last_response"] = payload.get("response")
        state["last_reasoning"] = payload.get("reasoning")
        for call in payload.get("tool_calls") or []:
            if isinstance(call, dict):
                add("tool_call", {
                    "call_id": call.get("call_id") or call.get("id"),
                    "tool_name": call.get("tool_name") or call.get("name"),
                    "arguments": call.get("arguments") if "arguments" in call else call.get("input"),
                }, call)
        if payload.get("error") is not None:
            add("error", {"error": payload.get("error"), "status_code": payload.get("status_code")}, payload.get("error"))
        return events
