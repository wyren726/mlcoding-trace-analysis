from __future__ import annotations

from typing import Any

from .common import redact, stable_id


class SessionJSONLAdapter:
    """Adapter for Claude-style `_internal/session.jsonl` transcript files."""

    name = "session_jsonl"

    def __init__(self) -> None:
        self._active_turns: dict[str, str] = {}

    def detect(self, record: dict[str, Any]) -> bool:
        return (record.get("type") in {"user", "assistant", "attachment", "queue-operation",
                                      "system", "progress", "ai-title", "last-prompt"}
                and isinstance(record.get("sessionId"), str)
                and "payload" not in record and "__source__" not in record)

    def convert(self, record: dict[str, Any], *, batch_id: str,
                source_file: str, record_index: int) -> list[dict[str, Any]]:
        session_id = str(record.get("sessionId") or stable_id("ses", source_file))
        record_type = str(record.get("type") or "unknown")
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        blocks = message.get("content")
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if not isinstance(blocks, list):
            blocks = []
        is_user_prompt = record_type == "user" and any(
            isinstance(block, dict) and block.get("type") == "text" for block in blocks
        )
        if is_user_prompt:
            native_turn = record.get("promptId") or record.get("uuid") or f"{source_file}:{record_index}"
            self._active_turns[session_id] = stable_id("turn", session_id, native_turn)
        turn_id = self._active_turns.get(session_id)
        base_sequence = (record_index + 1) * 100
        raw = redact(record)
        events: list[dict[str, Any]] = []

        def emit(event_type: str, data: Any, offset: int, *, assigned_turn: str | None = turn_id) -> None:
            events.append({
                "schema_version": "v1.0", "batch_id": batch_id,
                "event_id": stable_id("evt", session_id, record.get("uuid"), record_index, offset, event_type),
                "session_id": session_id, "trace_id": None, "turn_id": assigned_turn,
                "sequence": base_sequence + offset, "timestamp": record.get("timestamp"),
                "type": event_type, "data": redact(data), "raw": raw,
                "source": {"input_file": source_file, "record_index": record_index,
                           "adapter": self.name},
            })

        if record_type == "user":
            for index, block in enumerate(blocks, 1):
                if not isinstance(block, dict):
                    emit("unknown", {"content": block, "harness": "claude_code"}, index)
                elif block.get("type") == "text":
                    emit("user_message", {"content": block.get("text", ""), "harness": "claude_code"}, index)
                elif block.get("type") == "tool_result":
                    emit("tool_result", {"call_id": block.get("tool_use_id"),
                                          "content": block.get("content"),
                                          "is_error": block.get("is_error"),
                                          "harness": "claude_code"}, index)
                else:
                    emit("unknown", {"content_block": block, "harness": "claude_code"}, index)
        elif record_type == "assistant":
            model = message.get("model")
            emit("model_call", {"request_id": message.get("id"), "model": model,
                                "provider": message.get("provider"), "harness": "claude_code"}, 1)
            for index, block in enumerate(blocks, 2):
                if not isinstance(block, dict):
                    emit("unknown", {"content": block, "model": model, "harness": "claude_code"}, index)
                    continue
                kind = block.get("type")
                if kind == "thinking":
                    emit("reasoning", {"content": block.get("thinking", ""), "model": model,
                                       "harness": "claude_code"}, index)
                elif kind == "redacted_thinking":
                    emit("reasoning", {"content": "<REDACTED_THINKING>", "model": model,
                                       "harness": "claude_code", "redacted": True}, index)
                elif kind == "text":
                    emit("assistant_message", {"content": block.get("text", ""), "model": model,
                                               "harness": "claude_code"}, index)
                elif kind == "tool_use":
                    emit("tool_call", {"call_id": block.get("id"), "tool_name": block.get("name"),
                                       "arguments": block.get("input"), "model": model,
                                       "harness": "claude_code"}, index)
                else:
                    emit("unknown", {"content_block": block, "model": model,
                                     "harness": "claude_code"}, index)
            usage = message.get("usage")
            if isinstance(usage, dict):
                emit("token_usage", {**usage, "model": model, "harness": "claude_code"}, len(blocks) + 2)
            if record.get("isApiErrorMessage") or record.get("error"):
                emit("error", {"error": record.get("error") or "API error",
                               "status_code": record.get("apiErrorStatus"),
                               "model": model, "harness": "claude_code"}, len(blocks) + 3)
            emit("model_result", {"request_id": message.get("id"), "model": model,
                                  "stop_reason": message.get("stop_reason"),
                                  "harness": "claude_code"}, len(blocks) + 4)
            if message.get("stop_reason") in {"end_turn", "stop_sequence"}:
                self._active_turns.pop(session_id, None)
        else:
            data = {key: value for key, value in record.items()
                    if key not in {"message", "cwd", "gitBranch"}}
            data.update({"kind": record_type, "harness": "claude_code"})
            # Platform metadata must never create a user Turn. If execution is active,
            # progress/system records remain attached for chronological fidelity.
            assigned = turn_id if record_type in {"progress", "system"} else None
            emit("system_event", data, 1, assigned_turn=assigned)
        return events
