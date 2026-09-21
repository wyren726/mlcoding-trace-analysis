from __future__ import annotations

import hashlib
import json
import re
from typing import Any


REDACTIONS = (
    (re.compile(
        r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
        r"-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
        re.I | re.S,
    ), "<PRIVATE_KEY>"),
    (re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{12,}\b"), "<API_KEY>"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I), "Bearer <TOKEN>"),
    (re.compile(
        r"(?P<label>密码|口令|password|passwd|pwd|passphrase|"
        r"api[ _-]?key|access[ _-]?token|secret(?:[ _-]?key)?)"
        r"(?P<separator>\s*(?::|：|=|是)?\s*)"
        r"(?P<secret>(?!<)[A-Za-z0-9_./+@=-]{6,})",
        re.I,
    ), r"\g<label>\g<separator><SECRET>"),
    # Bound both sides to real-world address limits.  The previous unbounded
    # local-part search could backtrack quadratically across a multi-megabyte
    # tool result that did not contain an ``@`` at all.
    (re.compile(r"\b[\w.+-]{1,64}@[\w.-]{1,253}\.[A-Za-z]{2,63}\b"), "<EMAIL>"),
    (re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"), "<IP_ADDRESS>"),
    (re.compile(r"(?<![\w])/(?:Users|home)/[^\s\"']+"), "<LOCAL_PATH>"),
    (re.compile(r"(?<![\w])/mnt/(?:agent-workspace|data)(?:/[^\s\"']*)?"), "<LOCAL_PATH>"),
    (re.compile(r"\b[A-Za-z]:\\Users\\[^\s\"']+", re.I), "<LOCAL_PATH>"),
)


def redact_text(value: str) -> str:
    for pattern, replacement in REDACTIONS:
        value = pattern.sub(replacement, value)
    return value


def redact(value: Any) -> Any:
    """Return a structure-preserving, deeply redacted copy of an input record."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


def stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def canonical_type(source_type: str, nested_type: str = "") -> str:
    value = nested_type or source_type
    return {
        "user_message": "user_message",
        "agent_message": "assistant_message",
        "message": "assistant_message",
        "agent_reasoning": "reasoning",
        "reasoning": "reasoning",
        "function_call": "tool_call",
        "function_call_output": "tool_result",
        "custom_tool_call": "tool_call",
        "custom_tool_call_output": "tool_result",
        "token_count": "token_usage",
        "task_started": "system_event",
        "task_complete": "system_event",
        "turn_aborted": "error",
        "patch_apply_end": "system_event",
        "web_search_end": "system_event",
        "thread_settings_applied": "system_event",
        "context_compacted": "system_event",
        "compacted": "system_event",
    }.get(value, "system_event" if source_type in {
        "session_meta", "turn_context", "world_state", "event_msg", "compacted"
    } else "unknown")
