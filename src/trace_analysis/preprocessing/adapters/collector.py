from __future__ import annotations

from typing import Any

from .common import canonical_type, redact, stable_id


class CollectorEventsAdapter:
    name = "collector_events"

    def detect(self, record: dict[str, Any]) -> bool:
        return all(key in record for key in ("trace_schema_version", "event_id", "session_id", "payload", "source"))

    def convert(self, record: dict[str, Any], *, batch_id: str, source_file: str, record_index: int) -> dict[str, Any]:
        wrapper = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        inner = wrapper.get("payload") if isinstance(wrapper.get("payload"), dict) else {}
        nested_type = str(inner.get("type") or "")
        source_type = str(record.get("event_type") or wrapper.get("type") or "unknown")
        event_type = canonical_type(source_type, nested_type)
        # Codex rollout response_item.message mirrors the user/agent event_msg
        # representation. Keep it for provenance but do not count the same
        # conversational message twice in downstream features and prompts.
        if source_type == "response_item" and nested_type == "message":
            event_type = "system_event"
        data: Any = inner
        if nested_type in {"user_message", "agent_message"}:
            data = {**inner, "content": inner.get("message")}
        elif nested_type in {"function_call", "custom_tool_call"}:
            data = {
                "call_id": inner.get("call_id") or inner.get("id"),
                "tool_name": inner.get("name") or inner.get("namespace") or "unknown",
                "arguments": inner.get("arguments") if "arguments" in inner else inner.get("input"),
                **({"status": inner.get("status")} if inner.get("status") is not None else {}),
            }
        elif nested_type in {"function_call_output", "custom_tool_call_output"}:
            data = {
                "call_id": inner.get("call_id") or inner.get("id"),
                "content": inner.get("output"),
                "is_error": inner.get("is_error"),
            }
        elif nested_type == "reasoning":
            data = {**inner, "content": inner.get("summary")}
        elif nested_type == "token_count":
            info = inner.get("info") if isinstance(inner.get("info"), dict) else {}
            data = {
                "usage": info.get("last_token_usage") or {},
                "total_usage": info.get("total_token_usage") or {},
                "model_context_window": info.get("model_context_window"),
            }
        return {
            "schema_version": "v1.0",
            "batch_id": batch_id,
            "event_id": str(record.get("event_id") or stable_id("evt", source_file, record_index, record)),
            "session_id": str(record.get("session_id") or stable_id("ses", source_file, record_index)),
            "trace_id": record.get("trace_id"),
            "turn_id": record.get("turn_id") or inner.get("turn_id"),
            "sequence": record.get("sequence") or record_index + 1,
            "timestamp": record.get("timestamp"),
            "type": event_type,
            "data": redact(data),
            "raw": redact(record),
            "source": {"input_file": source_file, "record_index": record_index, "adapter": self.name},
        }
