from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..incremental import compatible_feature_files, completed_turn_ids
from ...registries import registry_root_for_output, write_immutable_record


FIELDS = ["feature_run_id", "feature_set", "feature_version", "input_batch", "processed_at", "record_count",
          "error_count", "provider", "model", "prompt_version", "output_path", "status"]


def length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def content(data: dict[str, Any]) -> Any:
    return data.get("content") or data.get("text") or data.get("message") or ""


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
            return datetime.fromtimestamp(seconds).astimezone()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError, OSError):
        return None


def _token_values(data: dict[str, Any]) -> tuple[int, int, int]:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else data
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens", 0) or 0
    try:
        input_value, output_value = int(input_tokens), int(output_tokens)
        return input_value, output_value, int(total_tokens) if total_tokens else input_value + output_value
    except (TypeError, ValueError):
        return 0, 0, 0


def extract_turn(turn: dict[str, Any], run_id: str) -> dict[str, Any]:
    user_length = assistant_length = reasoning_length = model_calls = 0
    input_tokens = output_tokens = total_tokens = 0
    agent_models: set[str] = {str(value) for value in turn.get("agent_models") or []}
    tool_usage: dict[str, dict[str, int]] = defaultdict(lambda: {"call_count": 0, "result_count": 0, "argument_length": 0, "result_length": 0})
    calls: dict[str, str] = {}
    counted_call_ids: set[str] = set()
    counted_result_ids: set[str] = set()
    matched_call_ids: set[str] = set()
    unmatched_results: list[str] = []
    timestamps: list[datetime] = []
    first_user_at: datetime | None = None
    first_agent_at: datetime | None = None
    for event in turn.get("events") or []:
        event_type = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        event_at = _timestamp(event.get("timestamp"))
        if event_at:
            timestamps.append(event_at)
        model_name = data.get("model_name") or data.get("model")
        if model_name:
            agent_models.add(str(model_name))
        if event_type == "user_message":
            user_length += length(content(data))
            first_user_at = first_user_at or event_at
        elif event_type == "assistant_message":
            assistant_length += length(content(data))
            first_agent_at = first_agent_at or event_at
        elif event_type == "reasoning":
            reasoning_length += length(content(data))
            first_agent_at = first_agent_at or event_at
        elif event_type == "model_call":
            model_calls += 1
            first_agent_at = first_agent_at or event_at
        elif event_type == "token_usage":
            item_input, item_output, item_total = _token_values(data)
            input_tokens += item_input
            output_tokens += item_output
            total_tokens += item_total
        elif event_type == "tool_call":
            first_agent_at = first_agent_at or event_at
            name = str(data.get("tool_name") or data.get("name") or "unknown")
            call_id = str(data.get("call_id") or data.get("id") or "")
            if call_id:
                calls[call_id] = name
            # Some sources log the same logical call twice (for example once in a
            # model response and once at execution). Count a call_id only once.
            if not call_id or call_id not in counted_call_ids:
                tool_usage[name]["call_count"] += 1
                tool_usage[name]["argument_length"] += length(
                    data.get("arguments") if "arguments" in data else data.get("input")
                )
                if call_id:
                    counted_call_ids.add(call_id)
        elif event_type == "tool_result":
            call_id = str(data.get("call_id") or data.get("tool_call_id") or "")
            name = calls.get(call_id, "unknown")
            if call_id and call_id in calls:
                matched_call_ids.add(call_id)
            else:
                unmatched_results.append(call_id or "<missing_call_id>")
            if not call_id or call_id not in counted_result_ids:
                tool_usage[name]["result_count"] += 1
                tool_usage[name]["result_length"] += length(
                    data.get("output") if "output" in data else data.get("content")
                )
                if call_id:
                    counted_result_ids.add(call_id)
    return {
        "trace_id": turn.get("trace_id"), "session_id": turn.get("session_id"), "turn_id": turn.get("turn_id"),
        "feature_set": "basic_turn_features", "feature_version": "v4", "feature_run_id": run_id,
        "generated_by": {"method": "rule", "extractor": "basic_turn_features_v4"},
        "values": {"turn_status": turn.get("status"), "harness": turn.get("harness"),
                   "source_batch": turn.get("batch_id"),
                   "user_input_length": user_length, "assistant_output_length": assistant_length,
                   "reasoning_length": reasoning_length, "model_call_count": model_calls,
                   "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                                   "total_tokens": total_tokens},
                   "duration_ms": (round((max(timestamps) - min(timestamps)).total_seconds() * 1000)
                                   if len(timestamps) >= 2 else None),
                   "first_response_ms": (round((first_agent_at - first_user_at).total_seconds() * 1000)
                                         if first_user_at and first_agent_at and first_agent_at >= first_user_at else None),
                   "agent_models": sorted(agent_models),
                   "tool_usage": dict(sorted(tool_usage.items())),
                   "tool_pairing": {"matched_count": len(matched_call_ids),
                                    "unmatched_call_ids": sorted(set(calls) - matched_call_ids),
                                    "unmatched_result_call_ids": unmatched_results}},
    }


def run_basic_features(batch_dir: Path, output_root: Path) -> dict[str, Any]:
    turn_path = batch_dir / "unified_turns.jsonl"
    if not turn_path.exists():
        raise FileNotFoundError(turn_path)
    now = dt.datetime.now().astimezone()
    suffix = hashlib.sha256(str(turn_path.resolve()).encode()).hexdigest()[:8]
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}_{suffix}"
    run_dir = output_root / "basic_turn_features" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "features.jsonl"
    count = 0
    reused_files = compatible_feature_files(
        output_root, "basic_turn_features", "v4", input_batch=batch_dir.name
    )
    completed = completed_turn_ids(reused_files)
    skipped = 0
    with turn_path.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip():
                turn = json.loads(line)
                if str(turn.get("turn_id")) in completed:
                    skipped += 1
                    continue
                target.write(json.dumps(extract_turn(turn, run_id), ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    registry = output_root / "feature_registry.csv"
    write_header = not registry.exists()
    row = {"feature_run_id": run_id, "feature_set": "basic_turn_features", "feature_version": "v4",
           "input_batch": batch_dir.name, "processed_at": now.isoformat(), "record_count": count,
           "error_count": 0, "provider": "", "model": "", "prompt_version": "",
           "output_path": str(output.resolve()), "status": "success"}
    with registry.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    registry_record = write_immutable_record(
        "feature-runs", run_id, row, artifacts=(output,),
        root=registry_root_for_output(output_root),
    )
    return {**row, "skipped_existing_count": skipped,
            "reused_feature_files": [str(path.resolve()) for path in reused_files],
            "registry_record_path": str(registry_record.resolve())}
