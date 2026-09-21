from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterator

from .._compat import (
    TraceBundle,
    _real_user_event,
    iter_trace_bundles,
)


SCHEMA_VERSION = "user-trace-v2"
EXTRACTOR_VERSION = "real-user-turns-v2"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def user_turn_path(batch_dir: Path) -> Path:
    batch = batch_dir.expanduser().resolve()
    return batch / f"user_turns__{batch.name}.jsonl"


def build_user_turns(batch_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Materialize the deterministic, Trace-level User-only Stage 01 view."""
    batch = batch_dir.expanduser().resolve()
    source = batch / "unified_turns.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    output = user_turn_path(batch)
    manifest_path = batch / "user_turns_manifest.json"
    if output.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {**manifest, "status": "reused"}

    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    trace_count = 0
    user_turn_count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for bundle in iter_trace_bundles(source):
            session_ids: set[str] = set()
            harnesses: set[str] = set()
            models: set[str] = set()
            user_turns = []
            source_records = []
            for turn in bundle.turns:
                if turn.get("session_id"):
                    session_ids.add(str(turn["session_id"]))
                if turn.get("harness"):
                    harnesses.add(str(turn["harness"]))
                models.update(
                    str(value) for value in turn.get("agent_models") or []
                    if value and value != "<synthetic>"
                )
                message = _real_user_event(turn)
                if message is None:
                    continue
                event_id = str(message.get("event_id") or "")
                data = message.get("data") if isinstance(message.get("data"), dict) else {}
                user_turns.append({
                    "turn_id": turn.get("turn_id"),
                    "turn_index": turn.get("turn_index"),
                    "event_id": event_id,
                    # This is the position in the complete User/Assistant/Tool
                    # event stream.  It is deliberately distinct from
                    # turn_index, which only orders User turns.
                    "event_sequence": message.get("sequence"),
                    "timestamp": message.get("timestamp"),
                    "content": data.get("content"),
                })
                original = next(
                    (event for event in turn.get("events") or []
                     if str(event.get("event_id") or "") == event_id),
                    {},
                )
                source_ref = original.get("source") if isinstance(original, dict) else None
                if isinstance(source_ref, dict) and source_ref:
                    source_records.append({"event_id": event_id, **source_ref})
            record = {
                "schema_version": SCHEMA_VERSION,
                "batch_id": bundle.batch_id,
                "trace_id": bundle.trace_id,
                "session_ids": sorted(session_ids),
                "harnesses": sorted(harnesses) or ["unknown"],
                "agent_models": sorted(models) or ["unknown"],
                "input_view": "all_real_user_turns",
                "user_turn_count": len(user_turns),
                "user_turns": user_turns,
                "source_records": source_records,
            }
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            trace_count += 1
            user_turn_count += len(user_turns)
    temporary.replace(output)
    manifest = {
        "manifest_version": "1.0",
        "stage": "01_preprocess",
        "product": "user_turns",
        "status": "completed",
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "batch_id": batch.name,
        "input_path": str(source),
        "output_path": str(output),
        "trace_count": trace_count,
        "user_turn_count": user_turn_count,
        "output_size_bytes": output.stat().st_size,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def iter_user_trace_bundles(path: Path) -> Iterator[TraceBundle]:
    """Rebuild the minimal User-only bundle required by Stage 02 validation."""
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            trace_id = str(value.get("trace_id") or "")
            if not trace_id:
                raise ValueError(f"User Trace line {line_number} has no trace_id: {path}")
            batch_id = str(value.get("batch_id") or path.parent.name)
            session_ids = value.get("session_ids") or [""]
            harnesses = value.get("harnesses") or ["unknown"]
            models = value.get("agent_models") or ["unknown"]
            turns = []
            for item in value.get("user_turns") or []:
                turns.append({
                    "schema_version": "v1.0",
                    "batch_id": batch_id,
                    "trace_id": trace_id,
                    "session_id": session_ids[0] if len(session_ids) == 1 else "",
                    "turn_id": item.get("turn_id"),
                    "turn_index": item.get("turn_index"),
                    "status": "unknown",
                    "harness": harnesses[0] if len(harnesses) == 1 else "mixed",
                    "agent_models": models,
                    "events": [{
                        "event_id": item.get("event_id"),
                        "sequence": item.get("event_sequence"),
                        "timestamp": item.get("timestamp"),
                        "type": "user_message",
                        "data": {"content": item.get("content")},
                    }],
                })
            yield TraceBundle(batch_id, trace_id, tuple(turns))


__all__ = [
    "EXTRACTOR_VERSION", "SCHEMA_VERSION", "build_user_turns",
    "iter_user_trace_bundles", "user_turn_path",
]
