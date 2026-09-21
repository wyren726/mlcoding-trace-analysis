from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .runner import iter_trace_bundles, user_trace_payload


def _old_negative_trace_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        with path.expanduser().resolve().open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                values = row.get("values") if isinstance(row, dict) else None
                if not isinstance(values, dict):
                    continue
                if values.get("explicit") is True and values.get("sentiment") in {
                    "negative", "mixed",
                }:
                    result.add(str(row.get("trace_id") or ""))
    result.discard("")
    return result


def _rank(trace_id: str) -> str:
    return hashlib.sha256(f"trace-pilot-v1|{trace_id}".encode()).hexdigest()


def build_pilot_sample(batch_dir: Path, output: Path, *, size: int = 16,
                       direct_max_chars: int = 120_000,
                       feedback_files: Iterable[Path] = ()) -> dict[str, Any]:
    if size <= 0:
        raise ValueError("Pilot size must be positive")
    batch = batch_dir.expanduser().resolve()
    turn_path = batch / "unified_turns.jsonl"
    if not turn_path.is_file():
        raise FileNotFoundError(turn_path)
    old_negative = _old_negative_trace_ids(feedback_files)
    candidates: list[dict[str, Any]] = []
    for bundle in iter_trace_bundles(turn_path):
        user_payload_chars = len(json.dumps(
            user_trace_payload(bundle), ensure_ascii=False, separators=(",", ":")
        ))
        source_event_chars = sum(
            len(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            for turn in bundle.turns for event in turn.get("events") or []
        )
        candidates.append({
            "batch_id": batch.name,
            "trace_id": bundle.trace_id,
            "turn_count": len(bundle.turns),
            "user_payload_chars": user_payload_chars,
            "source_event_chars": source_event_chars,
            "trace_size": (
                "short" if source_event_chars <= direct_max_chars else "long"
            ),
            "old_explicit_negative_signal": bundle.trace_id in old_negative,
        })
    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for trace_size in ("short", "long"):
        for negative in (True, False):
            bucket = [
                row for row in candidates
                if row["trace_size"] == trace_size
                and row["old_explicit_negative_signal"] is negative
            ]
            buckets[(trace_size, negative)] = sorted(
                bucket, key=lambda row: _rank(str(row["trace_id"]))
            )
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    base, remainder = divmod(min(size, len(candidates)), 4)
    for index, key in enumerate((
        ("short", True), ("long", True),
        ("short", False), ("long", False),
    )):
        quota = base + int(index < remainder)
        for row in buckets[key][:quota]:
            chosen.append(row)
            chosen_ids.add(str(row["trace_id"]))
    if len(chosen) < min(size, len(candidates)):
        remaining = sorted(
            (row for row in candidates if str(row["trace_id"]) not in chosen_ids),
            key=lambda row: _rank(str(row["trace_id"])),
        )
        chosen.extend(remaining[:min(size, len(candidates)) - len(chosen)])
    chosen.sort(key=lambda row: (
        row["trace_size"], not row["old_explicit_negative_signal"],
        row["turn_count"], row["trace_id"],
    ))
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in chosen:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    return {
        "status": "completed",
        "batch_id": batch.name,
        "candidate_trace_count": len(candidates),
        "sample_trace_count": len(chosen),
        "old_negative_trace_count": len(old_negative),
        "sample_trace_size_counts": dict(Counter(row["trace_size"] for row in chosen)),
        "sample_negative_signal_counts": dict(Counter(
            str(row["old_explicit_negative_signal"]).lower() for row in chosen
        )),
        "output_path": str(destination),
    }


__all__ = ["build_pilot_sample"]
