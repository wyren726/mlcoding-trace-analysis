from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "event_sequence", "review_id", "case_id", "batch_id", "analysis_run_id",
    "trace_id", "episode_id", "decision", "note", "reviewer_name", "revision",
    "created_at",
)
ALLOWED_DECISIONS = {"confirmed", "excluded", "review"}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _normalize(record: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Review {record.get('review_id')!r} misses: {', '.join(missing)}")
    if record["decision"] not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported review decision: {record['decision']!r}")
    normalized = {
        "schema_version": "site-human-review-event-v1",
        "event_sequence": int(record["event_sequence"]),
        "review_id": str(record["review_id"]),
        "case_id": str(record["case_id"]),
        "batch_id": str(record["batch_id"]),
        "analysis_run_id": str(record["analysis_run_id"]),
        "trace_id": str(record["trace_id"]),
        "episode_id": str(record["episode_id"]),
        "decision": str(record["decision"]),
        "attribution_override": record.get("attribution_override") or None,
        "note": str(record["note"]).strip(),
        "reviewer_name": str(record["reviewer_name"]).strip(),
        "revision": int(record["revision"]),
        "supersedes_review_id": record.get("supersedes_review_id") or None,
        "created_at": str(record["created_at"]),
    }
    if normalized["event_sequence"] < 1 or normalized["revision"] < 1:
        raise ValueError(f"Review {normalized['review_id']!r} has an invalid sequence or revision")
    return normalized


def _render(records: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def import_site_review_export(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Merge a Sites D1 export into the canonical local, human-readable mirror."""
    source = input_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    events_path = destination / "site_review_events.jsonl"
    decisions_path = destination / "human_review_decisions.jsonl"
    manifest_path = destination / "site_sync_manifest.json"

    by_id: dict[str, dict[str, Any]] = {}
    if events_path.is_file():
        for record in _read_jsonl(events_path):
            normalized = _normalize(record)
            by_id[normalized["review_id"]] = normalized
    for record in _read_jsonl(source):
        normalized = _normalize(record)
        previous = by_id.get(normalized["review_id"])
        if previous is not None and previous != normalized:
            raise ValueError(f"Conflicting content for review_id {normalized['review_id']}")
        by_id[normalized["review_id"]] = normalized

    events = sorted(by_id.values(), key=lambda item: (
        item["event_sequence"], item["created_at"], item["review_id"],
    ))
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        current = latest.get(event["case_id"])
        if current is None or (event["revision"], event["event_sequence"]) > (
            current["revision"], current["event_sequence"],
        ):
            latest[event["case_id"]] = event
    decisions = [
        {"schema_version": "human-review-decision-v1", **{
            key: value for key, value in event.items() if key != "schema_version"
        }}
        for event in sorted(latest.values(), key=lambda item: (
            item["batch_id"], item["analysis_run_id"], item["trace_id"], item["episode_id"],
        ))
    ]

    events_content = _render(events)
    decisions_content = _render(decisions)
    _atomic_write(events_path, events_content)
    _atomic_write(decisions_path, decisions_content)
    manifest = {
        "schema_version": "site-human-review-sync-v1",
        "source": "chatgpt-sites-d1",
        "input_path": str(source),
        "synced_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event_count": len(events),
        "latest_decision_count": len(decisions),
        "last_event_sequence": max((item["event_sequence"] for item in events), default=0),
        "events_path": str(events_path),
        "decisions_path": str(decisions_path),
        "events_sha256": hashlib.sha256(events_content).hexdigest(),
        "decisions_sha256": hashlib.sha256(decisions_content).hexdigest(),
    }
    _atomic_write(manifest_path, json.dumps(
        manifest, ensure_ascii=False, indent=2,
    ).encode("utf-8"))
    return {**manifest, "manifest_path": str(manifest_path)}


__all__ = ["import_site_review_export"]
