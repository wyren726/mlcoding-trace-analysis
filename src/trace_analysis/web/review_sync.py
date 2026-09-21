from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .review_store import SQLiteReviewStore


def _render_jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            {"schema_version": "human-review-decision-v1", **record},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for record in records
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def sync_reviews_to_jsonl(
    store: SQLiteReviewStore,
    output_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Export latest human decisions without mutating model analysis JSONL."""
    records = store.latest_reviews()
    content = _render_jsonl(records)
    previous = output_path.read_bytes() if output_path.is_file() else None
    changed = previous != content
    if changed:
        _atomic_write(output_path, content)
    manifest = {
        "schema_version": "human-review-sync-v1",
        "synced_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "review_count": len(records),
        "source_event_count": store.event_count(),
        "last_event_sequence": store.max_event_sequence(),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "changed": changed,
    }
    if manifest_path is not None:
        _atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return manifest
