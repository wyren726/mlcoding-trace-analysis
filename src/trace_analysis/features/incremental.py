from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable


def compatible_feature_files(output_root: Path, feature_set: str, feature_version: str,
                             provider: str = "", model: str = "",
                             input_batch: str = "") -> list[Path]:
    registry = output_root / "feature_registry.csv"
    paths: list[Path] = []
    if registry.exists():
        with registry.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("feature_set") != feature_set or row.get("feature_version") != feature_version:
                    continue
                if provider and row.get("provider") != provider:
                    continue
                if model and row.get("model") != model:
                    continue
                if input_batch and row.get("input_batch") != input_batch:
                    continue
                path = Path(row.get("output_path") or "")
                if path.is_file() and path not in paths:
                    paths.append(path)
    # A semantic run writes this manifest before its first request. This lets a
    # later invocation reuse records that were flushed before an interruption,
    # even though the interrupted run never reached the registry-finalization step.
    for manifest_path in (output_root / feature_set).glob("*/run_manifest.json"):
        try:
            row = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("feature_set") != feature_set or row.get("feature_version") != feature_version:
            continue
        if provider and row.get("provider") != provider:
            continue
        if model and row.get("model") != model:
            continue
        if input_batch and row.get("input_batch") != input_batch:
            continue
        path = Path(row.get("output_path") or "")
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths


def completed_turn_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record: dict[str, Any] = json.loads(line)
                if record.get("turn_id"):
                    result.add(str(record["turn_id"]))
    return result


def consolidate_feature_set(output_root: Path, feature_set: str, feature_version: str,
                            input_batches: list[str], provider: str = "",
                            model: str = "") -> dict[str, Any]:
    destination = output_root / feature_set / "consolidated"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    total = 0
    for batch in input_batches:
        source_paths = compatible_feature_files(
            output_root, feature_set, feature_version, provider, model, batch
        )
        records: dict[str, dict[str, Any]] = {}
        provenance: dict[str, str] = {}
        for source_path in source_paths:
            with source_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("feature_set") != feature_set:
                        continue
                    turn_id = str(record.get("turn_id") or "")
                    if not turn_id:
                        raise ValueError(f"Feature record without turn_id: {source_path}")
                    if turn_id in records:
                        raise ValueError(
                            f"Duplicate successful feature for {turn_id}: "
                            f"{provenance[turn_id]} and {source_path}"
                        )
                    records[turn_id] = record
                    provenance[turn_id] = str(source_path.resolve())
        output_path = destination / f"{batch}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for turn_id in sorted(records):
                record = dict(records[turn_id])
                record["consolidated_from"] = provenance[turn_id]
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        outputs.append({"input_batch": batch, "record_count": len(records),
                        "source_file_count": len(source_paths),
                        "output_path": str(output_path.resolve())})
        total += len(records)
    manifest = {
        "feature_set": feature_set, "feature_version": feature_version,
        "provider": provider, "model": model,
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "batch_count": len(outputs), "record_count": total, "outputs": outputs,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path.resolve())}
