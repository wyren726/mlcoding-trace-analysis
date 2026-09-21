from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def pipeline_status(preprocessed_root: Path, analysis_root: Path,
                    *, only_incomplete: bool = False) -> dict[str, Any]:
    preprocessed = preprocessed_root.expanduser().resolve()
    analyses = analysis_root.expanduser().resolve()
    registered_trace_counts: dict[str, int] = {}
    registry = preprocessed / "data_registry.csv"
    if registry.is_file():
        with registry.open(encoding="utf-8-sig", newline="") as source:
            for item in csv.DictReader(source):
                try:
                    registered_trace_counts[str(item.get("batch_id") or "")] = int(
                        item.get("session_count") or 0
                    )
                except (TypeError, ValueError):
                    continue
    rows: list[dict[str, Any]] = []
    for batch in sorted(preprocessed.glob("batch_*")):
        if not (batch / "unified_turns.jsonl").is_file():
            continue
        candidates = []
        for manifest_path in [
            *(analyses / batch.name).glob("*/02_user_screen/manifest.json"),
            *(analyses / batch.name).glob("*/02_analyze/manifest.json"),
        ]:
            manifest = _read_json(manifest_path)
            if manifest.get("status") == "preview":
                continue
            candidates.append((str(manifest.get("updated_at") or ""), manifest_path, manifest))
        latest = max(candidates, default=None, key=lambda item: (item[0], str(item[1])))
        if latest is None:
            expected = registered_trace_counts.get(batch.name)
            row = {
                "batch_id": batch.name,
                "batch_path": str(batch),
                "01_preprocess": "completed",
                "02_analyze": "pending",
                "03_export": "pending",
                "overall_status": "pending",
                "analysis_run_id": None,
                "total_trace_count": expected,
                "completed_trace_count": 0,
                "failed_trace_count": 0,
                "pending_trace_count": expected,
            }
        else:
            _, manifest_path, manifest = latest
            run_dir = manifest_path.parent.parent
            new_layout = manifest_path.parent.name == "02_user_screen"
            verify_manifest = _read_json(run_dir / "03_agent_verify" / "manifest.json")
            results_manifest = _read_json(run_dir / "04_results" / "manifest.json")
            publish_manifest = _read_json(run_dir / "05_report_publish" / "manifest.json")
            export_manifest = _read_json(run_dir / "03_export" / "manifest.json")
            total = manifest.get("target_trace_count", manifest.get("total_trace_count"))
            completed = int(manifest.get("completed_trace_count") or 0)
            try:
                pending = max(0, int(total) - completed)
            except (TypeError, ValueError):
                pending = None
            analyze_status = str(manifest.get("status") or "unknown")
            if new_layout:
                verify_status = str(verify_manifest.get("status") or "pending")
                results_status = str(results_manifest.get("status") or "pending")
                publish_status = str(publish_manifest.get("status") or "pending")
                overall = (
                    "completed" if all(value == "completed" for value in (
                        analyze_status, verify_status, results_status, publish_status
                    )) else next(
                        value for value in (
                            analyze_status, verify_status, results_status, publish_status
                        ) if value != "completed"
                    )
                )
            else:
                verify_status = "legacy_combined"
                results_status = str(export_manifest.get("status") or "pending")
                publish_status = "not_tracked"
                overall = (
                    "completed" if analyze_status == "completed"
                    and results_status == "completed" else analyze_status
                )
            row = {
                "batch_id": batch.name,
                "batch_path": str(batch),
                "01_preprocess": "completed",
                "02_user_screen": analyze_status,
                "03_agent_verify": verify_status,
                "04_results": results_status,
                "05_report_publish": publish_status,
                # Compatibility aliases for existing status consumers.
                "02_analyze": analyze_status,
                "03_export": results_status,
                "overall_status": overall,
                "analysis_run_id": manifest.get("analysis_run_id"),
                "total_trace_count": total,
                "completed_trace_count": completed,
                "failed_trace_count": int(manifest.get("failed_trace_count") or 0),
                "pending_trace_count": pending,
                "run_dir": str(run_dir),
                "updated_at": manifest.get("updated_at"),
            }
        if not only_incomplete or row["overall_status"] != "completed":
            rows.append(row)
    return {
        "batch_count": len(rows),
        "incomplete_batch_count": sum(row["overall_status"] != "completed" for row in rows),
        "batches": rows,
    }


def write_pipeline_status(status: dict[str, Any], output: Path) -> str:
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in status.get("batches") or []:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    return str(destination)


__all__ = ["pipeline_status", "write_pipeline_status"]
