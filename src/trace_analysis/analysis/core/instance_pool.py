from __future__ import annotations

import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare_instance_pool(scope_dirs: list[Path], output_dir: Path,
                          consolidated_root: Path | None = None) -> dict[str, Any]:
    """Merge strictly included demand instances and report reusable Turn features."""
    output_dir.mkdir(parents=True, exist_ok=False)
    instances: dict[str, dict[str, Any]] = {}
    source_summaries = []

    for scope_dir in scope_dirs:
        summary = json.loads((scope_dir / "scope_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or summary.get("unaudited_included") != 0:
            raise ValueError(f"Scope result is not complete and strictly audited: {scope_dir}")
        source_summaries.append({
            "scope_dir": str(scope_dir.resolve()),
            "scope_version": summary.get("scope_version"),
            "included": summary.get("included"),
        })
        for path in sorted((scope_dir / "filtered_features").glob("*.jsonl")):
            for record in _jsonl(path):
                for instance in record.get("values", {}).get("instances") or []:
                    instance_id = instance.get("instance_id")
                    if not instance_id:
                        raise ValueError(f"Included instance without instance_id in {path}")
                    row = {
                        "instance_id": instance_id,
                        "trace_id": record.get("trace_id"),
                        "session_id": record.get("session_id"),
                        "turn_id": record.get("turn_id"),
                        "source_feature_file": str(path.resolve()),
                        "demand_feature_version": record.get("feature_version"),
                        "user_goal": record.get("values", {}).get("user_goal"),
                        "scope": instance.get("ml_llm_coding_scope"),
                        "instance": instance,
                    }
                    known = instances.get(instance_id)
                    if known and known != row:
                        raise ValueError(f"Conflicting duplicate instance_id: {instance_id}")
                    instances[instance_id] = row

    feature_by_turn: dict[str, dict[str, str]] = defaultdict(dict)
    if consolidated_root and consolidated_root.exists():
        for path in sorted(consolidated_root.glob("*/consolidated/*.jsonl")):
            if any(name in path.parts for name in ("demand_capability_instances", "ml_llm_coding_scope")):
                continue
            for record in _jsonl(path):
                turn_id = record.get("turn_id")
                feature_set = record.get("feature_set")
                if turn_id and feature_set:
                    feature_by_turn[turn_id][feature_set] = str(record.get("feature_version") or "")

    feature_sets = sorted({name for values in feature_by_turn.values() for name in values})
    pool_path = output_dir / "demand_capability_instances.jsonl"
    index_path = output_dir / "instance_index.jsonl"
    coverage_path = output_dir / "feature_coverage.csv"
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in instances.values():
        key = (row["trace_id"], row["session_id"], row["turn_id"], row["source_feature_file"],
               row["demand_feature_version"])
        group = grouped.setdefault(key, {
            "feature_set": "demand_capability_instances",
            "feature_version": row["demand_feature_version"],
            "trace_id": row["trace_id"], "session_id": row["session_id"], "turn_id": row["turn_id"],
            "values": {"user_goal": row["user_goal"], "instances": []},
            "source_feature_file": row["source_feature_file"],
        })
        group["values"]["instances"].append(row["instance"])

    with pool_path.open("w", encoding="utf-8") as pool, index_path.open("w", encoding="utf-8") as index:
        for group in sorted(grouped.values(), key=lambda item: (str(item["trace_id"]), str(item["turn_id"]))):
            group["values"]["instances"].sort(key=lambda item: item["instance_id"])
            pool.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")) + "\n")
        for row in sorted(instances.values(), key=lambda item: item["instance_id"]):
            coverage = feature_by_turn.get(row["turn_id"], {})
            index.write(json.dumps({
                **{k: row[k] for k in ("instance_id", "trace_id", "session_id", "turn_id",
                                        "source_feature_file", "demand_feature_version")},
                "scope_version": (row.get("scope") or {}).get("scope_version"),
                "available_turn_features": coverage,
                "missing_turn_features": [name for name in feature_sets if name not in coverage],
            }, ensure_ascii=False, separators=(",", ":")) + "\n")

    fields = ["instance_id", "trace_id", "session_id", "turn_id", "demand_feature_version",
              "scope_version", *feature_sets]
    with coverage_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(instances.values(), key=lambda item: item["instance_id"]):
            coverage = feature_by_turn.get(row["turn_id"], {})
            writer.writerow({
                **{k: row.get(k) for k in ("instance_id", "trace_id", "session_id", "turn_id",
                                            "demand_feature_version")},
                "scope_version": (row.get("scope") or {}).get("scope_version"),
                **{name: coverage.get(name, "") for name in feature_sets},
            })

    counts = {name: sum(name in feature_by_turn.get(row["turn_id"], {}) for row in instances.values())
              for name in feature_sets}
    manifest = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "instance_count": len(instances),
        "turn_record_count": len(grouped),
        "source_scope_results": source_summaries,
        "turn_feature_coverage": counts,
        "outputs": {
            "demand_capability_instances": str(pool_path.resolve()),
            "instance_index": str(index_path.resolve()),
            "feature_coverage": str(coverage_path.resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
