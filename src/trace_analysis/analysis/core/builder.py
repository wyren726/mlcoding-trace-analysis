from __future__ import annotations

import csv
import datetime as dt
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ...preprocessing.adapters.common import stable_id
from ...registries import registry_root_for_output, write_immutable_record


def _records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("feature_set") == "demand_capability_instances":
                        yield record


def _typed_evidence(instance: dict[str, Any]) -> list[dict[str, Any]]:
    if "requirement_evidence" in instance:
        rows = []
        for kind, field in (("requirement", "requirement_evidence"),
                            ("fulfillment", "fulfillment_evidence")):
            rows.extend({**item, "evidence_type": kind} for item in instance.get(field) or [])
        return rows
    return [{**item, "evidence_type": "unspecified"} for item in instance.get("evidence") or []]


def _normalized_name(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def _taxonomy_nodes(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        if node.get("capability_id") != "capability_root":
            rows.append(node)
        for child in node.get("children") or []:
            visit(child)

    visit(taxonomy["root"])
    return rows


def _match_capability(name: str, instance: dict[str, Any], existing_nodes: list[dict[str, Any]],
                      threshold: float = 0.78) -> dict[str, Any]:
    normalized = _normalized_name(name)
    for node in existing_nodes:
        if _normalized_name(str(node.get("name", ""))) == normalized:
            return {"capability_id": node["capability_id"], "name": node["name"],
                    "match_status": "matched_existing", "match_score": 1.0,
                    "suggested_existing_capability_id": None}
    candidate_context = _normalized_name(" ".join(str(instance.get(field) or "") for field in
                                                   ("candidate_capability", "need", "object", "expected_outcome")))
    scored = []
    for node in existing_nodes:
        node_name = _normalized_name(str(node.get("name", "")))
        node_context = _normalized_name(" ".join([
            str(node.get("name") or ""), str(node.get("definition") or ""),
            " ".join(node.get("inclusion_criteria") or []),
            " ".join(node.get("exclusion_criteria") or []),
        ]))
        name_score = difflib.SequenceMatcher(None, normalized, node_name).ratio()
        context_score = difflib.SequenceMatcher(None, candidate_context, node_context).ratio()
        scored.append((max(name_score, context_score), node))
    score, closest = max(scored, default=(0.0, {}), key=lambda item: item[0])
    candidate_id = stable_id("cap", normalized)
    if score >= threshold:
        return {"capability_id": candidate_id, "name": name, "match_status": "review_possible_match",
                "match_score": round(score, 4),
                "suggested_existing_capability_id": closest.get("capability_id")}
    return {"capability_id": candidate_id, "name": name, "match_status": "new_candidate",
            "match_score": round(score, 4) if existing_nodes else None,
            "suggested_existing_capability_id": None}


def build_candidate_core(feature_paths: list[Path], output_root: Path,
                         taxonomy_version: str | None = None,
                         existing_taxonomy: Path | None = None) -> dict[str, Any]:
    now = dt.datetime.now().astimezone()
    version = taxonomy_version or f"draft_{now.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / f"version_{version}" / "core"
    run_dir.mkdir(parents=True, exist_ok=False)

    existing_nodes = _taxonomy_nodes(existing_taxonomy)
    cases_by_id: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for record in _records(feature_paths):
        for instance in record.get("values", {}).get("instances") or []:
            name = str(instance.get("candidate_capability") or "待人工命名").strip()
            match = _match_capability(name, instance, existing_nodes)
            capability_id = match["capability_id"]
            instance_id = instance.get("instance_id") or stable_id(
                "case", record.get("trace_id"), record.get("turn_id"), instance.get("need"), name)
            source = {"feature_run_id": record.get("feature_run_id"),
                      "feature_version": record.get("feature_version"),
                      "generated_by": record.get("generated_by")}
            case = {
                "taxonomy_version": version,
                "capability_id": capability_id,
                "case_id": instance_id,
                "trace_id": record.get("trace_id"),
                "session_id": record.get("session_id"),
                "turn_ids": [record.get("turn_id")],
                "user_need": instance.get("need"),
                "expected_outcome": instance.get("expected_outcome"),
                "constraints": instance.get("constraints") or [],
                "fulfillment": instance.get("fulfillment"),
                "unmet_need": instance.get("agent_gap") or "",
                "evidence": _typed_evidence(instance),
                "source_features": [source],
            }
            if instance_id in cases_by_id:
                known = cases_by_id[instance_id]
                if source not in known["source_features"]:
                    known["source_features"].append(source)
                for evidence in case["evidence"]:
                    if evidence not in known["evidence"]:
                        known["evidence"].append(evidence)
                continue
            cases_by_id[instance_id] = case
            group = groups.setdefault(capability_id, {**match, "cases": []})
            group["cases"].append(case)

    cases = list(cases_by_id.values())

    nodes = []
    for capability_id, group in sorted(groups.items(), key=lambda item: item[1]["name"]):
        node_cases = group["cases"]
        nodes.append({
            "capability_id": capability_id, "parent_id": "capability_root", "level": 1,
            "name": group["name"], "path": ["Coding Agent 能力", group["name"]], "is_leaf": True,
            "status": group["match_status"], "match_score": group["match_score"],
            "suggested_existing_capability_id": group["suggested_existing_capability_id"],
            "definition": "待基于案例人工审核和归并",
            "inclusion_criteria": [], "exclusion_criteria": [],
            "case_count": len({case["case_id"] for case in node_cases}),
            "trace_count": len({case["trace_id"] for case in node_cases}),
            "evidence_count": sum(len(case["evidence"]) for case in node_cases),
        })
    taxonomy = {
        "taxonomy_version": version, "status": "analysis_snapshot", "generated_at": now.isoformat(),
        "construction_method": "deduplicated_instance_pool_with_existing_taxonomy_matching",
        "base_taxonomy": str(existing_taxonomy.resolve()) if existing_taxonomy else None,
        "warning": "这是当前数据支持下的分析快照；案例数、Trace 数和证据数用于判断结论强弱。",
        "root": {"capability_id": "capability_root", "name": "Coding Agent 能力", "children": nodes},
    }

    taxonomy_path = run_dir / "capability_taxonomy.json"
    cases_path = run_dir / "capability_cases.jsonl"
    distribution_path = run_dir / "capability_distribution.csv"
    taxonomy_path.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    fields = ["capability_id", "parent_id", "level", "capability_path", "name", "is_leaf",
              "status", "case_count", "trace_count", "evidence_count"]
    with distribution_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in nodes:
            writer.writerow({**{key: node[key] for key in fields if key != "capability_path"},
                             "capability_path": " > ".join(node["path"])})
    registry_record = write_immutable_record(
        "analysis-runs", version,
        {"taxonomy_version": version, "status": "analysis_snapshot", "stage": "candidate_core",
         "case_count": len(cases), "node_count": len(nodes),
         "input_feature_files": [str(path.resolve()) for path in feature_paths],
         "output_path": str(run_dir.resolve())},
        artifacts=(taxonomy_path, cases_path, distribution_path),
        root=registry_root_for_output(output_root),
    )
    return {"taxonomy_version": version, "status": "analysis_snapshot", "case_count": len(cases),
            "candidate_capability_count": len(nodes),
            "matched_existing_count": sum(node["status"] == "matched_existing" for node in nodes),
            "review_possible_match_count": sum(node["status"] == "review_possible_match" for node in nodes),
            "new_candidate_count": sum(node["status"] == "new_candidate" for node in nodes),
            "output_path": str(run_dir.resolve()), "registry_record_path": str(registry_record.resolve())}
