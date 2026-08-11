from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def regenerate_distribution_csv(core_dir: Path) -> dict[str, Any]:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    with (core_dir / "capability_cases.jsonl").open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    cases = [case for case in cases if case.get("assignment_status", "published") == "published"]
    direct: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        direct.setdefault(case["capability_id"], []).append(case)
    rows: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], parent_id: str, level: int, path: list[str]) -> list[dict[str, Any]]:
        node_path = path + [node["name"]]
        descendant = list(direct.get(node["capability_id"]) or [])
        for child in node.get("children") or []:
            descendant.extend(visit(child, node["capability_id"], level + 1, node_path))
        rows.append({"capability_id": node["capability_id"], "parent_id": parent_id, "level": level,
                     "capability_path": " > ".join(node_path), "name": node["name"],
                     "is_leaf": not bool(node.get("children")), "status": node.get("status", "unknown"),
                     "case_count": len({case.get("source_case_id", case["case_id"]) for case in descendant}),
                     "query_count": len({turn for case in descendant for turn in case.get("turn_ids") or []}),
                     "trace_count": len({case.get("trace_id") for case in descendant}),
                     "evidence_count": sum(len(case.get("evidence") or []) for case in descendant)})
        return descendant

    root_name = str(taxonomy["root"].get("name") or "Coding Agent 能力")
    for child in taxonomy["root"].get("children") or []:
        visit(child, "capability_root", 1, [root_name])
    rows.sort(key=lambda row: (row["capability_path"], row["capability_id"]))
    output = core_dir / "capability_distribution.csv"
    fields = ["capability_id", "parent_id", "level", "capability_path", "name", "is_leaf",
              "status", "query_count", "case_count", "trace_count", "evidence_count"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "success", "row_count": len(rows), "output_path": str(output.resolve())}


def validate_core_snapshot(core_dir: Path) -> dict[str, Any]:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    with (core_dir / "capability_cases.jsonl").open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("capability_cases.jsonl contains duplicate case_id values")
    published_cases = [case for case in cases if case.get("assignment_status", "published") == "published"]
    by_capability: dict[str, list[dict[str, Any]]] = {}
    for case in published_cases:
        by_capability.setdefault(case["capability_id"], []).append(case)

    nodes: dict[str, dict[str, Any]] = {}
    expected_rows: dict[str, dict[str, Any]] = {}

    def visit(node: dict[str, Any], parent_id: str, level: int, path: list[str]) -> list[dict[str, Any]]:
        node_id = node["capability_id"]
        if node_id in nodes:
            raise ValueError(f"Taxonomy contains duplicate node: {node_id}")
        nodes[node_id] = node
        if node.get("parent_id", parent_id) != parent_id:
            raise ValueError(f"Incorrect parent_id for node: {node_id}")
        children = node.get("children") or []
        if not children and taxonomy.get("status") == "published":
            policy = (taxonomy.get("review") or {}).get("publication_policy") or {}
            if policy.get("require_definition") and not str(node.get("definition") or "").strip():
                raise ValueError(f"Published leaf lacks definition: {node_id}")
            if policy.get("require_boundaries") and (not node.get("inclusion_criteria") or not node.get("exclusion_criteria")):
                raise ValueError(f"Published leaf lacks inclusion/exclusion boundaries: {node_id}")
        if children and by_capability.get(node_id):
            raise ValueError(f"A non-leaf capability directly owns cases: {node_id}")
        descendant_cases = list(by_capability.get(node_id) or [])
        for child in children:
            descendant_cases.extend(visit(child, node_id, level + 1, path + [node["name"]]))
        expected = {
            "parent_id": parent_id, "level": level, "capability_path": " > ".join(path + [node["name"]]),
            "is_leaf": not children,
            "case_count": len({case.get("source_case_id", case["case_id"]) for case in descendant_cases}),
            "trace_count": len({case.get("trace_id") for case in descendant_cases}),
            "evidence_count": sum(len(case.get("evidence") or []) for case in descendant_cases),
        }
        for field in ("level", "is_leaf", "case_count", "trace_count", "evidence_count"):
            if field in node and node[field] != expected[field]:
                raise ValueError(f"Incorrect {field} for node {node_id}: {node[field]} != {expected[field]}")
        expected_rows[node_id] = expected
        return descendant_cases

    root_name = str(taxonomy["root"].get("name") or "Coding Agent 能力")
    for child in taxonomy["root"].get("children") or []:
        visit(child, "capability_root", 1, [root_name])
    unknown_case_nodes = set(by_capability) - set(nodes)
    if unknown_case_nodes:
        raise ValueError(f"Published cases reference unknown capabilities: {sorted(unknown_case_nodes)}")

    with (core_dir / "capability_distribution.csv").open(encoding="utf-8-sig") as handle:
        distribution = {row["capability_id"]: row for row in csv.DictReader(handle)}
    if set(distribution) != set(nodes):
        raise ValueError("Distribution node IDs do not match taxonomy node IDs")
    for node_id, expected in expected_rows.items():
        row = distribution[node_id]
        for field in ("parent_id", "capability_path"):
            if row[field] != str(expected[field]):
                raise ValueError(f"Distribution {field} mismatch for node: {node_id}")
        for field in ("level", "case_count", "trace_count", "evidence_count"):
            if int(row[field]) != expected[field]:
                raise ValueError(f"Distribution {field} mismatch for node: {node_id}")
        if row["is_leaf"].lower() != str(expected["is_leaf"]).lower():
            raise ValueError(f"Distribution is_leaf mismatch for node: {node_id}")
    return {"status": "valid", "taxonomy_version": taxonomy.get("taxonomy_version"),
            "node_count": len(nodes), "leaf_count": sum(value["is_leaf"] for value in expected_rows.values()),
            "case_count": len(cases), "published_case_count": len(published_cases)}
