from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...preprocessing.adapters.common import stable_id


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _flatten(root: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    nodes: dict[str, dict[str, Any]] = {}
    parents: dict[str, str] = {}

    def visit(node: dict[str, Any], parent_id: str) -> None:
        node_id = node["capability_id"]
        if node_id in nodes:
            raise ValueError(f"Capability appears more than once in taxonomy: {node_id}")
        clean = {key: value for key, value in node.items() if key not in {"children", "path", "level", "is_leaf"}}
        nodes[node_id] = clean
        parents[node_id] = parent_id
        for child in node.get("children") or []:
            visit(child, node_id)

    for child in root.get("children") or []:
        visit(child, "capability_root")
    return nodes, parents


def organize_hierarchy(core_dir: Path, operations_path: Path, output_root: Path,
                       taxonomy_version: str) -> dict[str, Any]:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    cases = _jsonl(core_dir / "capability_cases.jsonl")
    change_set = json.loads(operations_path.read_text(encoding="utf-8"))
    reviewer = str(change_set.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("reviewer is required")
    nodes, parents = _flatten(taxonomy["root"])
    audit_operations = []
    for operation in change_set.get("operations") or []:
        action = operation.get("action")
        reason = str(operation.get("reason") or "").strip()
        if not reason:
            raise ValueError("Every hierarchy operation requires a reason")
        if action == "create_parent":
            name = str(operation.get("name") or "").strip()
            if not name:
                raise ValueError("create_parent requires a name")
            node_id = operation.get("capability_id") or stable_id("capgroup", name)
            if node_id in nodes or node_id == "capability_root":
                raise ValueError(f"Capability already exists: {node_id}")
            parent_id = operation.get("parent_id") or "capability_root"
            if parent_id != "capability_root" and parent_id not in nodes:
                raise ValueError(f"Unknown parent capability: {parent_id}")
            child_ids = operation.get("child_ids") or []
            if not child_ids:
                raise ValueError("create_parent requires child_ids")
            unknown = set(child_ids) - set(nodes)
            if unknown:
                raise ValueError(f"Unknown child capabilities: {sorted(unknown)}")
            nodes[node_id] = {"capability_id": node_id, "name": name, "status": "published",
                              "definition": operation.get("definition") or "",
                              "inclusion_criteria": operation.get("inclusion_criteria") or [],
                              "exclusion_criteria": operation.get("exclusion_criteria") or []}
            parents[node_id] = parent_id
            for child_id in child_ids:
                parents[child_id] = node_id
        elif action == "move":
            node_id = operation.get("capability_id")
            parent_id = operation.get("parent_id") or "capability_root"
            if node_id not in nodes:
                raise ValueError(f"Unknown capability to move: {node_id}")
            if parent_id != "capability_root" and parent_id not in nodes:
                raise ValueError(f"Unknown target parent: {parent_id}")
            parents[node_id] = parent_id
        elif action == "rename":
            node_id = operation.get("capability_id")
            name = str(operation.get("name") or "").strip()
            if node_id not in nodes or not name:
                raise ValueError("rename requires an existing capability_id and non-empty name")
            nodes[node_id]["name"] = name
        else:
            raise ValueError(f"Unsupported hierarchy action: {action}")
        audit_operations.append(operation)

    for node_id in nodes:
        seen = {node_id}
        current = parents[node_id]
        while current != "capability_root":
            if current not in nodes:
                raise ValueError(f"Capability has unknown parent: {node_id} -> {current}")
            if current in seen:
                raise ValueError(f"Hierarchy cycle detected involving: {current}")
            seen.add(current)
            current = parents[current]

    children: dict[str, list[str]] = defaultdict(list)
    for node_id, parent_id in parents.items():
        children[parent_id].append(node_id)
    published_cases = [case for case in cases if case.get("assignment_status", "published") == "published"]
    direct_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in published_cases:
        capability_id = case.get("capability_id")
        if capability_id not in nodes:
            raise ValueError(f"Published case references unknown capability: {case.get('case_id')} -> {capability_id}")
        direct_cases[capability_id].append(case)
    for node_id in nodes:
        if children[node_id] and direct_cases[node_id]:
            raise ValueError(f"Cases may only belong to leaf capabilities: {node_id}")

    def build(node_id: str, path: list[str], level: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        node = dict(nodes[node_id])
        child_results = [build(child_id, path + [node["name"]], level + 1)
                         for child_id in sorted(children[node_id], key=lambda item: nodes[item]["name"])]
        descendant_cases = direct_cases[node_id] + [case for _, child_cases in child_results for case in child_cases]
        node.update({"parent_id": parents[node_id], "level": level, "path": path + [node["name"]],
                     "is_leaf": not child_results, "case_count": len({case["case_id"] for case in descendant_cases}),
                     "trace_count": len({case.get("trace_id") for case in descendant_cases}),
                     "evidence_count": sum(len(case.get("evidence") or []) for case in descendant_cases),
                     "children": [child for child, _ in child_results]})
        return node, descendant_cases

    roots = [build(node_id, ["Coding Agent 能力"], 1)[0]
             for node_id in sorted(children["capability_root"], key=lambda item: nodes[item]["name"])]
    now = dt.datetime.now().astimezone()
    output_taxonomy = {
        **{key: value for key, value in taxonomy.items() if key not in {"root", "taxonomy_version", "generated_at"}},
        "taxonomy_version": taxonomy_version, "status": "published", "generated_at": now.isoformat(),
        "base_taxonomy_version": taxonomy.get("taxonomy_version"),
        "root": {"capability_id": "capability_root", "name": "Coding Agent 能力", "children": roots},
    }
    for case in cases:
        case["taxonomy_version"] = taxonomy_version
    run_dir = output_root / f"version_{taxonomy_version}" / "core"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "capability_taxonomy.json").write_text(
        json.dumps(output_taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    for filename in ("capability_candidate_cases.jsonl", "capability_rejected_cases.jsonl"):
        source = core_dir / filename
        if source.exists():
            with source.open(encoding="utf-8") as input_handle, (run_dir / filename).open("w", encoding="utf-8") as output_handle:
                for line in input_handle:
                    if line.strip():
                        record = json.loads(line)
                        record["taxonomy_version"] = taxonomy_version
                        output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    if (core_dir / "review_audit.json").exists():
        shutil.copyfile(core_dir / "review_audit.json", run_dir / "review_audit.json")
    fields = ["capability_id", "parent_id", "level", "capability_path", "name", "is_leaf",
              "status", "case_count", "trace_count", "evidence_count"]
    flat_output: list[dict[str, Any]] = []

    def collect(node: dict[str, Any]) -> None:
        flat_output.append(node)
        for child in node["children"]:
            collect(child)

    for root_node in roots:
        collect(root_node)
    with (run_dir / "capability_distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in flat_output:
            writer.writerow({**{key: node[key] for key in fields if key != "capability_path"},
                             "capability_path": " > ".join(node["path"])})
    audit = {**change_set, "source_operations_path": str(operations_path.resolve()),
             "base_taxonomy_version": taxonomy.get("taxonomy_version"), "published_at": now.isoformat()}
    (run_dir / "hierarchy_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"taxonomy_version": taxonomy_version, "status": "published", "node_count": len(nodes),
            "leaf_count": sum(not children[node_id] for node_id in nodes),
            "max_level": max((node["level"] for node in flat_output), default=0),
            "case_count": len(published_cases), "output_path": str(run_dir.resolve())}
