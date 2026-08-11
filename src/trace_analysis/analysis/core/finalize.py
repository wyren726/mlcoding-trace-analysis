from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...preprocessing.adapters.common import stable_id
from .validation import regenerate_distribution_csv, validate_core_snapshot


def finalize_taxonomy(core_dir: Path, corrections_path: Path, output_root: Path,
                      taxonomy_version: str) -> dict[str, Any]:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (core_dir / "capability_cases.jsonl").open(encoding="utf-8")
             if line.strip()]
    renames = corrections.get("renames") or {}
    splits = corrections.get("splits") or {}
    regroup = corrections.get("regroup") or {}
    split_targets: dict[str, list[str]] = {}

    def transform(node: dict[str, Any]) -> dict[str, Any]:
        node = dict(node)
        node_id = str(node["capability_id"])
        if node_id in renames:
            node.update(renames[node_id])
        if node_id in splits:
            raise ValueError("Split leaves are handled by their parent")
        children = []
        for child in node.get("children") or []:
            child_id = str(child["capability_id"])
            if child_id not in splits:
                children.append(transform(child))
                continue
            replacements = []
            for spec in splits[child_id]:
                replacement_id = stable_id("cap", taxonomy_version, spec["name"], child_id)
                replacements.append({"capability_id": replacement_id, **spec, "children": [],
                                     "is_leaf": True, "status": "supported"})
            split_targets[child_id] = [row["capability_id"] for row in replacements]
            children.extend(replacements)
        node["children"] = children
        if node_id in regroup:
            lookup = {str(child["capability_id"]): child for child in children}
            expected = set(lookup)
            assigned = [str(member) for group in regroup[node_id]
                        for member in group.get("member_ids") or []]
            if len(assigned) != len(set(assigned)) or set(assigned) != expected:
                raise ValueError(f"Regroup must cover every child exactly once: {node_id}")
            grouped = []
            for index, group in enumerate(regroup[node_id]):
                members = [lookup[str(member)] for member in group["member_ids"]]
                if len(members) == 1:
                    grouped.extend(members)
                else:
                    grouped.append({"capability_id": stable_id("capgroup", taxonomy_version, node_id,
                                                               group["name"], index),
                                    "name": group["name"], "definition": group["definition"],
                                    "children": members, "is_leaf": False, "status": "supported"})
            node["children"] = grouped
        return node

    root = dict(taxonomy["root"])
    root["children"] = [transform(child) for child in root.get("children") or []]
    rewritten_cases = []
    for case in cases:
        old_id = str(case["capability_id"])
        targets = split_targets.get(old_id, [old_id])
        for target_id in targets:
            row = dict(case)
            source_case_id = str(row.get("source_case_id") or row["case_id"])
            row.update({"source_case_id": source_case_id, "capability_id": target_id,
                        "case_id": stable_id("capcase", source_case_id, target_id),
                        "taxonomy_version": taxonomy_version})
            rewritten_cases.append(row)
    direct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in rewritten_cases:
        direct[str(case["capability_id"])].append(case)

    def update(node: dict[str, Any], parent_id: str, path: list[str], level: int) -> list[dict[str, Any]]:
        node["parent_id"], node["path"], node["level"] = parent_id, path + [node["name"]], level
        children = node.get("children") or []
        descendant = list(direct.get(str(node["capability_id"])) or [])
        for child in children:
            descendant.extend(update(child, node["capability_id"], node["path"], level + 1))
        node["is_leaf"] = not bool(children)
        node["status"] = "supported"
        node["case_count"] = len({row.get("source_case_id", row["case_id"]) for row in descendant})
        node["query_count"] = len({turn for row in descendant for turn in row.get("turn_ids") or []})
        node["trace_count"] = len({row.get("trace_id") for row in descendant})
        node["evidence_count"] = sum(len(row.get("evidence") or []) for row in descendant)
        return descendant

    for child in root["children"]:
        update(child, "capability_root", [root["name"]], 1)
    out = output_root / f"version_{taxonomy_version}" / "core"
    out.mkdir(parents=True, exist_ok=False)
    result_taxonomy = {**taxonomy, "taxonomy_version": taxonomy_version,
                       "generated_at": dt.datetime.now().astimezone().isoformat(),
                       "construction_method": taxonomy["construction_method"] + "+traceable_quality_corrections",
                       "base_taxonomy": str((core_dir / "capability_taxonomy.json").resolve()),
                       "quality_corrections": str(corrections_path.resolve()), "root": root}
    (out / "capability_taxonomy.json").write_text(
        json.dumps(result_taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in rewritten_cases:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    source_excluded = core_dir / "excluded_domain_tasks.jsonl"
    if source_excluded.exists():
        (out / source_excluded.name).write_bytes(source_excluded.read_bytes())
    regenerate_distribution_csv(out)
    validation = validate_core_snapshot(out)
    return {**validation, "status": "success", "output_path": str(out.resolve()),
            "corrections_path": str(corrections_path.resolve()),
            "rename_count": len(renames), "split_count": len(splits), "regroup_count": len(regroup)}
