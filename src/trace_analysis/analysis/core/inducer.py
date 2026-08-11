from __future__ import annotations

import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from ...preprocessing.adapters.common import stable_id
from .validation import regenerate_distribution_csv, validate_core_snapshot


class JSONClient(Protocol):
    model: str
    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]: ...


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def induce_taxonomy(draft_core: Path, output_root: Path, client: JSONClient,
                    provider: str, taxonomy_version: str | None = None) -> dict[str, Any]:
    cases = _jsonl(draft_core / "capability_cases.jsonl")
    if not cases:
        raise ValueError("No capability cases to organize")
    payload = [{"case_id": c["case_id"], "user_need": c.get("user_need"),
                "expected_outcome": c.get("expected_outcome"), "constraints": c.get("constraints"),
                "fulfillment": c.get("fulfillment"), "unmet_need": c.get("unmet_need")}
               for c in cases]
    expected_count = len(payload)
    system = (
        "你是 Coding Agent 能力体系归纳器。根据案例自下而上建立递归能力树。返回JSON对象："
        f'{{"assigned_case_count":{expected_count},"tree":[node]}}。中间node字段为name,definition,children；叶子node字段为name,definition,'
        "inclusion_criteria,exclusion_criteria,case_ids。每个叶子描述一个原子、可独立验证、与兄弟叶子尽量不重叠的能力；"
        f"输入共有{expected_count}个case_id；返回前逐项自检，必须全部且仅出现一次，不得发明ID。父节点只组织，不含case_ids。不要按具体项目名机械命名，"
        "但不要为增加案例数而合并不同能力。证据少的叶子仍保留，后续由代码标记provisional。"
    )
    value, api_meta = client.complete_json(system, json.dumps({"cases": payload}, ensure_ascii=False))
    tree = value.get("tree")
    if not isinstance(tree, list) or not tree:
        raise ValueError("Induced taxonomy must contain a non-empty tree")
    if value.get("assigned_case_count") != expected_count:
        raise ValueError(f"Model self-reported case count mismatch: {value.get('assigned_case_count')} != {expected_count}")
    expected = {str(c["case_id"]) for c in cases}
    assigned: list[str] = []

    def audit(node: dict[str, Any]) -> None:
        name = str(node.get("name") or "").strip()
        if not name:
            raise ValueError("Every taxonomy node requires a name")
        children = node.get("children")
        case_ids = node.get("case_ids")
        if children is not None:
            if not isinstance(children, list) or not children or case_ids is not None:
                raise ValueError(f"Invalid parent node: {name}")
            for child in children:
                if not isinstance(child, dict):
                    raise ValueError(f"Invalid child under: {name}")
                audit(child)
        else:
            if not isinstance(case_ids, list) or not case_ids:
                raise ValueError(f"Leaf has no cases: {name}")
            assigned.extend(str(item) for item in case_ids)

    for item in tree:
        audit(item)
    duplicates = sorted({item for item in assigned if assigned.count(item) > 1})
    missing = sorted(expected - set(assigned))
    extra = sorted(set(assigned) - expected)
    recovered_count = 0
    if missing and not duplicates and not extra and len(missing) <= max(2, len(expected) // 20):
        draft_taxonomy = json.loads((draft_core / "capability_taxonomy.json").read_text(encoding="utf-8"))
        names: dict[str, str] = {}
        def collect_names(node: dict[str, Any]) -> None:
            names[str(node.get("capability_id"))] = str(node.get("name") or "待进一步归纳能力")
            for child in node.get("children") or []:
                collect_names(child)
        collect_names(draft_taxonomy["root"])
        case_lookup = {str(case["case_id"]): case for case in cases}
        for case_id in missing:
            case = case_lookup[case_id]
            tree.append({"name": names.get(str(case.get("capability_id")), str(case.get("user_need"))),
                         "definition": str(case.get("user_need") or "当前数据中的单案例候选能力"),
                         "inclusion_criteria": [str(case.get("expected_outcome") or case.get("user_need"))],
                         "exclusion_criteria": [], "case_ids": [case_id]})
            assigned.append(case_id)
            recovered_count += 1
        missing = []
    if duplicates or missing or extra:
        raise ValueError(f"Case assignment mismatch; missing={sorted(expected-set(assigned))}, "
                         f"extra={sorted(set(assigned)-expected)}, duplicates={duplicates}")

    now = dt.datetime.now().astimezone()
    version = taxonomy_version or f"induced_{now.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / f"version_{version}" / "core"
    run_dir.mkdir(parents=True, exist_ok=False)
    case_by_id = {str(c["case_id"]): c for c in cases}
    assigned_cases: list[dict[str, Any]] = []

    def build(node: dict[str, Any], parent_id: str, path: list[str], level: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        name = str(node["name"]).strip()
        children_raw = node.get("children")
        node_id = stable_id("capgroup" if children_raw is not None else "cap", " > ".join(path+[name]))
        child_results = ([build(child, node_id, path+[name], level+1) for child in children_raw]
                         if children_raw is not None else [])
        direct = []
        if children_raw is None:
            for case_id in node["case_ids"]:
                case = dict(case_by_id[str(case_id)])
                case.update({"taxonomy_version": version, "capability_id": node_id})
                direct.append(case); assigned_cases.append(case)
        descendants = direct + [case for _, rows in child_results for case in rows]
        result = {"capability_id": node_id, "parent_id": parent_id, "level": level, "name": name,
                  "path": path+[name], "is_leaf": children_raw is None,
                  "status": "supported" if len(descendants) >= 2 else "provisional",
                  "definition": str(node.get("definition") or ""),
                  "inclusion_criteria": node.get("inclusion_criteria") or [],
                  "exclusion_criteria": node.get("exclusion_criteria") or [],
                  "case_count": len({c["case_id"] for c in descendants}),
                  "trace_count": len({c.get("trace_id") for c in descendants}),
                  "evidence_count": sum(len(c.get("evidence") or []) for c in descendants),
                  "children": [child for child, _ in child_results]}
        return result, descendants

    roots = [build(item, "capability_root", ["Coding Agent 能力"], 1)[0] for item in tree]
    taxonomy = {"taxonomy_version": version, "status": "analysis_snapshot", "generated_at": now.isoformat(),
                "construction_method": "llm_bottom_up_case_partition_with_deterministic_validation",
                "base_taxonomy": str((draft_core / 'capability_taxonomy.json').resolve()),
                "warning": "当前数据支持下的自动归纳快照；单案例叶子标记为 provisional。",
                "generated_by": {"method": "llm", "provider": provider, "model": client.model,
                                 "prompt_version": "bottom_up_v2", "api_meta": api_meta,
                                 "deterministically_recovered_case_count": recovered_count},
                "root": {"capability_id": "capability_root", "name": "Coding Agent 能力", "children": roots}}
    (run_dir / "capability_taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (run_dir / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in assigned_cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":"))+"\n")
    regenerate_distribution_csv(run_dir)
    validation = validate_core_snapshot(run_dir)
    return {**validation, "status": "success", "output_path": str(run_dir.resolve()),
            "provider": provider, "model": client.model}
