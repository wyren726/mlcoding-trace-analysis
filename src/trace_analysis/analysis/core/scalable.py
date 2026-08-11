from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ...preprocessing.adapters.common import stable_id
from ...registries import registry_root_for_output, write_immutable_record
from .validation import regenerate_distribution_csv, validate_core_snapshot


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_rows(draft_core: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    taxonomy = json.loads((draft_core / "capability_taxonomy.json").read_text(encoding="utf-8"))
    cases = _jsonl(draft_core / "capability_cases.jsonl")
    cases_by_capability: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        cases_by_capability.setdefault(str(case["capability_id"]), []).append(case)
    candidate_nodes = []

    def collect_leaves(node: dict[str, Any]) -> None:
        children = node.get("children") or []
        if children:
            for child in children:
                collect_leaves(child)
        elif node.get("capability_id") != "capability_root":
            candidate_nodes.append(node)

    collect_leaves(taxonomy["root"])
    rows = []
    for node in candidate_nodes:
        capability_id = str(node["capability_id"])
        examples = cases_by_capability.get(capability_id, [])[:3]
        rows.append({
            "candidate_id": capability_id,
            "name": str(node.get("name") or "待归纳能力"),
            "case_count": len(cases_by_capability.get(capability_id, [])),
            "examples": [str(item.get("user_need") or "")[:240] for item in examples],
        })
    return rows, cases_by_capability


def _cluster(rows: list[dict[str, Any]], target_size: int, max_size: int) -> list[list[dict[str, Any]]]:
    if len(rows) <= max_size:
        return [rows]
    texts = [row["name"] + " " + " ".join(row["examples"]) for row in rows]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                             max_features=60000, sublinear_tf=True).fit_transform(texts)
    cluster_count = max(2, round(len(rows) / target_size))
    labels = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, batch_size=1024,
                             n_init="auto").fit_predict(matrix)
    groups: dict[int, list[dict[str, Any]]] = {}
    for row, label in zip(rows, labels):
        groups.setdefault(int(label), []).append(row)
    result = []
    for group in groups.values():
        if len(group) <= max_size:
            result.append(group)
        else:
            ordered = sorted(group, key=lambda item: item["name"])
            result.extend(ordered[index:index + max_size] for index in range(0, len(ordered), max_size))
    return result


def _validate_groups(value: dict[str, Any], expected: set[str]) -> list[dict[str, Any]]:
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty list")
    assigned = []
    for group in groups:
        if not isinstance(group, dict) or not str(group.get("name") or "").strip():
            raise ValueError("Every group requires a name")
        members = group.get("member_candidate_ids")
        if not isinstance(members, list) or not members:
            raise ValueError("Every group requires member_candidate_ids")
        assigned.extend(str(item) for item in members)
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("Candidate assignment must cover every input exactly once")
    return groups


def induce_taxonomy_scalable(draft_core: Path, output_root: Path, client: Any, provider: str,
                             taxonomy_version: str | None = None, workers: int = 16,
                             target_unit_size: int = 30, max_unit_size: int = 40) -> dict[str, Any]:
    now = dt.datetime.now().astimezone()
    version = taxonomy_version or f"scalable_{now.strftime('%Y%m%d_%H%M%S')}"
    work_dir = output_root / "induction-work" / version
    unit_dir = work_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    candidates, cases_by_capability = _candidate_rows(draft_core)
    units = _cluster(candidates, target_unit_size, max_unit_size)
    plan_path = work_dir / "plan.json"
    plan = {"taxonomy_version": version, "candidate_count": len(candidates),
            "case_count": sum(len(v) for v in cases_by_capability.values()),
            "unit_count": len(units), "model": client.model, "status": "running"}
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    system = (
        "你是科研ML/LLM Coding Agent原子能力归纳器。输入已全部通过科研ML/LLM Coding范围筛选。"
        "返回JSON对象groups。"
        "每个group包含name、definition、inclusion_criteria、exclusion_criteria、member_candidate_ids。"
        "合并真正同义或边界相同的候选；不同可独立验证能力必须分开。所有输入candidate_id必须全部且仅出现一次。"
        "能力名称必须描述Agent在ML/LLM科研编码工作中可执行、可验证、可跨学科迁移的行为，"
        "例如数据管线实现、训练故障诊断、评测代码开发；不得按生物/材料等研究领域、具体数据集、论文主题或项目名分类。"
        "名称使用简洁中文，不按项目名或个别案例机械命名。"
    )

    def process(index: int, unit: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], bool]:
        path = unit_dir / f"unit_{index:04d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return index, saved["groups"], bool(saved.get("fallback"))
        payload = {"candidates": unit}
        expected = {row["candidate_id"] for row in unit}
        try:
            value, meta = client.complete_json(system, json.dumps(payload, ensure_ascii=False))
            groups = _validate_groups(value, expected)
            fallback = False
        except Exception as exc:
            groups = []
            sub_errors = []
            fallback = False
            for offset in range(0, len(unit), 20):
                subset = unit[offset:offset + 20]
                subset_expected = {row["candidate_id"] for row in subset}
                try:
                    sub_value, _ = client.complete_json(
                        system, json.dumps({"candidates": subset}, ensure_ascii=False)
                    )
                    groups.extend(_validate_groups(sub_value, subset_expected))
                except Exception as sub_exc:
                    fallback = True
                    sub_errors.append(str(sub_exc))
                    groups.extend({"name": row["name"],
                                   "definition": row["examples"][0] if row["examples"] else row["name"],
                                   "inclusion_criteria": row["examples"][:2], "exclusion_criteria": [],
                                   "member_candidate_ids": [row["candidate_id"]]} for row in subset)
            _validate_groups({"groups": groups}, expected)
            meta = {"initial_error": str(exc), "subgroup_errors": sub_errors,
                    "recovered_by_subdivision": not fallback}
        path.write_text(json.dumps({"groups": groups, "fallback": fallback, "api_meta": meta},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        return index, groups, fallback

    unit_results: dict[int, list[dict[str, Any]]] = {}
    fallback_units = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process, index, unit) for index, unit in enumerate(units)]
        for completed, future in enumerate(as_completed(futures), 1):
            index, groups, fallback = future.result()
            unit_results[index] = groups
            fallback_units += int(fallback)
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"stage": "leaf_induction", "completed_units": completed,
                                  "total_units": len(futures), "fallback_units": fallback_units},
                                 ensure_ascii=False), flush=True)

    unit_summaries = []
    for index in range(len(units)):
        groups = unit_results[index]
        unit_summaries.append({"unit_id": f"unit_{index:04d}",
                               "leaf_names": [str(group["name"]) for group in groups]})
    parent_groups = _cluster(
        [{"candidate_id": item["unit_id"], "name": " / ".join(item["leaf_names"][:8]),
          "case_count": 0, "examples": item["leaf_names"][:8]} for item in unit_summaries],
        18, 30,
    )
    top_nodes = []
    label_system = ("你是科研ML/LLM Coding Agent能力上层分类命名器。根据一组叶子能力名称返回JSON对象，"
                    "字段name和definition。名称应是跨学科可复用的ML/LLM编码能力，不得使用研究领域、"
                    "具体数据集、论文主题或项目名称；使用简洁中文，不得声称输入中没有的信息。")
    for parent_index, parent in enumerate(parent_groups):
        try:
            label, _ = client.complete_json(label_system, json.dumps({"units": parent}, ensure_ascii=False))
            parent_name = str(label.get("name") or f"能力域{parent_index + 1}")
            parent_definition = str(label.get("definition") or "")
        except Exception:
            parent_name = f"待进一步命名能力域{parent_index + 1}"
            parent_definition = "由语义相近的原子能力组成"
        children = []
        for item in parent:
            unit_index = int(str(item["candidate_id"]).split("_")[-1])
            leaves = []
            for group in unit_results[unit_index]:
                member_ids = [str(value) for value in group["member_candidate_ids"]]
                leaf_cases = [case for member in member_ids for case in cases_by_capability.get(member, [])]
                leaf_name = str(group["name"])
                leaf_id = stable_id("cap", version, leaf_name, sorted(member_ids))
                leaves.append({"capability_id": leaf_id, "parent_id": "", "level": 3,
                               "name": leaf_name, "path": [], "is_leaf": True,
                               "status": "supported" if len(leaf_cases) >= 2 else "provisional",
                               "definition": str(group.get("definition") or ""),
                               "inclusion_criteria": group.get("inclusion_criteria") or [],
                               "exclusion_criteria": group.get("exclusion_criteria") or [],
                               "case_count": len(leaf_cases), "trace_count": len({c.get("trace_id") for c in leaf_cases}),
                               "evidence_count": sum(len(c.get("evidence") or []) for c in leaf_cases),
                               "children": [], "_cases": leaf_cases})
            unit_name = "、".join(leaf["name"] for leaf in leaves[:3])
            unit_id = stable_id("capgroup", version, item["candidate_id"])
            children.append({"capability_id": unit_id, "parent_id": "", "level": 2,
                             "name": unit_name, "path": [], "is_leaf": False,
                             "status": "supported", "definition": "相关原子能力组",
                             "inclusion_criteria": [], "exclusion_criteria": [],
                             "case_count": sum(x["case_count"] for x in leaves),
                             "trace_count": len({c.get("trace_id") for x in leaves for c in x["_cases"]}),
                             "evidence_count": sum(x["evidence_count"] for x in leaves), "children": leaves})
        top_id = stable_id("capgroup", version, parent_name, parent_index)
        top_nodes.append({"capability_id": top_id, "parent_id": "capability_root", "level": 1,
                          "name": parent_name, "path": ["科研 ML/LLM Coding Agent 能力", parent_name], "is_leaf": False,
                          "status": "supported", "definition": parent_definition,
                          "inclusion_criteria": [], "exclusion_criteria": [],
                          "case_count": sum(x["case_count"] for x in children),
                          "trace_count": len({c.get("trace_id") for x in children for leaf in x["children"] for c in leaf["_cases"]}),
                          "evidence_count": sum(x["evidence_count"] for x in children), "children": children})

    assigned_cases = []
    for top in top_nodes:
        for middle in top["children"]:
            middle["parent_id"] = top["capability_id"]
            middle["path"] = top["path"] + [middle["name"]]
            for leaf in middle["children"]:
                leaf["parent_id"] = middle["capability_id"]
                leaf["path"] = middle["path"] + [leaf["name"]]
                for case in leaf.pop("_cases"):
                    updated = dict(case)
                    updated.update({"taxonomy_version": version, "capability_id": leaf["capability_id"]})
                    assigned_cases.append(updated)
    run_dir = output_root / f"version_{version}" / "core"
    run_dir.mkdir(parents=True, exist_ok=False)
    taxonomy = {"taxonomy_version": version, "status": "analysis_snapshot", "generated_at": now.isoformat(),
                "construction_method": "tfidf_neighborhoods_plus_llm_atomic_grouping",
                "base_taxonomy": str((draft_core / "capability_taxonomy.json").resolve()),
                "warning": "自动归纳快照；fallback邻域需后续复核。",
                "generated_by": {"method": "hybrid", "provider": provider, "model": client.model,
                                 "prompt_version": "scalable_bottom_up_v1", "unit_count": len(units),
                                 "fallback_unit_count": fallback_units},
                "root": {"capability_id": "capability_root", "name": "科研 ML/LLM Coding Agent 能力", "children": top_nodes}}
    (run_dir / "capability_taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in assigned_cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    regenerate_distribution_csv(run_dir)
    validation = validate_core_snapshot(run_dir)
    registry_record = write_immutable_record(
        "analysis-runs", version,
        {"taxonomy_version": version, "status": "success", "stage": "taxonomy_induction",
         "parent_taxonomy": str((draft_core / "capability_taxonomy.json").resolve()),
         **validation, "output_path": str(run_dir.resolve())},
        artifacts=(run_dir / "capability_taxonomy.json", run_dir / "capability_cases.jsonl",
                   run_dir / "capability_distribution.csv"),
        root=registry_root_for_output(output_root),
    )
    plan.update({"status": "completed", "fallback_unit_count": fallback_units,
                 "output_path": str(run_dir.resolve())})
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**validation, "status": "success", "output_path": str(run_dir.resolve()),
            "unit_count": len(units), "fallback_unit_count": fallback_units,
            "provider": provider, "model": client.model,
            "registry_record_path": str(registry_record.resolve())}


def refine_taxonomy_labels(core_dir: Path, output_root: Path, client: Any, provider: str,
                           taxonomy_version: str, workers: int = 16) -> dict[str, Any]:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    cases = _jsonl(core_dir / "capability_cases.jsonl")
    root_children = taxonomy["root"].get("children") or []
    middle_nodes = [child for top in root_children for child in top.get("children") or []]
    batches = [middle_nodes[index:index + 20] for index in range(0, len(middle_nodes), 20)]
    system = ("你是Coding Agent能力体系编辑。为每个中层能力组返回更概括、互相可区分的中文名称和一句定义。"
              "返回JSON对象labels，每项包含capability_id、name、definition；必须覆盖全部输入ID且不得修改ID。"
              "不要把多个叶子名称用顿号简单拼接，不要使用‘综合能力’等空泛名称。")

    def label_batch(nodes: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        payload = [{"capability_id": node["capability_id"],
                    "current_name": node["name"],
                    "leaf_names": [leaf["name"] for leaf in (node.get("children") or [])[:15]]}
                   for node in nodes]
        expected = {item["capability_id"] for item in payload}
        try:
            value, _ = client.complete_json(system, json.dumps({"groups": payload}, ensure_ascii=False))
            labels = value.get("labels")
            if not isinstance(labels, list) or {str(x.get("capability_id")) for x in labels} != expected:
                raise ValueError("Middle label coverage mismatch")
            return {str(item["capability_id"]): {"name": str(item.get("name") or ""),
                                                  "definition": str(item.get("definition") or "")}
                    for item in labels}
        except Exception:
            return {str(node["capability_id"]): {"name": str(node["name"]),
                                                  "definition": str(node.get("definition") or "")}
                    for node in nodes}

    labels: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for result in executor.map(label_batch, batches):
            labels.update(result)
    for node in middle_nodes:
        label = labels[str(node["capability_id"])]
        node["name"], node["definition"] = label["name"], label["definition"]

    # The first scalable pass creates temporary semantic buckets above middle nodes.
    # Regroup the meaningful middle nodes directly so placeholder buckets never leak
    # into the published taxonomy.
    top_payload = [{"capability_id": node["capability_id"], "name": node["name"],
                    "leaf_names": [child["name"] for child in (node.get("children") or [])[:20]]}
                   for node in middle_nodes]
    top_system = ("你是科研ML/LLM Coding Agent能力体系架构师。将输入的中层能力归入5到10个互不重叠的上层能力域。"
                  "返回JSON对象groups，每组包含name、definition、member_ids。所有输入capability_id必须全部且仅出现一次。"
                  "名称应是ML/LLM科研编码中清晰稳定且跨学科复用的能力维度，禁止按研究领域、项目或论文主题分类，"
                  "避免‘综合能力’和场景堆砌。")
    value, _ = client.complete_json(top_system, json.dumps({"nodes": top_payload}, ensure_ascii=False))
    groups = value.get("groups")
    expected_top = {str(item["capability_id"]) for item in top_payload}
    assigned = [str(item) for group in groups or [] for item in group.get("member_ids") or []]
    if (not isinstance(groups, list) or not 5 <= len(groups) <= 10
            or len(assigned) != len(set(assigned)) or set(assigned) != expected_top):
        raise ValueError("Top-level refinement did not cover every existing node exactly once")
    lookup = {str(node["capability_id"]): node for node in middle_nodes}
    refined_roots = []
    for index, group in enumerate(groups):
        children = [lookup[str(item)] for item in group["member_ids"]]
        node_id = stable_id("capgroup", taxonomy_version, group["name"], index)
        refined_roots.append({"capability_id": node_id, "parent_id": "capability_root", "level": 1,
                              "name": str(group["name"]), "path": ["科研 ML/LLM Coding Agent 能力", str(group["name"])],
                              "is_leaf": False, "status": "supported",
                              "definition": str(group.get("definition") or ""),
                              "inclusion_criteria": [], "exclusion_criteria": [], "children": children})

    cases_by_capability: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        cases_by_capability.setdefault(str(case["capability_id"]), []).append(case)

    def update(node: dict[str, Any], parent_id: str, path: list[str], level: int) -> list[dict[str, Any]]:
        node["parent_id"], node["path"], node["level"] = parent_id, path + [node["name"]], level
        children = node.get("children") or []
        descendants = ([case for child in children for case in update(child, node["capability_id"], node["path"], level + 1)]
                       if children else cases_by_capability.get(str(node["capability_id"]), []))
        node["case_count"] = len({case["case_id"] for case in descendants})
        node["trace_count"] = len({case.get("trace_id") for case in descendants})
        node["evidence_count"] = sum(len(case.get("evidence") or []) for case in descendants)
        node["is_leaf"] = not bool(children)
        return descendants

    for node in refined_roots:
        update(node, "capability_root", ["科研 ML/LLM Coding Agent 能力"], 1)
    now = dt.datetime.now().astimezone()
    output_dir = output_root / f"version_{taxonomy_version}" / "core"
    output_dir.mkdir(parents=True, exist_ok=False)
    refined = {**taxonomy, "taxonomy_version": taxonomy_version, "generated_at": now.isoformat(),
               "construction_method": taxonomy.get("construction_method") + "+llm_hierarchy_label_refinement",
               "base_taxonomy": str((core_dir / "capability_taxonomy.json").resolve()),
               "root": {"capability_id": "capability_root", "name": "科研 ML/LLM Coding Agent 能力", "children": refined_roots}}
    (output_dir / "capability_taxonomy.json").write_text(json.dumps(refined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            updated = dict(case); updated["taxonomy_version"] = taxonomy_version
            handle.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
    regenerate_distribution_csv(output_dir)
    validation = validate_core_snapshot(output_dir)
    registry_record = write_immutable_record(
        "analysis-runs", taxonomy_version,
        {"taxonomy_version": taxonomy_version, "status": "success", "stage": "hierarchy_refinement",
         "parent_taxonomy": str((core_dir / "capability_taxonomy.json").resolve()),
         **validation, "output_path": str(output_dir.resolve())},
        artifacts=(output_dir / "capability_taxonomy.json", output_dir / "capability_cases.jsonl",
                   output_dir / "capability_distribution.csv"),
        root=registry_root_for_output(output_root),
    )
    return {**validation, "status": "success", "output_path": str(output_dir.resolve()),
            "provider": provider, "model": client.model,
            "registry_record_path": str(registry_record.resolve())}
