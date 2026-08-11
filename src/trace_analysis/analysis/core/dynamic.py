from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ...preprocessing.adapters.common import stable_id
from .validation import regenerate_distribution_csv, validate_core_snapshot


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalized_name(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def _validate_atomic(value: dict[str, Any], expected: dict[str, set[str]]) -> list[dict[str, Any]]:
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Atomic response requires decisions")
    ids = [str(row.get("case_id")) for row in decisions]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Atomic decisions must cover every case exactly once")
    for row in decisions:
        status = row.get("status")
        capabilities = row.get("capabilities")
        if status not in {"included", "excluded"}:
            raise ValueError("Atomic decision status must be included or excluded")
        if status == "included" and (not isinstance(capabilities, list) or not capabilities):
            raise ValueError("Included case requires at least one capability")
        if status == "excluded" and capabilities:
            raise ValueError("Excluded case cannot have capabilities")
        for capability in capabilities or []:
            if not all(str(capability.get(key) or "").strip()
                       for key in ("name", "definition", "inclusion_criteria", "exclusion_criteria")):
                raise ValueError("Every atomic capability requires a name, definition and boundaries")
            support_type = capability.get("support_type")
            supporting_ids = [str(value) for value in capability.get("supporting_event_ids") or []]
            if support_type not in {"direct", "necessary", "negotiated"} or not supporting_ids:
                raise ValueError("Atomic capability requires a valid support_type and supporting_event_ids")
            if not set(supporting_ids).issubset(expected[str(row["case_id"])]):
                raise ValueError("Atomic capability cites evidence outside its source case")
    return decisions


def _validate_merge(value: dict[str, Any], expected: set[str]) -> list[dict[str, Any]]:
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Consolidation response requires groups")
    members = [str(member) for group in groups for member in group.get("member_ids") or []]
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("Consolidation must cover every proposal exactly once")
    for group in groups:
        if not all(str(group.get(key) or "").strip()
                   for key in ("name", "definition", "inclusion_criteria", "exclusion_criteria")):
            raise ValueError("Every canonical capability requires a name, definition and boundaries")
    # Models sometimes return separate groups with exactly the same canonical name.
    # Coalesce these deterministically so global uniqueness does not depend on another
    # non-deterministic retry.
    coalesced: dict[str, dict[str, Any]] = {}
    for group in groups:
        key = _normalized_name(str(group["name"]))
        if key not in coalesced:
            coalesced[key] = {**group, "member_ids": list(group["member_ids"])}
            continue
        target = coalesced[key]
        target["member_ids"].extend(group["member_ids"])
        for field in ("definition", "inclusion_criteria", "exclusion_criteria"):
            values = [str(target[field]).strip(), str(group[field]).strip()]
            target[field] = "；".join(dict.fromkeys(value for value in values if value))
    return list(coalesced.values())


def _validate_hierarchy(value: dict[str, Any], expected: set[str], max_children: int) -> list[dict[str, Any]]:
    groups = value.get("groups")
    if (not isinstance(groups, list) or not 2 <= len(groups) < len(expected)
            or any(not isinstance(group, dict) for group in groups)):
        raise ValueError("Hierarchy requires at least two proper groups")
    members = [str(member) for group in groups for member in group.get("member_ids") or []]
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("Hierarchy grouping must cover every member exactly once")
    if any(not str(group.get("name") or "").strip() for group in groups):
        raise ValueError("Hierarchy group requires a name")
    sizes = [len(group.get("member_ids") or []) for group in groups]
    if any(size == 0 or size >= len(expected) for size in sizes):
        raise ValueError("Hierarchy groups must be non-empty proper subsets")
    return groups


def _complete_validated(client: Any, system: str, payload: dict[str, Any], validator: Any,
                        attempts: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retry schema-invalid model output with a changed prompt (and therefore cache key)."""
    errors = []
    for attempt in range(attempts):
        retry_note = ("" if attempt == 0 else
                      f"\n这是第{attempt + 1}次格式纠错重试。此前错误：{errors[-1]}。"
                      "只返回规定的JSON对象，不要解释，不得遗漏顶层字段或输入ID。")
        value, meta = client.complete_json(system + retry_note,
                                           json.dumps(payload, ensure_ascii=False))
        try:
            return validator(value), {**meta, "schema_retry_count": attempt}
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(f"Model output remained schema-invalid after {attempts} attempts: {errors}")


def _complete_hierarchy(client: Any, system: str, payload: list[dict[str, Any]],
                        max_children: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build groups and repair IDs omitted by a large model response in small batches."""
    groups = None
    meta: dict[str, Any] = {}
    structure_errors = []
    for attempt in range(3):
        retry_note = ("" if attempt == 0 else
                      f"\n这是第{attempt + 1}次结构纠错重试。此前错误：{structure_errors[-1]}。"
                      "只返回JSON对象groups；每个group必须是含name、definition、member_ids的对象。")
        value, meta = client.complete_json(
            system + retry_note, json.dumps({"capabilities": payload}, ensure_ascii=False))
        candidate = value.get("groups")
        if (isinstance(candidate, list) and len(candidate) >= 2
                and all(isinstance(group, dict) for group in candidate)):
            groups = candidate
            meta = {**meta, "structure_retry_count": attempt}
            break
        structure_errors.append("groups不是至少包含两个对象的数组")
    if groups is None:
        raise ValueError(f"Hierarchy response remained structurally invalid: {structure_errors}")
    allowed = {str(item["id"]) for item in payload}
    seen: set[str] = set()
    for group in groups:
        if not str(group.get("name") or "").strip():
            raise ValueError("Hierarchy group requires a name")
        cleaned = []
        for member in group.get("member_ids") or []:
            member_id = str(member)
            if member_id in allowed and member_id not in seen:
                cleaned.append(member_id)
                seen.add(member_id)
        group["member_ids"] = cleaned
    groups = [group for group in groups if group["member_ids"]]
    missing = sorted(allowed - seen)
    if missing:
        lookup = {str(item["id"]): item for item in payload}
        repair_system = (
            "你是能力分组遗漏项修复器。为每个missing能力选择唯一最合适的现有group_index。"
            "返回JSON对象assignments，每项只有id、group_index；必须覆盖全部missing id，不能返回其他id。"
        )
        group_summaries = [{"group_index": index, "name": group["name"],
                            "definition": group.get("definition") or ""}
                           for index, group in enumerate(groups)]
        for offset in range(0, len(missing), 24):
            batch_ids = missing[offset:offset + 24]
            repair_payload = {"groups": group_summaries,
                              "missing": [lookup[item] for item in batch_ids]}

            def validate_repair(row: dict[str, Any]) -> list[dict[str, Any]]:
                assignments = row.get("assignments")
                if not isinstance(assignments, list):
                    raise ValueError("Repair response requires assignments")
                ids = [str(item.get("id")) for item in assignments]
                if len(ids) != len(set(ids)) or set(ids) != set(batch_ids):
                    raise ValueError("Repair must cover every missing ID exactly once")
                if any(not isinstance(item.get("group_index"), int)
                       or not 0 <= item["group_index"] < len(groups) for item in assignments):
                    raise ValueError("Repair returned invalid group_index")
                return assignments

            assignments, _ = _complete_validated(client, repair_system, repair_payload, validate_repair)
            for assignment in assignments:
                groups[int(assignment["group_index"])]["member_ids"].append(str(assignment["id"]))
    return _validate_hierarchy({"groups": groups}, allowed, max_children), {
        **meta, "repaired_missing_count": len(missing)}


def build_dynamic_taxonomy(core_dir: Path, output_root: Path, client: Any, provider: str,
                           taxonomy_version: str, max_children: int = 8,
                           workers: int = 16, batch_size: int = 10,
                           negotiation_feature_paths: list[Path] | None = None) -> dict[str, Any]:
    """Build a reusable, atomic Agent capability taxonomy from grounded demand cases.

    A source case may support multiple canonical leaves. Domain algorithms and research
    topics are treated as case context, never as capability names or hierarchy axes.
    """
    source_taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    source_cases = _jsonl(core_dir / "capability_cases.jsonl")
    for path in negotiation_feature_paths or []:
        for record in _jsonl(path):
            values = record.get("values") or {}
            items = {str(item.get("item_id")): item for item in values.get("plan_items") or []}
            for response in values.get("responses") or []:
                if response.get("response_type") not in {
                        "explicit_acceptance", "behavioral_acceptance", "partial_acceptance", "modified"}:
                    continue
                for item_id in response.get("accepted_item_ids") or []:
                    item = items.get(str(item_id))
                    if not item:
                        continue
                    case_id = stable_id("negotiatedcase", item_id, response.get("response_id"))
                    evidence = ([{**row, "evidence_type": "negotiation_proposal"}
                                 for row in item.get("proposal_evidence") or []] +
                                [{**row, "evidence_type": "requirement_acceptance"}
                                 for row in response.get("response_evidence") or []])
                    source_cases.append({
                        "taxonomy_version": taxonomy_version, "capability_id": "negotiated_requirement",
                        "case_id": case_id, "trace_id": record.get("trace_id"),
                        "session_id": record.get("session_id"), "turn_ids": [record.get("turn_id")],
                        "user_need": item.get("description"), "expected_outcome": item.get("description"),
                        "constraints": [], "fulfillment": "unclear", "unmet_need": "",
                        "evidence": evidence, "source_features": [{"feature_run_id": record.get("feature_run_id"),
                            "feature_version": record.get("feature_version"),
                            "generated_by": record.get("generated_by")}],
                        "requirement_origin": response.get("response_type"),
                        "negotiation_item_id": item_id, "negotiation_response_id": response.get("response_id"),
                    })
    source_nodes: dict[str, dict[str, Any]] = {}

    def collect_source(node: dict[str, Any]) -> None:
        source_nodes[str(node.get("capability_id"))] = node
        for child in node.get("children") or []:
            collect_source(child)

    collect_source(source_taxonomy["root"])
    work_dir = output_root / "taxonomy-work" / taxonomy_version
    atomic_dir = work_dir / "atomic_batches"
    atomic_dir.mkdir(parents=True, exist_ok=True)

    atomic_system = (
        "你是科研ML/LLM Coding Agent原子能力分析器。分析用户Query真正要求Agent执行的工作行为，而不是研究主题。"
        "每个case必须返回一个decision：case_id、status、reason、capabilities。"
        "只有跨论文、模型、数据集和学科仍可复用的Coding Agent行为才是capability；流匹配、扩散模型、动力学、"
        "某种生物/材料方法等只可作为案例对象，不得成为能力分类轴或叶子名称。若用户明确要求实现某领域算法，"
        "应转换为从研究描述提炼步骤、把数学机制实现为代码、设计验证实验、诊断实现偏差等被原文支持的通用行为；"
        "不得补出用户未要求的行为。宽泛目标不能按常见工作流展开：例如‘启动论文复现’不能推出下载数据、安装依赖、"
        "实现算法或运行实验。只有evidence直接表达、逻辑上不可分割地必要，或完整的Agent提议+用户接受证据链才能提取。"
        "纯理论创新、领域问题求解、没有编码行为或无法落到原子能力的宽泛目标status=excluded。"
        "capabilities允许多个，且必须拆成互不重叠、可独立执行和验证的原子行为。禁止使用“论文到代码生成”、"
        "“模型开发”、“算法设计与创新”等宽泛叶子。每项capability包含name、definition、inclusion_criteria、"
        "exclusion_criteria、support_type、supporting_event_ids；support_type只能direct、necessary、negotiated，"
        "supporting_event_ids必须引用输入evidence。四个文本字段都必须非空。名称采用“操作+对象+可验证目标”的简洁中文。"
        "返回JSON对象decisions，覆盖全部case_id且每个case只出现一次。"
    )
    payload_cases = []
    for case in source_cases:
        old = source_nodes.get(str(case.get("capability_id")), {})
        payload_cases.append({
            "case_id": str(case["case_id"]), "user_need": case.get("user_need"),
            "expected_outcome": case.get("expected_outcome"), "constraints": case.get("constraints") or [],
            "previous_label": old.get("name"), "previous_definition": old.get("definition"),
            "requirement_origin": case.get("requirement_origin") or "direct_user_requirement",
            "evidence": [{"event_id": row.get("event_id"), "quote": row.get("quote"),
                          "evidence_type": row.get("evidence_type")}
                         for row in case.get("evidence") or []
                         if row.get("evidence_type") in {"requirement", "negotiation_proposal",
                                                         "requirement_acceptance"}],
        })
    batches = [payload_cases[index:index + max(1, batch_size)]
               for index in range(0, len(payload_cases), max(1, batch_size))]

    def atomize(index: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        checkpoint = atomic_dir / f"batch_{index:04d}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            expected = {row["case_id"]: {str(e["event_id"]) for e in row.get("evidence") or []}
                        for row in batch}
            return index, _validate_atomic(saved, expected)
        expected = {row["case_id"]: {str(e["event_id"]) for e in row.get("evidence") or []}
                    for row in batch}
        decisions, meta = _complete_validated(
            client, atomic_system, {"cases": batch}, lambda value: _validate_atomic(value, expected))
        checkpoint.write_text(json.dumps({"decisions": decisions, "api_meta": meta}, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        return index, decisions

    atomic_results: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(atomize, index, batch) for index, batch in enumerate(batches)]
        for completed, future in enumerate(as_completed(futures), 1):
            index, decisions = future.result()
            atomic_results[index] = decisions
            if completed % 10 == 0 or completed == len(futures):
                print(json.dumps({"stage": "agent_capability_atomization", "completed_batches": completed,
                                  "total_batches": len(futures)}, ensure_ascii=False), flush=True)
    decisions = [row for index in range(len(batches)) for row in atomic_results[index]]
    decision_by_case = {str(row["case_id"]): row for row in decisions}

    proposals: list[dict[str, Any]] = []
    for decision in decisions:
        for position, capability in enumerate(decision.get("capabilities") or []):
            proposal_id = stable_id("capprop", decision["case_id"], position, capability["name"])
            proposals.append({"proposal_id": proposal_id, "source_case_id": str(decision["case_id"]), **capability})
    if not proposals:
        raise ValueError("No reusable Agent capability was extracted")

    # Independent evidence-entailment audit: atomicization proposes capabilities,
    # this stage decides whether each case is actually allowed to support them.
    entailment_dir = work_dir / "assignment_entailment"
    entailment_dir.mkdir(exist_ok=True)
    source_lookup = {str(case["case_id"]): case for case in source_cases}
    entailment_system = (
        "你是案例—能力证据蕴含审核器。只判断输入evidence是否足以支持该原子能力，返回JSON对象decisions，"
        "每项包含proposal_id、supported、reason。不能依据常见工作流、领域常识或Agent未被接受的建议外推。"
        "宽泛目标如‘启动论文复现’不能支持下载数据、安装依赖、实现算法或运行实验。direct需要用户原文直接表达；"
        "necessary必须是不执行就逻辑上无法完成原要求的不可分割行为；negotiated必须同时有Agent计划原文和后续用户接受原文。"
        "必须覆盖全部proposal_id且每项恰好一次。"
    )
    entailment_batches = [proposals[index:index + 30] for index in range(0, len(proposals), 30)]

    def audit_assignment(index: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        checkpoint = entailment_dir / f"batch_{index:04d}.json"
        expected = {str(row["proposal_id"]) for row in batch}

        def validate_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
            rows = value.get("decisions")
            ids = [str(row.get("proposal_id")) for row in rows or [] if isinstance(row, dict)]
            if not isinstance(rows, list) or len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError("Entailment audit must cover every proposal exactly once")
            if any(not isinstance(row.get("supported"), bool) for row in rows):
                raise ValueError("Entailment supported must be boolean")
            return rows

        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            return index, validate_rows(saved)
        payload = {"assignments": []}
        for proposal in batch:
            case = source_lookup[proposal["source_case_id"]]
            payload["assignments"].append({
                "proposal_id": proposal["proposal_id"], "capability": proposal["name"],
                "definition": proposal["definition"], "support_type": proposal.get("support_type"),
                "requirement_origin": case.get("requirement_origin") or "direct_user_requirement",
                "user_need": case.get("user_need"),
                "evidence": [{"event_id": row.get("event_id"), "quote": row.get("quote"),
                              "evidence_type": row.get("evidence_type")} for row in case.get("evidence") or []
                             if row.get("evidence_type") in {"requirement", "negotiation_proposal",
                                                             "requirement_acceptance"}],
            })
        rows, meta = _complete_validated(client, entailment_system, payload, validate_rows)
        checkpoint.write_text(json.dumps({"decisions": rows, "api_meta": meta}, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        return index, rows

    entailment_results: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(audit_assignment, index, batch)
                   for index, batch in enumerate(entailment_batches)]
        for future in as_completed(futures):
            index, rows = future.result()
            entailment_results[index] = rows
    allowed_proposals = {str(row["proposal_id"])
                         for index in range(len(entailment_batches))
                         for row in entailment_results[index] if row.get("supported") is True}
    proposals = [row for row in proposals if row["proposal_id"] in allowed_proposals]
    if not proposals:
        raise ValueError("All proposed case-capability assignments failed evidence-entailment audit")

    # One global consolidation call is intentional: local-only clustering caused the
    # duplicate and cross-branch errors in the previous taxonomy.
    merge_system = (
        "你是科研ML/LLM Coding Agent能力体系的全局边界审核器。输入是全部原子能力候选。返回JSON对象groups。"
        "每组包含name、definition、inclusion_criteria、exclusion_criteria、member_ids。必须全局合并同名、同义或"
        "证据边界相同的候选；不得合并只是相关、上下位或通常共同出现的行为。最终能力之间不能同义、范围包含或语义交叉。"
        "名称必须是跨研究主题可复用、可独立执行和验证的Agent行为，采用“操作+对象+可验证目标”；不得出现具体学科、"
        "项目、数据集或算法家族作为分类轴。若候选仍然宽泛，应依据已有定义收窄名称，但不得创造证据外的新行为。"
        "全部proposal_id必须恰好出现一次。四个文本字段必须为非空字符串。"
    )
    merge_payload = [{"id": row["proposal_id"], "name": row["name"], "definition": row["definition"],
                      "inclusion_criteria": row["inclusion_criteria"],
                      "exclusion_criteria": row["exclusion_criteria"]} for row in proposals]
    proposal_ids = {row["proposal_id"] for row in proposals}
    groups, merge_meta = _complete_validated(
        client, merge_system, {"proposals": merge_payload},
        lambda value: _validate_merge(value, proposal_ids))
    proposal_lookup = {row["proposal_id"]: row for row in proposals}
    canonical: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for group in groups:
        normalized = _normalized_name(str(group["name"]))
        if normalized in seen_names:
            raise ValueError(f"Global consolidation returned duplicate capability name: {group['name']}")
        seen_names.add(normalized)
        member_ids = [str(value) for value in group["member_ids"]]
        leaf_id = stable_id("cap", taxonomy_version, normalized, sorted(member_ids))
        case_ids = sorted({proposal_lookup[value]["source_case_id"] for value in member_ids})
        canonical.append({"capability_id": leaf_id, "name": str(group["name"]),
                          "definition": str(group["definition"]),
                          "inclusion_criteria": str(group["inclusion_criteria"]),
                          "exclusion_criteria": str(group["exclusion_criteria"]),
                          "source_case_ids": case_ids, "member_ids": member_ids,
                          "children": [], "is_leaf": True})
    (work_dir / "global_consolidation.json").write_text(
        json.dumps({"groups": groups, "api_meta": merge_meta}, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_system = (
        "你是科研ML/LLM Coding Agent原子能力边界终审员。逐项判断名称是否为跨论文、模型、数据集和学科复用的"
        "Agent工作行为。返回JSON对象decisions，每项包含capability_id、action、reason；action只能keep、rename、exclude。"
        "若名称包含流匹配、扩散模型、动力学、生物/材料方法、具体模型或算法家族等研究对象，不能keep：若定义明确包含"
        "编码行为，rename为原文支持的通用“操作+对象+可验证目标”；若只是理论/领域任务而无编码行为则exclude。"
        "“论文到代码生成”“算法设计”“模型开发”等过宽名称必须rename为可独立执行验证的行为。rename时还必须返回"
        "name、definition、inclusion_criteria、exclusion_criteria四个非空字符串。不得添加证据中没有的行为。"
        "必须覆盖全部capability_id且每项恰好一次。"
    )
    audit_dir = work_dir / "boundary_audit"
    audit_dir.mkdir(exist_ok=True)
    audit_batches = [canonical[index:index + 30] for index in range(0, len(canonical), 30)]

    def audit_batch(index: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        checkpoint = audit_dir / f"batch_{index:04d}.json"
        expected = {str(row["capability_id"]) for row in batch}

        def validate(value: dict[str, Any]) -> list[dict[str, Any]]:
            rows = value.get("decisions")
            if not isinstance(rows, list):
                raise ValueError("Boundary audit requires decisions")
            ids = [str(row.get("capability_id")) for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError("Boundary audit must cover every capability exactly once")
            for row in rows:
                if row.get("action") not in {"keep", "rename", "exclude"}:
                    raise ValueError("Boundary audit returned invalid action")
                if row.get("action") == "rename" and not all(str(row.get(key) or "").strip()
                        for key in ("name", "definition", "inclusion_criteria", "exclusion_criteria")):
                    raise ValueError("Renamed capability requires a name, definition and boundaries")
            return rows

        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            return index, validate(saved)
        payload = {"capabilities": [{key: row.get(key) for key in
                    ("capability_id", "name", "definition", "inclusion_criteria", "exclusion_criteria")}
                    for row in batch]}
        rows, meta = _complete_validated(client, audit_system, payload, validate)
        checkpoint.write_text(json.dumps({"decisions": rows, "api_meta": meta}, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        return index, rows

    audit_results: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(audit_batch, index, batch) for index, batch in enumerate(audit_batches)]
        for future in as_completed(futures):
            index, rows = future.result()
            audit_results[index] = rows
    audit_lookup = {str(row["capability_id"]): row
                    for index in range(len(audit_batches)) for row in audit_results[index]}
    audited_by_name: dict[str, dict[str, Any]] = {}
    boundary_excluded_proposals: set[str] = set()
    for leaf in canonical:
        decision = audit_lookup[str(leaf["capability_id"])]
        if decision["action"] == "exclude":
            boundary_excluded_proposals.update(leaf["member_ids"])
            continue
        updated = dict(leaf)
        if decision["action"] == "rename":
            updated.update({key: str(decision[key]) for key in
                            ("name", "definition", "inclusion_criteria", "exclusion_criteria")})
        key = _normalized_name(updated["name"])
        if key not in audited_by_name:
            audited_by_name[key] = updated
        else:
            target = audited_by_name[key]
            target["member_ids"].extend(updated["member_ids"])
            target["source_case_ids"] = sorted(set(target["source_case_ids"]) | set(updated["source_case_ids"]))
    canonical = []
    proposal_to_leaf: dict[str, str] = {}
    for key, leaf in audited_by_name.items():
        members = sorted(set(leaf.pop("member_ids")))
        leaf_id = stable_id("cap", taxonomy_version, key, members)
        leaf["capability_id"] = leaf_id
        canonical.append(leaf)
        proposal_to_leaf.update({member: leaf_id for member in members})

    hierarchy_system = (
        f"你是科研ML/LLM Coding Agent能力层级组织器。将输入能力划分为2到{max_children}个互不重叠、边界清晰的"
        "直接子能力组，返回JSON对象groups，每组包含name、definition、member_ids。按Agent执行行为分类，不按研究学科、"
        "项目、数据集、模型或算法家族分类。不同粒度不能作为同级；名称不得简单拼接成员名称。所有id必须恰好出现一次。"
    )

    def organize(items: list[dict[str, Any]], lineage: tuple[str, ...]) -> list[dict[str, Any]]:
        if len(items) <= max_children:
            return items
        payload = [{"id": row["capability_id"], "name": row["name"],
                    "definition": row.get("definition") or ""} for row in items]
        grouped, _ = _complete_hierarchy(client, hierarchy_system, payload, max_children)
        lookup = {str(row["capability_id"]): row for row in items}
        result = []
        for index, group in enumerate(grouped):
            members = [lookup[str(member)] for member in group["member_ids"]]
            if len(members) == 1:
                result.append(members[0])
                continue
            children = organize(members, lineage + (str(group["name"]),))
            node_id = stable_id("capgroup", taxonomy_version, lineage, group["name"], index,
                                sorted(row["capability_id"] for row in members))
            result.append({"capability_id": node_id, "name": str(group["name"]),
                           "definition": str(group.get("definition") or ""), "children": children,
                           "status": "supported", "is_leaf": False})
        if len(result) > max_children:
            return organize(result, lineage + ("上层归并",))
        return result

    roots = organize(canonical, ("科研 ML/LLM Coding Agent 能力",))

    def compress(node: dict[str, Any]) -> dict[str, Any]:
        children = [compress(child) for child in node.get("children") or []]
        # A one-child category adds no information. Likewise, a child repeating its
        # parent's name is an implementation artifact rather than a useful level.
        expanded = []
        for child in children:
            if (child.get("children")
                    and _normalized_name(str(child["name"])) == _normalized_name(str(node["name"]))):
                expanded.extend(child["children"])
            else:
                expanded.append(child)
        node["children"] = expanded
        if len(expanded) == 1 and expanded[0].get("children"):
            return expanded[0]
        return node

    roots = [compress(node) for node in roots]
    source_case_lookup = {str(case["case_id"]): case for case in source_cases}
    assigned_cases: list[dict[str, Any]] = []
    for proposal in proposals:
        if proposal["proposal_id"] in boundary_excluded_proposals:
            continue
        leaf_id = proposal_to_leaf[proposal["proposal_id"]]
        source_case = source_case_lookup[proposal["source_case_id"]]
        assignment_id = stable_id("capcase", source_case["case_id"], leaf_id)
        if any(row["case_id"] == assignment_id for row in assigned_cases):
            continue
        assigned_cases.append({**source_case, "case_id": assignment_id,
                               "source_case_id": source_case["case_id"],
                               "capability_id": leaf_id, "taxonomy_version": taxonomy_version,
                               "assignment_support": {
                                   "support_type": proposal.get("support_type"),
                                   "supporting_event_ids": proposal.get("supporting_event_ids") or [],
                                   "entailment_audit": "supported",
                               },
                               "assignment_status": "published"})

    cases_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in assigned_cases:
        cases_by_leaf[str(case["capability_id"])].append(case)

    def update(node: dict[str, Any], parent_id: str, path: list[str], level: int) -> list[dict[str, Any]]:
        node["parent_id"], node["path"], node["level"] = parent_id, path + [node["name"]], level
        children = node.get("children") or []
        descendants = ([case for child in children
                        for case in update(child, node["capability_id"], node["path"], level + 1)]
                       if children else cases_by_leaf.get(str(node["capability_id"]), []))
        node["case_count"] = len({case.get("source_case_id", case["case_id"]) for case in descendants})
        node["trace_count"] = len({case.get("trace_id") for case in descendants})
        node["evidence_count"] = sum(len(case.get("evidence") or []) for case in descendants)
        node["query_count"] = len({turn for case in descendants for turn in case.get("turn_ids") or []})
        node["is_leaf"] = not bool(children)
        if node["is_leaf"]:
            # Query count is descriptive evidence volume, not a publication tier.
            node["status"] = "supported"
        return descendants

    root_name = "科研 ML/LLM Coding Agent 能力"
    for node in roots:
        update(node, "capability_root", [root_name], 1)
    now = dt.datetime.now().astimezone()
    out = output_root / f"version_{taxonomy_version}" / "core"
    out.mkdir(parents=True, exist_ok=False)
    taxonomy = {
        "taxonomy_version": taxonomy_version, "status": "analysis_snapshot",
        "generated_at": now.isoformat(),
        "construction_method": "case_atomization_global_consolidation_recursive_hierarchy_multilabel",
        "base_taxonomy": str((core_dir / "capability_taxonomy.json").resolve()),
        "generated_by": {"method": "llm", "provider": provider, "model": client.model,
                         "prompt_version": "agent_atomic_global_v2", "max_children": max_children},
        "scope_policy": "Only cross-domain reusable and independently verifiable Agent work behaviors",
        "root": {"capability_id": "capability_root", "name": root_name, "children": roots},
    }
    (out / "capability_taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in assigned_cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    included_source_ids = {str(row["source_case_id"]) for row in assigned_cases}
    atomic_reason = {str(row["case_id"]): row.get("reason") for row in decisions}
    excluded = [{"case_id": str(case["case_id"]),
                 "reason": atomic_reason.get(str(case["case_id"])) or
                           "边界终审未发现可由用户原文支持的跨领域Coding Agent行为"}
                for case in source_cases if str(case["case_id"]) not in included_source_ids]
    (out / "excluded_domain_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in excluded),
        encoding="utf-8")
    regenerate_distribution_csv(out)
    validation = validate_core_snapshot(out)
    return {**validation, "status": "success", "output_path": str(out.resolve()),
            "source_case_count": len(source_cases), "included_source_cases": len(included_source_ids),
            "excluded_source_cases": len(excluded), "assignment_count": len(assigned_cases),
            "multi_label_source_cases": sum(count > 1 for count in Counter(
                str(row["source_case_id"]) for row in assigned_cases).values()),
            "max_children": max_children, "provider": provider, "model": client.model}
