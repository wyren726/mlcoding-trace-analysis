from __future__ import annotations

import datetime as dt
import hashlib
import json
import fcntl
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .mapping import CapabilityTaxonomy, load_taxonomy
from .prompt import PROMPT_VERSION, build_capability_messages

SCHEMA_VERSION = "trace-quality-episode-v5"
DOMAINS = {"生物", "化学", "环境", "材料", "计算机", "数学", "物理", "工科", "unknown"}
STAGES = {"question_formulation", "literature_or_data_search", "data_acquisition", "data_cleaning_and_preparation", "method_or_experiment_design", "implementation_or_coding", "environment_and_dependency_setup", "training_or_experiment_execution", "debugging", "result_analysis", "visualization", "validation_and_reproducibility", "report_or_paper_delivery", "unknown"}
STAGE_ALIASES = {"文献调研": "literature_or_data_search", "问题定义": "question_formulation", "实验设计": "method_or_experiment_design", "环境配置": "environment_and_dependency_setup", "实验执行": "training_or_experiment_execution", "实验执行与监控": "training_or_experiment_execution", "实验执行与迭代": "training_or_experiment_execution", "实验执行与验证": "training_or_experiment_execution", "实验核验": "validation_and_reproducibility", "实验状态诊断": "debugging", "结果分析": "result_analysis", "结果解释": "result_analysis", "结果可视化": "visualization", "论文撰写": "report_or_paper_delivery", "论文写作": "report_or_paper_delivery", "论文修订": "report_or_paper_delivery", "论文撰写与修订": "report_or_paper_delivery", "论文撰写与投稿": "report_or_paper_delivery", "论文写作与投稿": "report_or_paper_delivery", "成果发布": "report_or_paper_delivery"}
FACTORS = {"multi_step", "cross_artifact", "tool_dependency", "long_horizon_state", "goal_evolution", "scientific_reasoning", "verification_burden", "external_information_dependency"}
FACTOR_ALIASES = {"多轮交互": "multi_step", "多轮迭代需求": "goal_evolution", "多轮迭代修改": "goal_evolution", "多轮需求演进": "goal_evolution", "多轮迭代修订": "goal_evolution", "跨章节内容整合": "cross_artifact", "跨章节交叉核验": "verification_burden", "多源文献综合": "external_information_dependency", "跨学科知识整合": "scientific_reasoning", "跨学科概念整合": "scientific_reasoning", "论文-代码一致性核验": "verification_burden", "多任务并行": "multi_step", "多步流程协调": "multi_step", "跨会话状态恢复": "long_horizon_state", "高风险操作约束遵守": "verification_burden", "多研究模块协同": "cross_artifact", "格式规范适配": "verification_burden", "创新性论证": "scientific_reasoning"}
_FAILURE_ALIASES = {
    "意图漂移": "context_tracking_failure", "目标偏移": "context_tracking_failure",
    "约束遗漏": "constraint_loss", "未完成执行": "incomplete_execution",
    "执行不完整": "incomplete_execution", "错误假设": "wrong_assumption",
    "工具误用": "tool_misuse", "错误恢复失败": "error_recovery_failure",
    "验证失败": "verification_failure", "交付物缺失": "artifact_delivery_failure",
}
_FAILURE_VALUES = {"incomplete_execution", "constraint_loss", "wrong_assumption", "tool_misuse", "error_recovery_failure", "context_tracking_failure", "verification_failure", "artifact_delivery_failure", "unknown"}
GATE_KEYS = ("goal_observable", "agent_behavior_observable", "outcome_observable", "evidence_sufficient")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {number} is not an object")
            yield number, value


def _episode_payload(bundle, row):
    from ..stage_02_analyze.runner import candidate_agent_payload
    boundary = row.get("episode_boundary") or {}
    positions = {turn.get("turn_id"): i for i, turn in enumerate(bundle.turns)}
    start, end = (positions.get(boundary.get(key)) for key in ("start_turn_id", "end_turn_id"))
    if start is None or end is None or end < start:
        raise ValueError("Missing or invalid authoritative Episode boundaries; evidence references are not boundaries")
    episode = {"episode_id": row.get("episode_id"), "query_turn_id": row.get("query_turn_id"),
               "episode_boundary": boundary, "route": {"judgment": "analyze"}}
    return candidate_agent_payload(bundle, {"task_episodes": [episode]})


def _prepare_episode_inputs(rows):
    """Read each normalized original-event file once, using Stage03's view."""
    from ..stage_02_analyze.runner import iter_trace_bundles
    groups, results = {}, {}
    for number, row in rows:
        path = (row.get("lineage") or {}).get("preprocessed_input")
        if not path:
            results[number] = ValueError("Missing lineage.preprocessed_input")
            continue
        groups.setdefault(path, {}).setdefault(row.get("trace_id"), []).append((number, row))
    for path, cases in groups.items():
        for bundle in iter_trace_bundles(Path(path)):
            for number, row in cases.pop(bundle.trace_id, []):
                try:
                    results[number] = _episode_payload(bundle, row)
                except ValueError as exc:
                    results[number] = exc
        for missing in cases.values():
            for number, _ in missing:
                results[number] = ValueError("Trace missing from preprocessed input")
    return results


def _exclusive_run(function):
    @wraps(function)
    def wrapped(input_path, output_dir, *args, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(output_dir) / ".labeling.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("This Stage06 run already has an active writer") from exc
            return function(input_path, output_dir, *args, **kwargs)
    return wrapped


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _domain(row: dict[str, Any]) -> tuple[str, list[str]]:
    # Mentioning a domain in a reason is not a semantic domain judgment.
    return "unknown", []


def _quality(row: dict[str, Any]) -> tuple[str, str, str]:
    # Existence of references or agent_related=True cannot prove quality,
    # episode completeness, or semantic evidence sufficiency.
    return "unknown", "unknown", "unknown"


def _failure_pattern(row: dict[str, Any]) -> list[str]:
    # A coarse historical category does not establish its finer subtypes.
    return ["unknown"]


def _quality_from_gates(gates: dict[str, Any]) -> str:
    if set(gates) != {*GATE_KEYS, "episode_integrity"}:
        raise ValueError("quality_gates must contain exactly four judgments and episode_integrity")
    values = [gates[key] for key in GATE_KEYS]
    if any(value is not None and type(value) is not bool for value in values):
        raise ValueError("quality_gates judgments must be boolean or null")
    if gates["episode_integrity"] not in ("complete", "partial", "broken", "unknown"):
        raise ValueError("Invalid quality_gates episode_integrity")
    if all(value is True for value in values) and gates.get("episode_integrity") == "complete":
        return "high"
    if any(value is False for value in values) or gates.get("episode_integrity") == "broken":
        return "low"
    if any(value is None for value in values) or gates["episode_integrity"] == "unknown":
        return "unknown"
    return "medium"


def _enum_stage(value: Any) -> str:
    return value if value in STAGES else STAGE_ALIASES.get(value, "unknown") if isinstance(value, str) else "unknown"


def _enum_factor(value: Any) -> str | None:
    return value if value in FACTORS else FACTOR_ALIASES.get(value) if isinstance(value, str) else None


def _row(row: dict[str, Any], source_file: Path, line_number: int, tag_run_id: str,
         taxonomy: CapabilityTaxonomy | None = None) -> dict[str, Any]:
    user = row.get("user_screen") or {}
    agent = row.get("agent_verification") or {}
    quality, integrity, evidence_status = _quality(row)
    domain_primary, domain_secondary = _domain(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "batch_id": row.get("batch_id"),
            "source_result_file": str(source_file),
            "source_line_number": line_number,
            "session_ids": row.get("session_ids") or [],
            "trace_id": row.get("trace_id"),
            "episode_id": row.get("episode_id"),
            "query_turn_id": row.get("query_turn_id"),
        },
        "historical_result": {
            "pain_judgment": row.get("pain_judgment"),
            "outcome": row.get("outcome"),
        },
        "stage_04_labels": {
            "judgment_reason_codes": row.get("judgment_reason_codes") or [],
            "user_screen": user,
            "agent_verification": agent,
            "outcome": row.get("outcome"),
            "capability_gaps": row.get("capability_gaps") or [],
        },
        "reused": {
            "initial_query": user.get("initial_query"),
            "initial_query_clarity": user.get("initial_query_clarity"),
            "preliminary_goal": user.get("preliminary_goal"),
            "requirement_evolution": user.get("requirement_evolution"),
            "agent_failure": agent.get("agent_failure"),
            "agent_related": agent.get("agent_related"),
            "agent_related_reason": agent.get("agent_related_reason"),
            "capability_gaps": row.get("capability_gaps") or [],
            "attribution": agent.get("attribution"),
        },
        "labels": {
            "trace_quality": quality,
            "episode_integrity": integrity,
            "evidence_status": evidence_status,
            "domain_primary": domain_primary,
            "domain_secondary": domain_secondary,
            "research_stage_primary": "unknown",
            "research_stage_secondary": [],
            "difficulty": "unknown",
            "difficulty_factors": [],
            "capability_mapping": [],
            "failure_pattern": _failure_pattern(row),
        },
        "evidence_refs": row.get("evidence_references") or [],
        "label_reasons": {},
        "label_provenance": {
            "tag_run_id": tag_run_id,
            "label_version": "v5",
            "source_result_version": row.get("schema_version"),
            "decision_source": "deterministic",
            "review_status": "not_reviewed",
            "labeling_status": "pending_semantic_analysis",
            "taxonomy_sha256": taxonomy.sha256 if taxonomy else None,
        },
    }


def _semantic_labels(row: dict[str, Any], taxonomy: CapabilityTaxonomy, client: Any, episode: dict) -> dict[str, Any]:
    from ..stage_02_analyze.compact import (
        indexed_candidate_payload, partition_indexed_candidate_payload,
        normalise_selected_event_ids, selected_evidence_payload,
    )
    from ..stage_02_analyze.runner import complete_index_partition_with_fallback
    if len(json.dumps(episode, ensure_ascii=False)) <= 200_000:
        result = _semantic_labels_once(row, taxonomy, client, episode)
        result["metadata"]["evidence_strategy"] = "complete_episode"
        return result
    index = indexed_candidate_payload(episode)
    partitions = partition_indexed_candidate_payload(index, max_prompt_chars=40_000)
    selected, rounds = [], []
    context = {"purpose": "Stage06 新增标签：任务难度、科研领域、科研阶段、需求子需求、失败模式、交互完整性与证据质量。按这些维度选择证据，不只搜索报错。",
               "historical_result": row.get("user_screen")}
    for round_number in (1, 2):
        for partition in partitions:
            for effective, selection, metadata in complete_index_partition_with_fallback(client, partition, retrieval_context=context):
                ids = normalise_selected_event_ids(selection, episode,
                    limit=int(effective.get("partition", {}).get("selection_limit") or 1), excluded=set(selected))
                selected.extend(ids)
                rounds.append({"round": round_number, "selected_ids": ids,
                               "reason": selection.get("reason"), "model_metadata": metadata})
        evidence = selected_evidence_payload(episode, selected)
        evidence["episode_evidence_spans"] = episode["episode_evidence_spans"]
        evidence["coverage_note"] = "全部事件已建立索引；本次展示用户、Agent、错误正文和选中的工具原文。未取回的工具内容不能视为已经验证。"
        result = _semantic_labels_once(row, taxonomy, client, evidence)
        result["metadata"].update({"evidence_strategy": "stage03_indexed_retrieval", "retrieval_rounds": rounds,
                                   "selected_ids": list(selected), "full_episode_read": False})
        if result["labels"]["evidence_status"] == "sufficient" and result["quality_gates"]["evidence_sufficient"] is True:
            break
        context.update({"round": 2, "previously_selected_event_ids": list(selected),
                        "missing_evidence_queries": result["quality_gate_reasons"], "label_reasons": result["reason"]})
    return result


def _semantic_labels_once(row: dict[str, Any], taxonomy: CapabilityTaxonomy, client: Any, episode: dict) -> dict[str, Any]:
    messages = build_capability_messages({
        "source": {"session_ids": row.get("session_ids"), "trace_id": row.get("trace_id"),
                   "episode_id": row.get("episode_id"), "query_turn_id": row.get("query_turn_id")},
        "goal": (row.get("user_screen") or {}).get("preliminary_goal"),
        "initial_query": (row.get("user_screen") or {}).get("initial_query"),
        "initial_query_clarity": (row.get("user_screen") or {}).get("initial_query_clarity"),
        "agent_verification": row.get("agent_verification"),
        "episode_boundary": row.get("episode_boundary"),
        "historical_pain_judgment": row.get("pain_judgment"),
        "historical_outcome": row.get("outcome"),
        "requirement_evolution": (row.get("user_screen") or {}).get("requirement_evolution"),
        "stage_04_labels": {
            "pain_judgment": row.get("pain_judgment"),
            "judgment_reason_codes": row.get("judgment_reason_codes") or [],
            "user_screen": row.get("user_screen"),
            "agent_verification": row.get("agent_verification"),
            "outcome": row.get("outcome"),
            "capability_gaps": row.get("capability_gaps") or [],
        },
        "raw_episode": episode,
        "legacy_capability_gaps": row.get("capability_gaps") or [],
        "evidence_references": row.get("evidence_references") or [],
    }, taxonomy)
    system, user = messages[0]["content"], messages[1]["content"]
    if len(system) + len(user) > 650_000:
        raise ValueError("Stage06 selected evidence exceeds 650000 characters; no evidence was silently truncated")
    for attempt in range(3):
        value, metadata = client.complete_json(system, user)
        try:
            result = _validate_semantic_labels(value, metadata, taxonomy)
            result["metadata"] = {**metadata, "validation_retries": attempt}
            return result
        except ValueError as exc:
            if attempt == 2:
                raise
            user += "\n本次输出未通过校验，请重新输出完整 JSON。错误：" + str(exc)
            user += "\nsecondary 无适用标签时返回 []，不得填写 unknown。领域只允许：" + json.dumps(sorted(DOMAINS), ensure_ascii=False)
            user += "；阶段只允许：" + json.dumps(sorted(STAGES))
            user += "；难度因素只允许：" + json.dumps(sorted(FACTORS))
            user += "。需求映射仍只能选择原表 taxonomy_row。上次输出：" + json.dumps(value, ensure_ascii=False)


def _validate_semantic_labels(value: dict[str, Any], metadata: dict[str, Any], taxonomy: CapabilityTaxonomy) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    mapping = taxonomy.resolve_rows(value.get("capability_mapping"))
    allowed = {
        "episode_integrity": {"complete", "partial", "broken", "unknown"},
        "evidence_status": {"sufficient", "partial", "insufficient", "unknown"},
        "difficulty": {"low", "medium", "high", "unknown"},
    }
    for field, values in allowed.items():
        if not isinstance(value.get(field), str) or value[field] not in values:
            raise ValueError(f"Invalid {field}: {value.get(field)!r}")
    if not isinstance(value.get("quality_gates"), dict):
        raise ValueError("Invalid quality_gates")
    quality = _quality_from_gates(value["quality_gates"])
    if value["quality_gates"]["episode_integrity"] != value["episode_integrity"]:
        raise ValueError("Conflicting episode_integrity judgments")
    gate_reasons = value.get("quality_gate_reasons")
    if not isinstance(gate_reasons, dict) or set(gate_reasons) != set(GATE_KEYS) or not all(isinstance(v, str) and v.strip() for v in gate_reasons.values()):
        raise ValueError("Missing quality_gate_reasons")
    for field in ("domain_primary", "research_stage_primary"):
        if not isinstance(value.get(field), str):
            raise ValueError(f"Invalid {field}")
    for field in ("domain_secondary", "research_stage_secondary", "difficulty_factors", "failure_pattern"):
        if not isinstance(value.get(field), list) or not all(isinstance(x, str) for x in value[field]):
            raise ValueError(f"Invalid {field}")
    for field, items, choices in (
        ("domain_primary", [value["domain_primary"]], DOMAINS),
        ("domain_secondary", value["domain_secondary"], DOMAINS - {"unknown"}),
        ("research_stage_primary", [value["research_stage_primary"]], STAGES),
        ("research_stage_secondary", value["research_stage_secondary"], STAGES - {"unknown"}),
        ("difficulty_factors", value["difficulty_factors"], FACTORS),
    ):
        if any(item not in choices for item in items):
            raise ValueError(f"Invalid {field}: {items!r}")
    if any(item not in _FAILURE_VALUES for item in value["failure_pattern"]):
        raise ValueError(f"Invalid failure_pattern: {value['failure_pattern']!r}")
    return {"labels": {
        "trace_quality": quality,
        "episode_integrity": value["episode_integrity"],
        "evidence_status": value["evidence_status"],
        "domain_primary": value["domain_primary"], "domain_secondary": value["domain_secondary"],
        "research_stage_primary": value["research_stage_primary"],
        "research_stage_secondary": value["research_stage_secondary"],
        "difficulty": value["difficulty"], "difficulty_factors": value["difficulty_factors"],
        "capability_mapping": mapping, "failure_pattern": value["failure_pattern"],
    }, "quality_gates": value["quality_gates"], "quality_gate_reasons": gate_reasons, "reason": value.get("reason") or {}, "evidence_refs": [],
    "metadata": metadata}


@_exclusive_run
def run_labeling(input_path: Path, output_dir: Path, resume: bool = True,
                 taxonomy_path: Path | None = None, client: Any = None,
                 workers: int = 1, limit: int | None = None,
                 include_lines: set[int] | None = None) -> dict[str, Any]:
    """Prepare or label Stage 04 records, appending only pending successes."""
    if client is not None and taxonomy_path is None:
        raise ValueError("Semantic labeling requires a taxonomy")
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    taxonomy = load_taxonomy(taxonomy_path) if taxonomy_path else None
    tag_run_id = output_dir.name
    output = output_dir / f"labeled_candidates__{input_path.stem}__{tag_run_id}.jsonl"
    errors = output_dir / f"errors__{tag_run_id}.jsonl"
    progress = output_dir / f"progress__{tag_run_id}.json"
    manifest = output_dir / f"manifest__{tag_run_id}.json"
    signature = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "taxonomy_sha256": taxonomy.sha256 if taxonomy else None,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": getattr(client, "model", None),
        "mode": "semantic" if client is not None else "prepare",
        "pain_judgments": ["confirmed", "review"],
    }
    if output.exists():
        if not resume:
            raise ValueError("Output exists; choose a new run directory instead of overwriting")
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Input, taxonomy or schema changed; choose a new run directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_base = {
        "schema_version": SCHEMA_VERSION, "signature": signature,
        "input": str(input_path), "output": str(output), "tag_run_id": tag_run_id,
        "taxonomy": taxonomy.metadata() if taxonomy else None,
        "labeling_status": "pending_semantic_analysis", "semantic_completed": 0,
    }
    manifest.write_text(json.dumps(manifest_base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed = set()
    if resume and output.exists():
        for _, item in _read_jsonl(output):
            source = item.get("source") or {}
            completed.add((source.get("source_line_number"), source.get("trace_id"), source.get("episode_id")))
    total = success = failed = 0
    mode = "a" if resume else "w"
    rows = list(_read_jsonl(input_path))
    input_total = len(rows)
    rows = [(n, row) for n, row in rows if row.get("pain_judgment") in {"confirmed", "review"}]
    manifest_base.update({"input_total": input_total, "eligible_total": len(rows), "skipped": input_total - len(rows)})
    if limit is not None:
        rows = rows[:limit]
    source_keys = {(n, r.get("trace_id"), r.get("episode_id")) for n, r in rows}
    success = len(completed & source_keys)
    total = len(rows)
    if include_lines is not None and include_lines - {n for n, _ in rows}:
        raise ValueError("Requested source lines do not exist")
    pending = [item for item in rows if
               (item[0], item[1].get("trace_id"), item[1].get("episode_id")) not in completed
               and (include_lines is None or item[0] in include_lines)]
    episode_inputs = _prepare_episode_inputs(pending) if client else {}
    def process(item: tuple[int, dict[str, Any]]):
        line_number, row = item
        result = _row(row, input_path, line_number, tag_run_id, taxonomy)
        if client and taxonomy:
            episode = episode_inputs[line_number]
            if isinstance(episode, Exception):
                raise episode
            semantic = _semantic_labels(row, taxonomy, client, episode)
            result["stage_04_labels"] = row
            result["evidence_input"] = {
                "preprocessed_input": row["lineage"]["preprocessed_input"],
                "input_view": episode["input_view"],
                "episode_evidence_spans": episode["episode_evidence_spans"],
                "sha256": hashlib.sha256(json.dumps(episode, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            }
            result["labels"] = semantic["labels"]
            result["quality_gates"] = semantic["quality_gates"]
            result["quality_gate_reasons"] = semantic["quality_gate_reasons"]
            result["label_reasons"] = semantic["reason"]
            result["evidence_refs"] = semantic["evidence_refs"] or result["evidence_refs"]
            result["label_provenance"].update({
                "decision_source": "model_and_deterministic",
                "review_status": "model_only",
                "labeling_status": "semantic_completed",
                "model_metadata": semantic["metadata"],
                "model": client.model,
                "prompt_version": PROMPT_VERSION,
                "mapping_method": "taxonomy_row_lookup",
            })
        return line_number, result
    with output.open(mode, encoding="utf-8") as target, errors.open("a", encoding="utf-8") as error_target:
        pending = [item for item in rows if
                   (item[0], item[1].get("trace_id"), item[1].get("episode_id")) not in completed
                   and (include_lines is None or item[0] in include_lines)]
        futures = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for item in pending:
                futures[executor.submit(process, item)] = item
            for future in as_completed(futures):
                line_number, row = futures[future]
                try:
                    _, value = future.result()
                    target.write(json.dumps(value, ensure_ascii=False) + "\n")
                    target.flush()
                    success += 1
                except Exception as exc:
                    error_target.write(json.dumps({"source_line_number": line_number, "error": str(exc)}, ensure_ascii=False) + "\n")
                    error_target.flush()
                    failed += 1
                progress.write_text(json.dumps({"updated_at": _now(), "total_seen": total, "prepared": success, "failed": failed, "semantic_completed": success if client else 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    progress_value = {"updated_at": _now(), "total_seen": total, "prepared": success,
                      "failed": failed, "pending": total - success,
                      "attempted_this_run": len(pending),
                      "semantic_completed": success if client else 0,
                      "labeling_status": ("complete" if success == total else "partial") if client else "pending_semantic_analysis"}
    progress.write_text(json.dumps(progress_value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({**manifest_base, **progress_value}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    session_paths = output_dir / f"session_paths__confirmed_high__{input_path.stem}__{tag_run_id}.jsonl"
    if client and success == total:
        source_rows = {number: value for number, value in rows}
        exported = []
        seen_paths = set()
        for _, labeled in _read_jsonl(output):
            if (labeled.get("historical_result") or {}).get("pain_judgment") != "confirmed":
                continue
            if (labeled.get("labels") or {}).get("trace_quality") != "high":
                continue
            source = labeled.get("source") or {}
            original = source_rows.get(source.get("source_line_number")) or {}
            paths = sorted({
                ((reference.get("source") or {}).get("input_file"))
                for reference in (original.get("source_references") or [])
                if ((reference.get("source") or {}).get("input_file") or "").endswith("/_internal/session.jsonl")
            })
            if not paths:
                raise ValueError(f"No original session path for confirmed high case: {source}")
            if len(paths) != 1:
                raise ValueError(f"Multiple original session paths require explicit resolution: {source}")
            session_path = paths[0]
            if session_path in seen_paths:
                continue
            seen_paths.add(session_path)
            exported.append({
                "session_jsonl_path": session_path,
                "workspace_path": str(Path(session_path).parent.parent),
            })
        session_paths.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in exported), encoding="utf-8")
        final_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        final_manifest["session_paths_output"] = str(session_paths)
        final_manifest["session_paths_count"] = len(exported)
        manifest.write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"input": str(input_path), "output": str(output), "session_paths_output": str(session_paths), "total": total, "prepared": success, "failed": failed, "tag_run_id": tag_run_id, "semantic_completed": success if client else 0, "taxonomy": taxonomy.metadata() if taxonomy else None}
