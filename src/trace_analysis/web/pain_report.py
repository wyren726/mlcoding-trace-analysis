from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


STATUS_LABELS = {
    "confirmed": "确认痛点",
    "review": "需人工复核",
    "excluded": "排除候选",
}
ATTRIBUTION_LABELS = {
    "reasoning_or_response": "推理与响应",
    "execution_or_verification": "执行与验证",
    "tool_use": "工具使用",
    "context_handling": "上下文处理",
    "permission": "权限",
    "external": "外部限制",
    "user_input": "用户输入或需求变化",
    "unknown": "未知",
}
OUTCOME_LABELS = {
    "completed": "完成",
    "partially_completed": "部分完成",
    "failed": "失败",
    "unknown": "未知",
}
JUDGMENT_LABELS = {
    "clear": "清晰",
    "partly_clear": "部分清晰",
    "unclear": "不清晰",
    "high": "高",
    "medium": "中",
    "low": "低",
    "uncertain": "不确定",
    "met": "满足",
    "partially_met": "部分满足",
    "unmet": "未满足",
    "unknown": "未知",
    "initial_query": "初始要求",
    "clarified_later": "后续澄清",
    "new_later": "后续新增",
}
MODEL_LABELS = {
    "anthropic/claude-4.6-opus-20260205": "Claude 4.6 Opus",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "unknown": "未知模型",
}

_PATH_RE = re.compile(r"(?<![\w.])/(?:[^\s/]+/)+[^\s,;，。；：:]+")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SECRET_RE = re.compile(r"\b(?:sk|api)[-_][A-Za-z0-9_-]{16,}\b", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not an object: {path}")
            records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _public_text(value: Any, limit: int = 800) -> str:
    """Keep evidence readable while removing common public-site secrets and paths."""
    text = str(value or "").strip()
    text = _URL_RE.sub("[链接]", text)
    text = _EMAIL_RE.sub("[邮箱]", text)
    text = _SECRET_RE.sub("[凭据]", text)
    text = _PATH_RE.sub("[本地路径]", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _episode_status(episode: dict[str, Any]) -> str:
    assessment = episode.get("agent_assessment")
    related = assessment.get("agent_related") if isinstance(assessment, dict) else None
    if related is True:
        return "confirmed"
    if related is False:
        return "excluded"
    return "review"


def _trace_status(episodes: list[dict[str, Any]]) -> str:
    statuses = {_episode_status(episode) for episode in episodes}
    if "confirmed" in statuses:
        return "confirmed"
    if "review" in statuses:
        return "review"
    return "excluded"


def _models(values: set[str]) -> list[str]:
    result = sorted(value for value in values if value and value != "<synthetic>")
    return result or ["unknown"]


def _model_label(values: list[str]) -> str:
    return " + ".join(MODEL_LABELS.get(value, value) for value in values)


def _short(value: Any, limit: int = 150) -> str:
    text = " ".join(_public_text(value, limit * 2).split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text or "—"


def _episode_title(episode: dict[str, Any], status: str) -> str:
    assessment = episode.get("agent_assessment") or {}
    gaps = assessment.get("capability_gaps") or []
    if status == "confirmed" and gaps:
        return "、".join(_short(gap.get("capability") or "能力缺口", 32) for gap in gaps[:2])
    if status == "review":
        return _short(assessment.get("agent_failure") or "现有证据不足", 58)
    return "候选信号未构成 Agent 痛点"


def _evidence_item(kind: str, item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    quote = _public_text(item.get("quote"))
    event_id = str(item.get("event_id") or "")
    if not quote or not event_id:
        return None
    return {
        "kind": kind,
        "turn_id": str(item.get("turn_id") or ""),
        "event_id": event_id,
        "quote": quote,
    }


def _evidence_chain(episode: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, item: Any) -> None:
        row = _evidence_item(kind, item)
        if row is None:
            return
        key = (row["event_id"], row["quote"])
        if key not in seen:
            seen.add(key)
            rows.append(row)

    for signal in (episode.get("candidate_signals") or {}).get("signals") or []:
        add("用户候选信号", signal)
    assessment = episode.get("agent_assessment") or {}
    for item in (assessment.get("goal_assessment") or {}).get("evidence") or []:
        add("目标确认", item)
    for gap in assessment.get("capability_gaps") or []:
        for item in gap.get("evidence") or []:
            add("能力缺口", item)
    return rows


def _benchmark_note(episode: dict[str, Any], status: str) -> str:
    if status == "excluded":
        return "不建议作为能力缺陷 Benchmark；当前证据更符合正常等待、任务推进、外部限制或用户新增要求。"
    if status == "review":
        return "暂不进入 Benchmark 候选池；应先人工复核失败是否真实发生、是否由 Agent 导致。"
    clarity = (episode.get("initial_query_clarity") or {}).get("judgment")
    value = (episode.get("analysis_value") or {}).get("judgment")
    if clarity == "clear" and value in {"high", "medium"}:
        return "建议进入 Benchmark 候选池；后续需要结合 Workspace 固化输入、环境、验收标准和参考结果。"
    return "可以保留为 Benchmark 候选，但需要先补足任务输入、验收标准或初始 Query 的明确性。"


def _public_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for item in requirement.get("evidence") or []:
        row = _evidence_item("要求证据", item)
        if row:
            evidence.append(row)
    origin = str(requirement.get("origin") or "unknown")
    status = str(requirement.get("status") or "unknown")
    return {
        "requirement_id": requirement.get("requirement_id"),
        "text": _public_text(requirement.get("text")),
        "origin": origin,
        "origin_label": JUDGMENT_LABELS.get(origin, origin),
        "status": status,
        "status_label": JUDGMENT_LABELS.get(status, status),
        "evidence": evidence,
    }


def _public_case(
    episode: dict[str, Any],
    trace: dict[str, Any],
    display_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = _episode_status(episode)
    assessment = episode.get("agent_assessment") or {}
    preliminary = episode.get("preliminary_goal") or {}
    confirmed_goal = assessment.get("goal_assessment") or {}
    clarity = episode.get("initial_query_clarity") or {}
    analysis_value = episode.get("analysis_value") or {}
    attribution = str(assessment.get("attribution") or "unknown")
    outcome = str(episode.get("outcome") or "unknown")
    gaps = [
        {
            "capability": _public_text(gap.get("capability")),
            "reason": _public_text(gap.get("reason")),
            "evidence": [
                row for item in gap.get("evidence") or []
                if (row := _evidence_item("能力缺口", item))
            ],
        }
        for gap in assessment.get("capability_gaps") or []
        if isinstance(gap, dict)
    ]
    candidate = episode.get("candidate_signals") or {}
    signals = [
        row for item in candidate.get("signals") or []
        if (row := _evidence_item(str(item.get("type") or "候选信号"), item))
    ]
    summary = assessment.get("agent_failure") or assessment.get("agent_related_reason") or candidate.get("reason")
    common = {
        "display_number": display_number,
        "episode_id": str(episode.get("episode_id") or ""),
        "trace_id": trace["trace_id"],
        "query_turn_id": episode.get("query_turn_id"),
        "status": status,
        "status_label": STATUS_LABELS[status],
        "title": _episode_title(episode, status),
        "summary": _short(summary, 220),
        "models": trace["models"],
        "model_label": trace["model_label"],
        "harnesses": trace["harnesses"],
        "judge_model": trace["judge_model"],
        "attribution": attribution,
        "attribution_label": ATTRIBUTION_LABELS.get(attribution, attribution),
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        "capability_gaps": [gap["capability"] for gap in gaps if gap["capability"]],
        "initial_query_clarity": str(clarity.get("judgment") or "unknown"),
        "initial_query_clarity_label": JUDGMENT_LABELS.get(str(clarity.get("judgment") or "unknown"), str(clarity.get("judgment") or "unknown")),
        "analysis_value": str(analysis_value.get("judgment") or "unknown"),
        "analysis_value_label": JUDGMENT_LABELS.get(str(analysis_value.get("judgment") or "unknown"), str(analysis_value.get("judgment") or "unknown")),
        "benchmark_note": _benchmark_note(episode, status),
        "warning_count": len(trace["validation_warnings"]),
    }
    detail = {
        **common,
        "episode_boundary": {
            "start_turn_id": (episode.get("episode_boundary") or {}).get("start_turn_id"),
            "end_turn_id": (episode.get("episode_boundary") or {}).get("end_turn_id"),
            "reason": _public_text((episode.get("episode_boundary") or {}).get("reason")),
        },
        "goal": _public_text(confirmed_goal.get("goal") or preliminary.get("goal")),
        "goal_reason": _public_text(confirmed_goal.get("reason") or preliminary.get("reason")),
        "initial_query_clarity_reason": _public_text(clarity.get("reason")),
        "analysis_value_reason": _public_text(analysis_value.get("reason")),
        "requirements": [
            _public_requirement(item)
            for item in preliminary.get("requirements") or []
            if isinstance(item, dict)
        ],
        "candidate_reason": _public_text(candidate.get("reason")),
        "candidate_signals": signals,
        "agent_failure": _public_text(assessment.get("agent_failure")),
        "agent_related_reason": _public_text(assessment.get("agent_related_reason")),
        "capability_gap_details": gaps,
        "evidence_chain": _evidence_chain(episode),
        "validation_warnings": [_public_text(item, 240) for item in trace["validation_warnings"]],
        "privacy_note": "公开视图仅包含经过脱敏和长度限制的关键证据，不包含完整 User Turn、Tool Result、原始文件路径或 Session 路径。",
    }
    return common, detail


def publish_pain_report(
    analysis_path: Path,
    user_turn_path: Path,
    unified_turn_path: Path,
    manifest_path: Path,
    site_dir: Path,
) -> dict[str, Any]:
    """Publish a public-safe JSON view of Agent-verified pain analysis."""
    analysis_records = [
        record for record in _read_jsonl(analysis_path)
        if (record.get("route") or {}).get("judgment") == "analyze"
    ]
    analysis_ids = {str(record.get("trace_id")) for record in analysis_records}
    user_records = {
        str(record.get("trace_id")): record
        for record in _read_jsonl(user_turn_path)
        if str(record.get("trace_id")) in analysis_ids
    }
    trace_metadata: dict[str, dict[str, set[str]]] = {
        trace_id: {"models": set(), "harnesses": set()} for trace_id in analysis_ids
    }
    with unified_turn_path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            turn = json.loads(line)
            trace_id = str(turn.get("trace_id") or "")
            if trace_id not in trace_metadata:
                continue
            trace_metadata[trace_id]["models"].update(
                str(value) for value in turn.get("agent_models") or [] if value
            )
            if turn.get("harness"):
                trace_metadata[trace_id]["harnesses"].add(str(turn["harness"]))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    traces: list[dict[str, Any]] = []
    for record in analysis_records:
        trace_id = str(record.get("trace_id"))
        episodes = [
            episode for episode in record.get("task_episodes") or []
            if isinstance(episode.get("agent_assessment"), dict)
        ]
        if not episodes:
            continue
        user_record = user_records.get(trace_id, {})
        model_values = set(trace_metadata[trace_id]["models"])
        model_values.update(str(value) for value in user_record.get("agent_models") or [] if value)
        harness_values = set(trace_metadata[trace_id]["harnesses"])
        if user_record.get("harness"):
            harness_values.add(str(user_record["harness"]))
        models = _models(model_values)
        traces.append({
            "trace_id": trace_id,
            "episodes": episodes,
            "status": _trace_status(episodes),
            "models": models,
            "model_label": _model_label(models),
            "harnesses": sorted(harness_values) or ["unknown"],
            "judge_model": str(
                (record.get("analysis_provenance") or {}).get("agent_verification_model")
                or manifest.get("model") or "unknown"
            ),
            "validation_warnings": record.get("validation_warnings") or [],
        })
    status_order = {"confirmed": 0, "review": 1, "excluded": 2}
    traces.sort(key=lambda item: (status_order[item["status"]], item["trace_id"]))

    output_root = site_dir / "data" / "pain-report"
    cases_dir = output_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for old_case in cases_dir.glob("*.json"):
        old_case.unlink()

    case_index: list[dict[str, Any]] = []
    episode_counts: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter(trace["status"] for trace in traces)
    model_trace_counts: dict[str, Counter[str]] = defaultdict(Counter)
    display_number = 0
    for trace in traces:
        for model in trace["models"]:
            model_trace_counts[model][trace["status"]] += 1
        for episode in trace["episodes"]:
            display_number += 1
            index_record, detail = _public_case(episode, trace, display_number)
            episode_id = index_record["episode_id"]
            if not episode_id:
                raise ValueError(f"Verified episode in {trace['trace_id']} has no episode_id")
            _write_json(cases_dir / f"{episode_id}.json", detail)
            case_index.append(index_record)
            episode_counts[index_record["status"]] += 1

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    model_summary = []
    for model, counts in sorted(model_trace_counts.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        model_summary.append({
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "trace_count": sum(counts.values()),
            "confirmed_trace_count": counts["confirmed"],
            "review_trace_count": counts["review"],
            "excluded_trace_count": counts["excluded"],
        })
    summary = {
        "total_trace_count": int(manifest.get("total_trace_count") or 0),
        "completed_trace_count": int(manifest.get("completed_trace_count") or 0),
        "verified_trace_count": len(traces),
        "verified_episode_count": len(case_index),
        "trace_status_counts": dict(trace_counts),
        "episode_status_counts": dict(episode_counts),
        "model_summary": model_summary,
    }
    metadata = {
        "schema_version": "pain-report-public-v1",
        "generated_at": generated_at,
        "batch_id": manifest.get("batch_id"),
        "analysis_run_id": manifest.get("analysis_run_id"),
        "run_status": manifest.get("status"),
        "judge_models": sorted({trace["judge_model"] for trace in traces}),
        "user_prompt_versions": [
            value for value in (manifest.get("prompt_version_history") or [manifest.get("prompt_version")]) if value
        ],
        "agent_prompt_versions": [
            value for value in (manifest.get("attribution_prompt_version_history") or [manifest.get("attribution_prompt_version")]) if value
        ],
        "methodology": "先用完整 User Turn 筛选候选 Episode，再结合 Agent、Tool 和结果证据验证是否存在可归因于 Agent 的失败。",
        "comparison_caveat": "当前样本不是同任务、同环境下的受控模型实验；模型分布只描述本批候选，不能据此比较模型能力高低。",
        "privacy_note": "公开视图不包含完整 Trace、完整 User Turn、Tool Result、原始 Session 路径或本地绝对路径。",
    }
    _write_json(output_root / "metadata.json", metadata)
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "case-index.json", case_index)
    return {
        "status": "success",
        "output_dir": str(output_root.resolve()),
        "trace_count": len(traces),
        "episode_count": len(case_index),
        "confirmed_episode_count": episode_counts["confirmed"],
        "review_episode_count": episode_counts["review"],
        "excluded_episode_count": episode_counts["excluded"],
        "generated_at": generated_at,
    }
