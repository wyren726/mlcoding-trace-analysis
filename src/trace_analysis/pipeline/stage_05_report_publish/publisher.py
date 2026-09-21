from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .._compat import _observable_turn, iter_trace_bundles


_PATH_RE = re.compile(r"(?<![\w.])/(?:[^\s/]+/)+[^\s,;，。；：:]+")
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SECRET_RE = re.compile(r"\b(?:sk|api)[-_][A-Za-z0-9_-]{16,}\b", re.IGNORECASE)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not an object: {path}")
            yield value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=9, mtime=0) as target:
        target.write(payload)
    temporary.replace(path)


def _public_text(value: Any, limit: int = 800) -> str:
    text = str(value or "").strip()
    text = _URL_RE.sub("[链接]", text)
    text = _EMAIL_RE.sub("[邮箱]", text)
    text = _SECRET_RE.sub("[凭据]", text)
    text = _PATH_RE.sub("[本地路径]", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _benchmark_note(screen: dict[str, Any], judgment: str) -> str:
    """Derive the stable, human-facing benchmark recommendation locally."""
    if judgment == "excluded":
        return (
            "不建议作为能力缺陷 Benchmark；当前证据更符合正常等待、任务推进、"
            "外部限制或用户新增要求。"
        )
    if judgment == "review":
        return "暂不进入 Benchmark 候选池；应先人工复核失败是否真实发生、是否由 Agent 导致。"
    clarity = (screen.get("initial_query_clarity") or {}).get("judgment")
    value = (screen.get("analysis_value") or {}).get("judgment")
    if clarity == "clear" and value in {"high", "medium"}:
        return (
            "建议进入 Benchmark 候选池；后续需要结合 Workspace 固化输入、环境、"
            "验收标准和参考结果。"
        )
    return "可以保留为 Benchmark 候选，但需要先补足任务输入、验收标准或初始 Query 的明确性。"


def _find_results(run: Path) -> tuple[Path, Path]:
    manifest = run / "04_results" / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    path = Path(str(data.get("output_path") or ""))
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    if not path.is_file():
        matches = sorted(manifest.parent.glob("pain_cases__*.jsonl"))
        if len(matches) != 1:
            raise FileNotFoundError(path)
        path = matches[0]
    return path, manifest


def _public_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    screen = case.get("user_screen") or {}
    verification = case.get("agent_verification") or {}
    goal = (verification.get("goal_assessment") or {}).get("goal")
    if not goal:
        goal = (screen.get("preliminary_goal") or {}).get("goal")
    gaps = [
        {
            "capability": _public_text(
                gap.get("capability_name") or gap.get("capability"), 120
            ),
            "capability_key": _public_text(gap.get("capability_key"), 120),
            "reason": _public_text(gap.get("reason"), 500),
            "evidence": [
                {
                    "event_id": item.get("event_id"),
                    "turn_id": item.get("turn_id"),
                    "quote": _public_text(item.get("quote"), 500),
                }
                for item in gap.get("evidence") or [] if isinstance(item, dict)
            ],
        }
        for gap in case.get("capability_gaps") or [] if isinstance(gap, dict)
    ]
    judgment = str(case.get("pain_judgment") or "review")
    status_labels = {"confirmed": "确认痛点", "review": "需复核", "excluded": "排除"}
    models = case.get("models") or ["unknown"]
    harnesses = case.get("harnesses") or ["unknown"]
    outcome = str(case.get("outcome") or "unknown")
    requirements = [
        {
            "text": _public_text(item.get("text"), 500),
            "origin": item.get("origin") or "unknown",
            "origin_label": item.get("origin") or "unknown",
            "status": item.get("status") or "unknown",
            "status_label": item.get("status") or "unknown",
        }
        for item in (screen.get("preliminary_goal") or {}).get("requirements") or []
        if isinstance(item, dict)
    ]
    # Some legacy Stage 02 records contain a verified goal but no requirement
    # decomposition. Preserve that distinction while keeping the detail page
    # useful; this is a display fallback, not a reconstructed analysis claim.
    if not requirements and judgment != "excluded" and goal:
        requirements = [{
            "text": _public_text(goal, 500),
            "origin": "goal_fallback",
            "origin_label": "由综合目标回填（原分析未逐项拆分）",
            "status": "unknown",
            "status_label": "未逐项判断",
        }]
    index = {
        "case_id": case.get("case_id"),
        "batch_id": case.get("batch_id"),
        "analysis_run_id": case.get("analysis_run_id"),
        "trace_id": case.get("trace_id"),
        "episode_id": case.get("episode_id"),
        "pain_judgment": judgment,
        "judgment_reason_codes": case.get("judgment_reason_codes") or [],
        "status": judgment,
        "status_label": status_labels.get(judgment, judgment),
        "title": _case_title(judgment, gaps, verification, goal),
        "summary": _public_text(
            verification.get("agent_failure")
            or verification.get("agent_related_reason") or goal,
            260,
        ),
        "models": models,
        "model_label": " + ".join(models),
        "harnesses": harnesses,
        "capability_gaps": [gap["capability"] for gap in gaps if gap["capability"]],
        "outcome": outcome,
        "outcome_label": outcome,
        "attribution": verification.get("attribution") or "unknown",
        "attribution_label": verification.get("attribution") or "unknown",
        "initial_query_clarity": (screen.get("initial_query_clarity") or {}).get("judgment") or "unknown",
        "initial_query_clarity_label": (screen.get("initial_query_clarity") or {}).get("judgment") or "unknown",
        "analysis_value": (screen.get("analysis_value") or {}).get("judgment") or "unknown",
        "analysis_value_label": (screen.get("analysis_value") or {}).get("judgment") or "unknown",
    }
    detail = {
        **index,
        "query_turn_id": case.get("query_turn_id"),
        "episode_boundary": case.get("episode_boundary"),
        "goal": _public_text(goal, 800),
        "goal_reason": _public_text(
            (verification.get("goal_assessment") or {}).get("reason")
            or (screen.get("preliminary_goal") or {}).get("reason"),
            800,
        ),
        "initial_query_clarity_reason": _public_text(
            (screen.get("initial_query_clarity") or {}).get("reason"), 800
        ),
        "analysis_value_reason": _public_text(
            (screen.get("analysis_value") or {}).get("reason"), 800
        ),
        "requirements": requirements,
        "initial_query_clarity": screen.get("initial_query_clarity"),
        "analysis_value": screen.get("analysis_value"),
        "candidate_signals": screen.get("candidate_signals"),
        "candidate_reason": _public_text(
            (screen.get("candidate_signals") or {}).get("reason"), 800
        ),
        "agent_failure": _public_text(verification.get("agent_failure"), 1000),
        "agent_related_reason": _public_text(
            verification.get("agent_related_reason"), 1000
        ),
        "capability_gap_details": gaps,
        "evidence_chain": [
            {
                "evidence_role": item.get("evidence_role"),
                "event_id": item.get("event_id"),
                "turn_id": item.get("turn_id"),
                "quote": _public_text(item.get("quote"), 500),
            }
            for item in case.get("evidence_references") or []
        ],
        "validation_warnings": [
            _public_text(value, 300) for value in case.get("validation_warnings") or []
        ],
        "benchmark_note": _benchmark_note(screen, judgment),
        "privacy_note": "公开视图不包含完整 Trace、Tool Result 或源文件绝对路径。",
    }
    return index, detail


def _summary(index: list[dict[str, Any]]) -> dict[str, Any]:
    judgments = Counter(str(item.get("pain_judgment") or "unknown") for item in index)
    models = Counter(model for item in index for model in item.get("models") or ["unknown"])
    harnesses = Counter(value for item in index for value in item.get("harnesses") or ["unknown"])
    gaps = Counter(value for item in index for value in item.get("capability_gaps") or [])
    return {
        "case_count": len(index),
        "pain_judgment_counts": dict(judgments),
        "model_counts": dict(models),
        "harness_counts": dict(harnesses),
        "capability_gap_counts": dict(gaps),
    }


def _case_title(
    judgment: str,
    gaps: list[dict[str, Any]],
    verification: dict[str, Any],
    goal: Any,
) -> str:
    """Keep list titles scannable; retain the full failure in summary/detail."""
    if judgment == "confirmed":
        names = [str(gap.get("capability") or "").strip() for gap in gaps]
        names = [name for name in names if name]
        if names:
            return _public_text("、".join(names[:2]), 72)
        return _public_text(verification.get("agent_failure") or goal or "确认痛点", 58)
    if judgment == "review":
        return _public_text(
            verification.get("agent_failure")
            or verification.get("agent_related_reason")
            or goal
            or "现有证据不足",
            58,
        )
    return "候选信号未构成 Agent 痛点"


def _event_text(event: dict[str, Any]) -> str:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    kind = str(event.get("type") or "")
    if kind == "tool_call":
        name = str(data.get("tool_name") or "未知工具")
        arguments = data.get("arguments")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False, indent=2)
        return f"{name}\n{arguments or ''}".strip()
    content = data.get("content")
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content or data.get("message") or "").strip()


def _transcript_event(event: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(event.get("type") or "")
    roles = {
        "user_message": "user", "assistant_message": "assistant",
        "tool_call": "tool", "tool_result": "tool", "error": "error",
    }
    if kind not in roles:
        return None
    original = _event_text(event)
    limit = 20_000 if kind in {"user_message", "assistant_message"} else 12_000
    content = _public_text(original, limit)
    if not content:
        return None
    return {
        "event_id": event.get("event_id"),
        "sequence": event.get("sequence"),
        "timestamp": event.get("timestamp"),
        "role": roles[kind],
        "type": kind,
        "content": content,
        "truncated": len(original) > limit,
        "original_char_count": len(original),
    }


def _write_transcripts(
    turn_path: Path,
    cases: list[dict[str, Any]],
    transcripts_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Stream the multi-GB Turn product once and emit one lazy asset per Episode."""
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_trace.setdefault(str(case.get("trace_id") or ""), []).append(case)
    if transcripts_dir.exists():
        shutil.rmtree(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, Any]] = {}
    for bundle in iter_trace_bundles(turn_path):
        selected = by_trace.get(bundle.trace_id)
        if not selected:
            continue
        positions = {
            str(turn.get("turn_id") or ""): index
            for index, turn in enumerate(bundle.turns)
        }
        for case in selected:
            boundary = case.get("episode_boundary") or {}
            start = positions.get(str(boundary.get("start_turn_id") or ""))
            end = positions.get(str(boundary.get("end_turn_id") or ""))
            if start is None or end is None or end < start:
                episode_turns = []
            else:
                episode_turns = bundle.turns[start : end + 1]
            turns = []
            case_id = str(case.get("case_id") or "")
            for turn in episode_turns:
                observable = _observable_turn(turn)
                events = [
                    value for event in observable.get("events") or []
                    if (value := _transcript_event(event)) is not None
                ]
                user = next((event for event in events if event["role"] == "user"), None)
                if user is None:
                    continue
                agent_events = [event for event in events if event is not user]
                turns.append({
                    "turn_id": observable.get("turn_id"),
                    "turn_index": observable.get("turn_index"),
                    "user": user,
                    "agent_event_count": len(agent_events),
                    "agent_events": agent_events,
                })
            transcript = {
                "schema_version": "pain-episode-transcript-v1",
                "case_id": case_id,
                "trace_id": bundle.trace_id,
                "episode_id": case.get("episode_id"),
                "turns": turns,
                "publication_note": (
                    "按 Episode 顺序展示可观察交互；已排除内部推理、遥测和系统事件，"
                    "并对公开内容脱敏。超长事件会标注截断。"
                ),
            }
            _atomic_gzip_json(transcripts_dir / f"{case_id}.json.gz", transcript)
            stats[case_id] = {
                "turn_count": len(turns),
                "event_count": sum(1 + turn["agent_event_count"] for turn in turns),
            }
    return stats


def _write_markdown(path: Path, metadata: dict[str, Any], index: list[dict[str, Any]]) -> None:
    summary = _summary(index)
    lines = [
        f"# 痛点分析报告：{metadata['batch_id']}", "",
        f"- 分析 Run：`{metadata['analysis_run_id']}`",
        f"- 案例数：{summary['case_count']}",
        f"- 确认痛点：{summary['pain_judgment_counts'].get('confirmed', 0)}",
        f"- 需复核：{summary['pain_judgment_counts'].get('review', 0)}",
        f"- 排除：{summary['pain_judgment_counts'].get('excluded', 0)}", "",
        "## 案例概览", "",
    ]
    for item in index:
        lines.extend([
            f"### {item['case_id']} · {item['title']}", "",
            f"- 判断：`{item['pain_judgment']}`",
            f"- 模型：{', '.join(item['models'])}",
            f"- Harness：{', '.join(item['harnesses'])}",
            f"- Trace：`{item['trace_id']}` / Episode：`{item['episode_id']}`",
            f"- 概述：{item['summary'] or '—'}", "",
        ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _rebuild_global(publication_root: Path, catalog: dict[str, Any]) -> None:
    all_index: list[dict[str, Any]] = []
    for batch in catalog.get("batches") or []:
        index_path = publication_root / str(batch["case_index"])
        if index_path.is_file():
            values = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(values, list):
                all_index.extend(value for value in values if isinstance(value, dict))
    summary = _summary(all_index)
    summary.update({
        "batch_count": len(catalog.get("batches") or []),
        "updated_at": _now(),
    })
    _atomic_json(publication_root / "summary.json", summary)
    indexes = publication_root / "indexes"
    for name, key in (
        ("models.json", "models"),
        ("harnesses.json", "harnesses"),
        ("capability-gaps.json", "capability_gaps"),
    ):
        counts = Counter(
            value for item in all_index for value in item.get(key) or []
        )
        _atomic_json(indexes / name, [
            {"value": value, "case_count": count}
            for value, count in counts.most_common()
        ])


def publish_pain_cases(
    run_dir: Path,
    publication_root: Path,
    *,
    refresh_existing: bool = False,
) -> dict[str, Any]:
    """Build one immutable batch bundle and atomically activate it in the catalog."""
    run = run_dir.expanduser().resolve()
    pain_path, result_manifest_path = _find_results(run)
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    batch_id = str(result_manifest["batch_id"])
    run_id = str(result_manifest["analysis_run_id"])
    cases = list(_read_jsonl(pain_path))
    metadata = {
        "schema_version": "pain-report-bundle-v1",
        "batch_id": batch_id,
        "analysis_run_id": run_id,
        "generated_at": _now(),
        "source_path": str(pain_path),
        "case_count": len(cases),
    }
    stage_dir = run / "05_report_publish"
    bundle = stage_dir / "web_bundle"
    cases_dir = bundle / "cases"
    transcripts_dir = bundle / "transcripts"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for old in cases_dir.glob("*.json"):
        old.unlink()
    index: list[dict[str, Any]] = []
    for case in cases:
        item, detail = _public_case(case)
        case_id = str(item.get("case_id") or "")
        if not case_id:
            raise ValueError("pain_case has no case_id")
        index.append(item)
        _atomic_json(cases_dir / f"{case_id}.json", detail)
    turn_candidates = [
        run.parents[2] / "preprocessed" / batch_id / "unified_turns.jsonl",
        run.parents[3] / "preprocessed" / batch_id / "unified_turns.jsonl",
    ]
    turn_path = next((path for path in turn_candidates if path.is_file()), turn_candidates[0])
    if not turn_path.is_file():
        raise FileNotFoundError(turn_path)
    transcript_stats = _write_transcripts(turn_path, cases, transcripts_dir)
    index.sort(key=lambda value: (
        {"confirmed": 0, "review": 1, "excluded": 2}.get(
            str(value.get("pain_judgment")), 3
        ),
        str(value.get("case_id")),
    ))
    for display_number, item in enumerate(index, 1):
        item["display_number"] = display_number
        detail_path = cases_dir / f"{item['case_id']}.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        detail["display_number"] = display_number
        relative_transcript = f"transcripts/{item['case_id']}.json.gz"
        item["transcript_path"] = relative_transcript
        detail["transcript_path"] = relative_transcript
        detail["transcript_stats"] = transcript_stats.get(item["case_id"], {})
        _atomic_json(detail_path, detail)
    _atomic_json(bundle / "metadata.json", metadata)
    _atomic_json(bundle / "summary.json", _summary(index))
    _atomic_json(bundle / "case-index.json", index)
    report = stage_dir / f"pain_analysis__{batch_id}__{run_id}.md"
    _write_markdown(report, metadata, index)

    publication = publication_root.expanduser().resolve()
    published_bundle = publication / "batches" / batch_id / run_id
    if not published_bundle.exists():
        published_bundle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, published_bundle)
    elif refresh_existing:
        # Analysis remains immutable; this only refreshes the derived public
        # representation after a publisher/UI bug fix.
        published_transcripts = published_bundle / "transcripts"
        if published_transcripts.exists():
            shutil.rmtree(published_transcripts)
        shutil.copytree(bundle, published_bundle, dirs_exist_ok=True)
    # A published Batch/Run bundle is immutable.  Re-publishing the same Run
    # only updates catalog activation; changed analysis must use a new run_id.
    catalog_path = publication / "catalog.json"
    catalog = (
        json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog_path.is_file() else {"schema_version": "pain-report-catalog-v1", "batches": []}
    )
    previous = {
        str(value.get("batch_id")): value
        for value in catalog.get("batches") or [] if isinstance(value, dict)
    }
    old = previous.get(batch_id, {})
    runs = sorted(set(old.get("available_run_ids") or []) | {run_id})
    previous[batch_id] = {
        "batch_id": batch_id,
        "active_run_id": run_id,
        "available_run_ids": runs,
        "published_at": _now(),
        "metadata": f"batches/{batch_id}/{run_id}/metadata.json",
        "summary": f"batches/{batch_id}/{run_id}/summary.json",
        "case_index": f"batches/{batch_id}/{run_id}/case-index.json",
        "cases_pattern": f"batches/{batch_id}/{run_id}/cases/{{case_id}}.json",
        "transcripts_pattern": f"batches/{batch_id}/{run_id}/transcripts/{{case_id}}.json.gz",
    }
    catalog.update({
        "updated_at": _now(),
        "batches": sorted(previous.values(), key=lambda value: value["batch_id"]),
    })
    _atomic_json(catalog_path, catalog)
    _rebuild_global(publication, catalog)
    manifest = {
        "manifest_version": "1.0",
        "stage": "05_report_publish",
        "status": "completed",
        "batch_id": batch_id,
        "analysis_run_id": run_id,
        "input_path": str(pain_path),
        "report_path": str(report),
        "bundle_path": str(bundle),
        "publication_path": str(published_bundle),
        "catalog_path": str(catalog_path),
        "case_count": len(index),
        "generated_at": _now(),
    }
    _atomic_json(stage_dir / "manifest.json", manifest)
    return manifest
