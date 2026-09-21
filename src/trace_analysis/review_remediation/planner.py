from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "review-remediation-v1"
RULE_VERSION = "review-remediation-rules-v1"
INTERRUPTION_PREFIX = "[Request interrupted by user"
SYNTHETIC_USER_PREFIXES = (
    "Base directory for this skill:",
    "[System reminder]",
    "<task-notification>",
    "Resume from the paused agent session.",
    "The previous request encountered a temporary API error.",
    "This session is being continued from a previous conversation",
    "[system] Previously read image bytes have been cleared",
)
INTERRUPTION_ATTRIBUTION_RE = re.compile(
    r"截断|中断|打断|未完整|未完成|未响应|没有回复|未交付|"
    r"interrupted|truncated|incomplete response",
    re.IGNORECASE,
)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256(
        "|".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for value in values:
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _result_path(analysis_root: Path, batch_id: str, run_id: str) -> Path:
    result_dir = analysis_root / batch_id / run_id / "04_results"
    manifest_path = result_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = Path(str(manifest.get("output_path") or ""))
        if value.is_file():
            return value.resolve()
        if value and not value.is_absolute():
            candidate = (manifest_path.parent / value).resolve()
            if candidate.is_file():
                return candidate
    matches = sorted(result_dir.glob("pain_cases__*.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot resolve one pain_cases JSONL for {batch_id}/{run_id}"
        )
    return matches[0].resolve()


def _load_active_cases(
    analysis_root: Path, publication_root: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Path]]:
    catalog_path = publication_root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    sources: dict[tuple[str, str], Path] = {}
    for batch in catalog.get("batches") or []:
        batch_id = str(batch.get("batch_id") or "")
        run_id = str(batch.get("active_run_id") or "")
        if not batch_id or not run_id:
            continue
        path = _result_path(analysis_root, batch_id, run_id)
        sources[(batch_id, run_id)] = path
        cases.extend(_read_jsonl(path))
    return cases, sources


def _load_review_cases(
    reviews: list[dict[str, Any]], analysis_root: Path
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for review in reviews:
        grouped[(str(review["batch_id"]), str(review["analysis_run_id"]))].add(
            str(review["case_id"])
        )
    found: dict[str, dict[str, Any]] = {}
    for (batch_id, run_id), case_ids in grouped.items():
        for case in _read_jsonl(_result_path(analysis_root, batch_id, run_id)):
            case_id = str(case.get("case_id") or "")
            if case_id in case_ids:
                found[case_id] = case
    missing = sorted(
        str(review["case_id"]) for review in reviews
        if str(review["case_id"]) not in found
    )
    if missing:
        raise ValueError(f"Reviewed cases are missing from source results: {missing}")
    return found


def _warning_signature(case: dict[str, Any]) -> dict[str, bool]:
    warnings = [str(value) for value in case.get("validation_warnings") or []]
    verification = case.get("agent_verification") or {}
    failure = str(verification.get("agent_failure") or "").strip()
    gaps = case.get("capability_gaps") or []
    related = verification.get("agent_related")
    unsupported = any(
        "unsupported_agent_attribution" in warning
        or ("capability_gaps" in warning and (
            "quote_not_exact" in warning or warning.endswith(":incomplete")
        ))
        for warning in warnings
    )
    unresolved = (
        case.get("pain_judgment") == "review"
        and related is None
        and not failure
        and not gaps
    )
    return {
        "unsupported_attribution_evidence": bool(unsupported or (
            failure and related is None and not gaps
        )),
        "unresolved_agent_verification": bool(unresolved),
    }


def _classify_review(
    review: dict[str, Any], case: dict[str, Any]
) -> list[dict[str, Any]]:
    note = str(review.get("note") or "")
    signatures = _warning_signature(case)
    codes: list[str] = []
    if any(value in note for value in ("打断", "中断", "截断", "interrupted")):
        codes.append("user_interruption_unmodeled")
    if signatures["unsupported_attribution_evidence"]:
        codes.append("unsupported_attribution_evidence")
    if signatures["unresolved_agent_verification"] or any(
        value in note for value in ("为什么要打成需复核", "门槛", "证据不足")
    ):
        codes.append("unresolved_agent_verification")
    if any(value in note for value in ("证据turn", "证据 turn", "找不到", "佐证")):
        codes.append("evidence_presentation_check")
    if not codes:
        codes.append("manual_triage_required")

    definitions = {
        "user_interruption_unmodeled": (
            "03_agent_verify", "analysis_logic",
            "用户主动打断事件未进入验证输入，可能把用户控制行为误归因为Agent自行截断。",
        ),
        "unsupported_attribution_evidence": (
            "03_agent_verify", "evidence_validation",
            "失败描述仍存在，但能力缺口或精确证据没有通过校验，归因不能作为已支持结论。",
        ),
        "unresolved_agent_verification": (
            "03_agent_verify", "analysis_logic",
            "结果处于review/null，但没有足够结构化理由说明需要补什么证据或为何不能排除。",
        ),
        "evidence_presentation_check": (
            "05_report_publish", "publication",
            "需要核对已验证的证据是否进入公开详情；若源结果本身无有效证据，应先修复分析。",
        ),
        "manual_triage_required": (
            "manual", "unknown",
            "自由文本复核无法由现有确定性规则安全归类，需要先人工指定问题类型。",
        ),
    }
    issues = []
    for code in dict.fromkeys(codes):
        stage, scope, hypothesis = definitions[code]
        issues.append({
            "schema_version": f"{SCHEMA_VERSION}-issue",
            "issue_id": _stable_id("issue", review["review_id"], code),
            "issue_code": code,
            "issue_scope": scope,
            "affected_stage": stage,
            "hypothesis": hypothesis,
            "review_id": review["review_id"],
            "case_id": review["case_id"],
            "batch_id": review["batch_id"],
            "analysis_run_id": review["analysis_run_id"],
            "trace_id": review["trace_id"],
            "episode_id": review["episode_id"],
            "human_decision": review.get("decision"),
            "review_note": note,
            "rule_version": RULE_VERSION,
        })
    return issues


def _event_content(event: dict[str, Any]) -> str:
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    value = data.get("content") or data.get("message") or ""
    return value if isinstance(value, str) else ""


def _is_real_user_content(content: str) -> bool:
    return bool(content.strip()) and not content.startswith(
        (INTERRUPTION_PREFIX, *SYNTHETIC_USER_PREFIXES)
    )


def _scan_trace_structure(
    preprocessed_root: Path,
    cases: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], dict[str, int]],
]:
    """Read only selected traces and retain interruption/follow-up metadata."""
    traces_by_batch: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        traces_by_batch[str(case.get("batch_id") or "")].add(
            str(case.get("trace_id") or "")
        )
    interruptions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    positions: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for batch_id, trace_ids in traces_by_batch.items():
        path = preprocessed_root / batch_id / "unified_turns.jsonl"
        if not path.is_file():
            continue
        trace_turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for turn in _read_jsonl(path):
            trace_id = str(turn.get("trace_id") or "")
            if trace_id not in trace_ids:
                continue
            turn_id = str(turn.get("turn_id") or "")
            try:
                turn_index = int(turn.get("turn_index") or 0)
            except (TypeError, ValueError):
                turn_index = 0
            positions[(batch_id, trace_id)][turn_id] = turn_index
            turn_interruptions = []
            real_user = None
            for event in turn.get("events") or []:
                if event.get("type") != "user_message":
                    continue
                content = _event_content(event)
                if content.startswith(INTERRUPTION_PREFIX):
                    turn_interruptions.append({
                        "event_id": str(event.get("event_id") or ""),
                        "quote": content[:200],
                    })
                elif real_user is None and _is_real_user_content(content):
                    real_user = {
                        "event_id": str(event.get("event_id") or ""),
                        "quote": content[:300],
                    }
            trace_turns[trace_id].append({
                "turn_id": turn_id,
                "turn_index": turn_index,
                "interruptions": turn_interruptions,
                "real_user": real_user,
            })
        for trace_id, turns in trace_turns.items():
            turns.sort(key=lambda value: (value["turn_index"], value["turn_id"]))
            for index, turn in enumerate(turns):
                if not turn["interruptions"]:
                    continue
                followup = next(
                    (value for value in turns[index + 1:] if value["real_user"]), None
                )
                for event in turn["interruptions"]:
                    interruptions[(batch_id, trace_id)].append({
                        "turn_id": turn["turn_id"],
                        "turn_index": turn["turn_index"],
                        "event_id": event["event_id"],
                        "quote": event["quote"],
                        "next_real_user_turn_id": (
                            followup["turn_id"] if followup else None
                        ),
                        "next_real_user_quote": (
                            followup["real_user"]["quote"] if followup else None
                        ),
                    })
    return interruptions, positions


def _episode_interruptions(
    case: dict[str, Any],
    values: list[dict[str, Any]],
    positions: dict[str, int],
) -> list[dict[str, Any]]:
    boundary = case.get("episode_boundary") or {}
    start_index = positions.get(str(boundary.get("start_turn_id") or ""))
    end_index = positions.get(str(boundary.get("end_turn_id") or ""))
    if start_index is None or end_index is None or end_index < start_index:
        return []
    return [
        value for value in values
        if start_index <= int(value.get("turn_index") or 0) <= end_index
    ]


def _published_evidence_missing(
    case: dict[str, Any], publication_root: Path
) -> bool:
    source_ids = {
        str(value.get("event_id") or "")
        for value in case.get("evidence_references") or []
        if isinstance(value, dict) and value.get("event_id")
    }
    if not source_ids:
        return False
    path = (
        publication_root / "batches" / str(case.get("batch_id"))
        / str(case.get("analysis_run_id")) / "cases"
        / f"{case.get('case_id')}.json"
    )
    if not path.is_file():
        return True
    detail = json.loads(path.read_text(encoding="utf-8"))
    published_ids = {
        str(value.get("event_id") or "")
        for value in detail.get("evidence_chain") or []
        if isinstance(value, dict) and value.get("event_id")
    }
    return not source_ids.issubset(published_ids)


def _impact_records(
    active_cases: list[dict[str, Any]],
    review_cases: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    interruptions: dict[tuple[str, str], list[dict[str, Any]]],
    positions: dict[tuple[str, str], dict[str, int]],
    publication_root: Path,
) -> list[dict[str, Any]]:
    enabled = {str(issue["issue_code"]) for issue in issues}
    reviewed_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        reviewed_by_case[str(issue["case_id"])].append(issue)
    cases_by_id = {str(case.get("case_id") or ""): case for case in active_cases}
    # Keep a reviewed seed visible even if its Run is no longer active.
    for case_id, case in review_cases.items():
        cases_by_id.setdefault(case_id, case)
    values = []
    for case_id, case in cases_by_id.items():
        signatures = _warning_signature(case)
        matched: list[str] = []
        reasons: list[str] = []
        trace_key = (
            str(case.get("batch_id") or ""), str(case.get("trace_id") or "")
        )
        episode_interruptions = _episode_interruptions(
            case, interruptions.get(trace_key, []), positions.get(trace_key, {})
        )
        if "user_interruption_unmodeled" in enabled and episode_interruptions:
            matched.append("user_interruption_unmodeled")
            reasons.append(
                f"Episode范围内发现{len(episode_interruptions)}个用户主动打断事件"
            )
        for code in (
            "unsupported_attribution_evidence", "unresolved_agent_verification"
        ):
            if code in enabled and signatures[code]:
                matched.append(code)
                reasons.append({
                    "unsupported_attribution_evidence": (
                        "存在失败归因，但能力缺口/精确证据未通过校验"
                    ),
                    "unresolved_agent_verification": (
                        "结果为review且未确认失败、归因或能力缺口"
                    ),
                }[code])
        if (
            "evidence_presentation_check" in enabled
            and _published_evidence_missing(case, publication_root)
        ):
            matched.append("evidence_presentation_check")
            reasons.append("源结果中的证据没有完整进入已发布详情")
        seed_issues = reviewed_by_case.get(case_id, [])
        for issue in seed_issues:
            code = str(issue["issue_code"])
            if code not in matched:
                matched.append(code)
                reasons.append("人工复核种子案例")
        matched = list(dict.fromkeys(matched))
        if not matched:
            continue
        verification = case.get("agent_verification") or {}
        attribution_text = " ".join((
            str(verification.get("agent_failure") or ""),
            str(verification.get("agent_related_reason") or ""),
        ))
        interruption_direct_risk = bool(
            episode_interruptions and INTERRUPTION_ATTRIBUTION_RE.search(attribution_text)
        )
        reanalysis_recommended = bool(
            seed_issues
            or signatures["unsupported_attribution_evidence"]
            or signatures["unresolved_agent_verification"]
            or interruption_direct_risk
            or "evidence_presentation_check" in matched
        )
        values.append({
            "schema_version": f"{SCHEMA_VERSION}-affected-case",
            "impact_id": _stable_id(
                "impact", case.get("batch_id"), case.get("analysis_run_id"), case_id
            ),
            "case_id": case_id,
            "batch_id": case.get("batch_id"),
            "baseline_analysis_run_id": case.get("analysis_run_id"),
            "trace_id": case.get("trace_id"),
            "episode_id": case.get("episode_id"),
            "pain_judgment": case.get("pain_judgment"),
            "matched_issue_codes": matched,
            "match_reasons": reasons,
            "is_review_seed": bool(seed_issues),
            "trigger_review_ids": sorted({
                str(issue["review_id"]) for issue in seed_issues
            }),
            "agent_related": verification.get("agent_related"),
            "has_agent_failure": bool(str(
                verification.get("agent_failure") or ""
            ).strip()),
            "capability_gap_count": len(case.get("capability_gaps") or []),
            "validation_warnings": case.get("validation_warnings") or [],
            "interruption_events": episode_interruptions,
            "interruption_direct_attribution_risk": interruption_direct_risk,
            "reanalysis_recommended": reanalysis_recommended,
        })
    return sorted(values, key=lambda value: (
        str(value["batch_id"]), str(value["trace_id"]), str(value["episode_id"])
    ))


def _queue_records(affected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in affected:
        if not value.get("reanalysis_recommended"):
            continue
        grouped[(
            str(value["batch_id"]), str(value["baseline_analysis_run_id"]),
            str(value["trace_id"]),
        )].append(value)
    queue = []
    for (batch_id, run_id, trace_id), cases in grouped.items():
        codes = sorted({
            code for case in cases for code in case["matched_issue_codes"]
        })
        analysis_codes = set(codes) - {
            "evidence_presentation_check", "manual_triage_required"
        }
        if analysis_codes:
            mode = "agent_verify_only"
            from_stage = "03_agent_verify"
        elif "evidence_presentation_check" in codes:
            mode = "republish_only"
            from_stage = "05_report_publish"
        else:
            mode = "manual_triage"
            from_stage = "manual"
        seed = any(case["is_review_seed"] for case in cases)
        priority = (
            "high"
            if seed or "unsupported_attribution_evidence" in codes
            else "medium"
        )
        queue.append({
            "schema_version": f"{SCHEMA_VERSION}-queue-item",
            "queue_id": _stable_id("queue", batch_id, run_id, trace_id, *codes),
            "batch_id": batch_id,
            "baseline_analysis_run_id": run_id,
            "trace_id": trace_id,
            "case_ids": sorted(str(case["case_id"]) for case in cases),
            "episode_ids": sorted(str(case["episode_id"]) for case in cases),
            "issue_codes": codes,
            "trigger_review_ids": sorted({
                review_id for case in cases
                for review_id in case["trigger_review_ids"]
            }),
            "reanalysis_mode": mode,
            "rerun_from_stage": from_stage,
            "priority": priority,
            "approval_status": "pending",
            "automatic_relabel_allowed": False,
        })
    return sorted(queue, key=lambda value: (
        {"high": 0, "medium": 1}.get(value["priority"], 2),
        value["batch_id"], value["trace_id"],
    ))


def plan_review_remediation(
    *,
    reviews_path: Path,
    analysis_root: Path,
    preprocessed_root: Path,
    publication_root: Path,
    output_root: Path,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    """Create an auditable dry-run impact scan without calling a model."""
    reviews_file = reviews_path.expanduser().resolve()
    analysis = analysis_root.expanduser().resolve()
    preprocessed = preprocessed_root.expanduser().resolve()
    publication = publication_root.expanduser().resolve()
    reviews = list(_read_jsonl(reviews_file))
    if not reviews:
        raise ValueError("No review decisions were provided")
    active_cases, result_sources = _load_active_cases(analysis, publication)
    reviewed_cases = _load_review_cases(reviews, analysis)
    issues = [
        issue for review in reviews
        for issue in _classify_review(review, reviewed_cases[str(review["case_id"])])
    ]
    scan_cases_by_key = {
        (
            str(case.get("batch_id") or ""),
            str(case.get("analysis_run_id") or ""),
            str(case.get("case_id") or ""),
        ): case
        for case in [*active_cases, *reviewed_cases.values()]
    }
    scan_cases = list(scan_cases_by_key.values())
    interruptions, positions = _scan_trace_structure(preprocessed, scan_cases)
    affected = _impact_records(
        active_cases, reviewed_cases, issues, interruptions, positions, publication
    )
    queue = _queue_records(affected)

    if cycle_id is None:
        stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        digest = hashlib.sha256(
            "|".join(sorted(str(value["review_id"]) for value in reviews)).encode()
        ).hexdigest()[:8]
        cycle_id = f"review_cycle_{stamp}_{digest}"
    cycle_dir = output_root.expanduser().resolve() / cycle_id
    if cycle_dir.exists():
        raise FileExistsError(f"Review cycle already exists: {cycle_dir}")

    snapshot = cycle_dir / "01_review_snapshot" / "review_decisions.jsonl"
    issue_path = cycle_dir / "02_issue_classification" / "review_issues.jsonl"
    affected_path = cycle_dir / "03_impact_scan" / "affected_cases.jsonl"
    impact_summary_path = cycle_dir / "03_impact_scan" / "summary.json"
    queue_path = cycle_dir / "04_reanalysis_queue" / "reanalysis_queue.jsonl"
    _atomic_jsonl(snapshot, reviews)
    _atomic_jsonl(issue_path, issues)
    _atomic_jsonl(affected_path, affected)
    _atomic_json(impact_summary_path, {
        "schema_version": f"{SCHEMA_VERSION}-impact-summary",
        "active_case_count": len(active_cases),
        "affected_case_count": len(affected),
        "reanalysis_recommended_case_count": sum(
            1 for value in affected if value.get("reanalysis_recommended")
        ),
        "user_interruption_case_count": sum(
            1 for value in affected
            if "user_interruption_unmodeled" in value["matched_issue_codes"]
        ),
        "user_interruption_direct_risk_case_count": sum(
            1 for value in affected
            if value.get("interruption_direct_attribution_risk")
        ),
        "issue_match_counts": dict(Counter(
            code for value in affected for code in value["matched_issue_codes"]
        )),
        "judgment_counts": dict(Counter(
            str(value.get("pain_judgment") or "unknown") for value in affected
        )),
    })
    _atomic_jsonl(queue_path, queue)
    for batch_id in sorted({str(value["batch_id"]) for value in queue}):
        _atomic_jsonl(
            cycle_dir / "04_reanalysis_queue" / f"trace_queue__{batch_id}.jsonl",
            (value for value in queue if value["batch_id"] == batch_id),
        )

    issue_counts = Counter(str(value["issue_code"]) for value in issues)
    impact_counts = Counter(
        code for value in affected for code in value["matched_issue_codes"]
    )
    mode_counts = Counter(str(value["reanalysis_mode"]) for value in queue)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "planned",
        "execution_mode": "dry_run_no_model_calls",
        "rule_version": RULE_VERSION,
        "created_at": _now(),
        "inputs": {
            "reviews_path": str(reviews_file),
            "analysis_root": str(analysis),
            "preprocessed_root": str(preprocessed),
            "publication_root": str(publication),
            "active_result_paths": [
                str(value) for _, value in sorted(result_sources.items())
            ],
        },
        "counts": {
            "review_decisions": len(reviews),
            "classified_issues": len(issues),
            "issue_codes": dict(issue_counts),
            "active_cases_scanned": len(active_cases),
            "affected_cases": len(affected),
            "affected_issue_matches": dict(impact_counts),
            "queued_traces": len(queue),
            "queue_modes": dict(mode_counts),
        },
        "outputs": {
            "review_snapshot": str(snapshot),
            "review_issues": str(issue_path),
            "affected_cases": str(affected_path),
            "impact_summary": str(impact_summary_path),
            "reanalysis_queue": str(queue_path),
        },
        "safety": {
            "original_results_modified": False,
            "model_calls_started": False,
            "automatic_relabel_allowed": False,
            "publication_activated": False,
            "next_gate": "人工核对影响范围并批准具体queue item后才能补跑",
        },
    }
    _atomic_json(cycle_dir / "cycle_manifest.json", manifest)
    return {**manifest, "cycle_dir": str(cycle_dir)}


__all__ = ["plan_review_remediation"]
