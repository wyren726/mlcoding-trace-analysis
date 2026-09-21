from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "pain-case-v1"


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not an object: {path}")
            yield value


def _analysis_path(run: Path, batch_id: str, run_id: str) -> Path:
    candidates = [
        run / "03_agent_verify" / f"trace_analysis__{batch_id}__{run_id}.jsonl",
        run / "02_analyze" / f"trace_analysis__{batch_id}__{run_id}.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(run.glob("*/trace_analysis__*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(candidates[0])


def _case_id(batch_id: str, trace_id: str, episode_id: str) -> str:
    digest = hashlib.sha256(
        f"{batch_id}|{trace_id}|{episode_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"case_{digest}"


def _judgment_and_reasons(
    assessment: dict[str, Any], warnings: list[Any]
) -> tuple[str, list[str]]:
    """Explain the deterministic mapping from verification to case status."""
    related = assessment.get("agent_related")
    if related is True:
        return "confirmed", ["validated_agent_attribution"]
    if related is False:
        return "excluded", ["validated_non_agent_attribution"]
    warning_values = [str(value) for value in warnings]
    reasons: list[str] = []
    if any("unsupported_agent_attribution" in value for value in warning_values):
        reasons.append("unsupported_agent_attribution")
    if any("quote_not_exact" in value for value in warning_values):
        reasons.append("evidence_quote_not_exact")
    if any(
        value.endswith(":missing") or value.endswith(":incomplete")
        for value in warning_values
    ):
        reasons.append("required_analysis_fields_missing")
    if any("evidence_not_confirmed_sufficient" in value for value in warning_values):
        reasons.append("evidence_review_insufficient")
    if not str(assessment.get("agent_failure") or "").strip():
        reasons.append("agent_failure_not_determined")
    if not reasons:
        reasons.append("agent_attribution_unresolved")
    return "review", list(dict.fromkeys(reasons))


def _evidence_values(episode: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []

    def add(value: Any, role: str) -> None:
        if not isinstance(value, dict):
            return
        event_id = str(value.get("event_id") or "")
        if not event_id:
            return
        values.append({
            "evidence_role": role,
            "event_id": event_id,
            "turn_id": str(value.get("turn_id") or ""),
            "quote": str(value.get("quote") or ""),
        })

    for signal in (episode.get("candidate_signals") or {}).get("signals") or []:
        add(signal, "user_signal")
    preliminary = episode.get("preliminary_goal") or {}
    for requirement in preliminary.get("requirements") or []:
        for item in requirement.get("evidence") or []:
            add(item, "requirement")
    assessment = episode.get("agent_assessment") or {}
    for item in (assessment.get("goal_assessment") or {}).get("evidence") or []:
        add(item, "goal_assessment")
    for result in assessment.get("requirement_results") or []:
        for item in result.get("evidence") or []:
            add(item, "requirement_result")
    for gap in assessment.get("capability_gaps") or []:
        for item in gap.get("evidence") or []:
            add(item, "capability_gap")
    seen: set[tuple[str, str, str]] = set()
    result = []
    for value in values:
        key = (value["evidence_role"], value["event_id"], value["quote"])
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _trace_support(
    turn_path: Path, selected_ids: set[str], evidence_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metadata: dict[str, dict[str, Any]] = {}
    event_sources: dict[str, dict[str, Any]] = {}
    for turn in _read_jsonl(turn_path):
        trace_id = str(turn.get("trace_id") or "")
        if trace_id not in selected_ids:
            continue
        item = metadata.setdefault(trace_id, {
            "session_ids": set(), "models": set(), "harnesses": set(),
        })
        if turn.get("session_id"):
            item["session_ids"].add(str(turn["session_id"]))
        if turn.get("harness"):
            item["harnesses"].add(str(turn["harness"]))
        item["models"].update(
            str(value) for value in turn.get("agent_models") or []
            if value and value != "<synthetic>"
        )
        for event in turn.get("events") or []:
            event_id = str(event.get("event_id") or "")
            if event_id in evidence_ids:
                event_sources[event_id] = {
                    "event_id": event_id,
                    "turn_id": str(turn.get("turn_id") or ""),
                    "source": event.get("source") or {},
                }
    normalized = {
        trace_id: {
            "session_ids": sorted(value["session_ids"]),
            "models": sorted(value["models"]) or ["unknown"],
            "harnesses": sorted(value["harnesses"]) or ["unknown"],
        }
        for trace_id, value in metadata.items()
    }
    return normalized, event_sources


def build_pain_cases(run_dir: Path, batch_dir: Path) -> dict[str, Any]:
    """Build the one-row-per-Episode authoritative result without model calls."""
    run = run_dir.expanduser().resolve()
    batch = batch_dir.expanduser().resolve()
    batch_id = batch.name
    run_id = run.name
    analysis_path = _analysis_path(run, batch_id, run_id)
    records = [
        row for row in _read_jsonl(analysis_path)
        if (row.get("route") or {}).get("judgment") == "analyze"
    ]
    evidence_by_trace: dict[str, set[str]] = {}
    for record in records:
        ids = evidence_by_trace.setdefault(str(record.get("trace_id") or ""), set())
        for episode in record.get("task_episodes") or []:
            if isinstance(episode.get("agent_assessment"), dict):
                ids.update(item["event_id"] for item in _evidence_values(episode))
    all_evidence_ids = set().union(*evidence_by_trace.values()) if evidence_by_trace else set()
    metadata, event_sources = _trace_support(
        batch / "unified_turns.jsonl", set(evidence_by_trace), all_evidence_ids
    )

    output_dir = run / "04_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"pain_cases__{batch_id}__{run_id}.jsonl"
    temporary = output.with_suffix(f"{output.suffix}.tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    case_count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for trace in records:
            trace_id = str(trace.get("trace_id") or "")
            trace_meta = metadata.get(trace_id, {
                "session_ids": [], "models": ["unknown"], "harnesses": ["unknown"],
            })
            for episode in trace.get("task_episodes") or []:
                assessment = episode.get("agent_assessment")
                if not isinstance(assessment, dict):
                    continue
                episode_id = str(episode.get("episode_id") or "")
                if not episode_id:
                    raise ValueError(f"Verified Episode has no episode_id: {trace_id}")
                judgment, judgment_reasons = _judgment_and_reasons(
                    assessment, trace.get("validation_warnings") or []
                )
                evidence = _evidence_values(episode)
                source_refs = [
                    event_sources[item["event_id"]]
                    for item in evidence if item["event_id"] in event_sources
                ]
                value = {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": _case_id(batch_id, trace_id, episode_id),
                    "batch_id": batch_id,
                    "analysis_run_id": run_id,
                    "trace_id": trace_id,
                    "episode_id": episode_id,
                    "session_ids": trace_meta["session_ids"],
                    "models": trace_meta["models"],
                    "harnesses": trace_meta["harnesses"],
                    "pain_judgment": judgment,
                    "judgment_reason_codes": judgment_reasons,
                    "query_turn_id": episode.get("query_turn_id"),
                    "episode_boundary": episode.get("episode_boundary"),
                    "user_screen": {
                        "domain": episode.get("domain"),
                        "analysis_value": episode.get("analysis_value"),
                        "initial_query": episode.get("initial_query"),
                        "initial_query_clarity": episode.get("initial_query_clarity"),
                        "preliminary_goal": episode.get("preliminary_goal"),
                        "requirement_evolution": episode.get("requirement_evolution"),
                        "candidate_signals": episode.get("candidate_signals"),
                        "route": episode.get("route"),
                    },
                    "agent_verification": assessment,
                    "outcome": episode.get("outcome"),
                    "capability_gaps": assessment.get("capability_gaps") or [],
                    "evidence_references": evidence,
                    "source_references": source_refs,
                    "validation_warnings": trace.get("validation_warnings") or [],
                    "analysis_provenance": trace.get("analysis_provenance") or {},
                    "lineage": {
                        "analysis_input": str(analysis_path),
                        "preprocessed_input": str(batch / "unified_turns.jsonl"),
                    },
                }
                target.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[judgment] += 1
                case_count += 1
    temporary.replace(output)
    manifest = {
        "manifest_version": "1.0",
        "stage": "04_results",
        "status": "completed",
        "batch_id": batch_id,
        "analysis_run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "input_path": str(analysis_path),
        "output_path": str(output),
        "case_count": case_count,
        "judgment_counts": dict(counts),
        "generated_at": _now(),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return {**manifest, "output_dir": str(output_dir)}
