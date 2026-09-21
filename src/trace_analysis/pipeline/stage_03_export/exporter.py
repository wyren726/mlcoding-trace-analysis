from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from ...preprocessing.adapters.common import stable_id
from ...registries import registry_root_for_output, write_immutable_record
from ..stage_02_analyze.runner import iter_trace_bundles, user_trace_payload


EXPORT_SCHEMA_VERSION = "3.0"
EXPORT_PRODUCT_VERSION = "3.1"


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(_json_line(row))
            count += 1
    temporary.replace(path)
    return count


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def _labels(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value and value != "unknown" else []
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if item and str(item) != "unknown"})


def _trace_metadata(turn_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for bundle in iter_trace_bundles(turn_path):
        harnesses: set[str] = set()
        models: set[str] = set()
        api_providers: set[str] = set()
        sessions: set[str] = set()
        for turn in bundle.turns:
            harnesses.update(_labels(turn.get("harness")))
            models.update(_labels(turn.get("agent_models")))
            api_providers.update(_labels(turn.get("api_provider")))
            api_providers.update(_labels(turn.get("api_providers")))
            for event in turn.get("events") or []:
                if event.get("type") != "model_call" or not isinstance(event.get("data"), dict):
                    continue
                api_providers.update(_labels(event["data"].get("provider")))
                api_providers.update(_labels(event["data"].get("api_provider")))
            if turn.get("session_id"):
                sessions.add(str(turn["session_id"]))
        result[bundle.trace_id] = {
            "harnesses": sorted(harnesses),
            "models": sorted(models),
            "api_providers": sorted(api_providers),
            "session_ids": sorted(sessions),
            "metadata_complete": bool(harnesses and models and api_providers),
        }
    return result


def _verified_user_turn_views(
    turn_path: Path, trace_ids: set[str], metadata: dict[str, dict[str, Any]],
    batch_id: str, run_id: str,
) -> list[dict[str, Any]]:
    """Build reviewable user-only views with exact source-record pointers."""
    result = []
    for bundle in iter_trace_bundles(turn_path):
        if bundle.trace_id not in trace_ids:
            continue
        payload = user_trace_payload(bundle)
        source_by_event = {
            str(event.get("event_id")): event.get("source")
            for turn in bundle.turns
            for event in turn.get("events") or []
            if event.get("event_id") and isinstance(event.get("source"), dict)
        }
        user_turns = []
        source_records = []
        for turn in payload.get("turns") or []:
            message = turn.get("user_message") or {}
            event_id = str(message.get("event_id") or "")
            data = message.get("data") if isinstance(message.get("data"), dict) else {}
            user_turns.append({
                "turn_id": turn.get("turn_id"),
                "turn_index": turn.get("turn_index"),
                "event_id": event_id,
                "timestamp": message.get("timestamp"),
                "content": data.get("content"),
            })
            source = source_by_event.get(event_id)
            if source:
                source_records.append({
                    "event_id": event_id,
                    "input_file": source.get("input_file"),
                    "record_index": source.get("record_index"),
                })
        trace_meta = metadata.get(bundle.trace_id) or {}
        source_files = sorted({
            str(row["input_file"]) for row in source_records if row.get("input_file")
        })
        sessions = trace_meta.get("session_ids") or []
        harnesses = trace_meta.get("harnesses") or []
        result.append({
            "schema_version": "1.0",
            "batch_id": batch_id,
            "analysis_run_id": run_id,
            "trace_id": bundle.trace_id,
            "session_id": sessions[0] if len(sessions) == 1 else None,
            "harness": harnesses[0] if len(harnesses) == 1 else harnesses,
            "agent_models": trace_meta.get("models") or [],
            "input_view": "all_real_user_turns",
            "user_turn_count": len(user_turns),
            "user_turns": user_turns,
            "source_files": source_files,
            "source_records": source_records,
        })
    return result


def _normalised_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _distribution(rows: list[dict[str, Any]], *, label_field: str, id_field: str,
                  total_label: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        label = str(row.get(label_field) or "").strip()
        if not label:
            continue
        key = _normalised_label(label)
        labels.setdefault(key, label)
        groups[key].append(row)
    total = sum(len(members) for members in groups.values())
    result = []
    for key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        result.append({
            "schema_version": EXPORT_SCHEMA_VERSION,
            "label": labels[key],
            "normalised_label": key,
            "member_count": len(members),
            "trace_count": len({str(row.get("trace_id")) for row in members}),
            total_label: total,
            "share": (len(members) / total) if total else 0.0,
            "member_ids": [str(row[id_field]) for row in members],
        })
    return result


def _find_analysis_file(stage_dir: Path, batch_id: str, run_id: str) -> Path:
    expected = stage_dir / f"trace_analysis__{batch_id}__{run_id}.jsonl"
    if expected.is_file():
        return expected
    matches = sorted(stage_dir.glob("trace_analysis__*.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(expected)
    return matches[0]


def export_trace_analysis(run_dir: Path, batch_dir: Path) -> dict[str, Any]:
    """Build human-reviewable instance and distribution views without more model calls."""
    run = run_dir.expanduser().resolve()
    batch = batch_dir.expanduser().resolve()
    analyze_dir = run / "02_analyze"
    analyze_manifest = json.loads((analyze_dir / "manifest.json").read_text(encoding="utf-8"))
    run_id = str(analyze_manifest["analysis_run_id"])
    batch_id = str(analyze_manifest["batch_id"])
    if batch.name != batch_id:
        raise ValueError("Analysis run and preprocessing batch do not match")
    analysis_file = _find_analysis_file(analyze_dir, batch_id, run_id)
    metadata = _trace_metadata(batch / "unified_turns.jsonl")

    requirements: list[dict[str, Any]] = []
    pains: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    trace_count = 0
    review_count = 0
    analyzed_trace_ids: set[str] = set()
    agent_verified_traces: list[dict[str, Any]] = []
    for trace in _read_jsonl(analysis_file):
        trace_count += 1
        review_count += int(trace.get("analysis_status") == "needs_review")
        trace_id = str(trace.get("trace_id") or "")
        analyzed_trace_ids.add(trace_id)
        if (
            (trace.get("route") or {}).get("judgment") == "analyze"
            and any(
                isinstance(episode, dict)
                and isinstance(episode.get("agent_assessment"), dict)
                for episode in trace.get("task_episodes") or []
            )
        ):
            agent_verified_traces.append(trace)
        trace_meta = metadata.get(trace_id, {
            "harnesses": [], "models": [], "api_providers": [],
            "session_ids": [], "metadata_complete": False,
        })
        common = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "batch_id": batch_id,
            "analysis_run_id": run_id,
            "trace_id": trace_id,
            "analysis_status": trace.get("analysis_status"),
            "validation_warnings": trace.get("validation_warnings") or [],
            **trace_meta,
        }
        for episode in trace.get("task_episodes") or []:
            if not isinstance(episode, dict):
                continue
            episode_id = str(episode.get("episode_id") or "")
            preliminary = episode.get("preliminary_goal")
            agent = episode.get("agent_assessment")
            goal_assessment = (
                agent.get("goal_assessment") if isinstance(agent, dict) else None
            )
            if isinstance(goal_assessment, dict) and goal_assessment.get("judgment") in {
                "confirmed", "revised",
            }:
                goal = goal_assessment.get("goal")
            elif isinstance(preliminary, dict):
                goal = preliminary.get("goal")
            else:
                goal = episode.get("goal")  # v1/v2 compatibility
            episode_common = {
                **common,
                "episode_id": episode_id,
                "query_turn_id": episode.get("query_turn_id"),
                "goal": goal,
                "preliminary_goal": (
                    preliminary.get("goal") if isinstance(preliminary, dict) else None
                ),
                "goal_assessment": goal_assessment,
                "route": episode.get("route"),
                "outcome": episode.get("outcome"),
            }
            source_requirements = (
                preliminary.get("requirements")
                if isinstance(preliminary, dict) else episode.get("requirements")
            )
            for requirement in source_requirements or []:
                if not isinstance(requirement, dict):
                    continue
                requirements.append({
                    **episode_common,
                    "requirement_id": requirement.get("requirement_id"),
                    "requirement": requirement.get("text"),
                    # v3 deliberately defers capability classification until the
                    # complete Episode has confirmed the requirement.
                    "capability": requirement.get("capability"),
                    "origin": requirement.get("origin"),
                    "status": requirement.get("status"),
                    "evidence": requirement.get("evidence") or [],
                })
            # v3 exports pain only after full observable Agent verification.
            # The v2 branch remains so already generated pilot records are readable.
            if episode.get("pain_confirmed") is True and isinstance(agent, dict):
                candidate_signals = episode.get("candidate_signals")
                pain = {
                    "candidate_signals": (
                        candidate_signals.get("signals")
                        if isinstance(candidate_signals, dict) else []
                    ),
                    "initial_instruction_clarity": (
                        episode.get("initial_query_clarity") or {}
                    ).get("judgment"),
                    "initial_instruction_clarity_reason": (
                        episode.get("initial_query_clarity") or {}
                    ).get("reason"),
                    "pain_confirmed": True,
                    **agent,
                }
            elif episode.get("user_pain") is True:
                agent = episode.get("agent_assessment")
                if not isinstance(agent, dict):
                    agent = {
                        "agent_failure": None,
                        "agent_related": None,
                        "agent_related_reason": "",
                        "attribution": "unknown",
                        "capability_gaps": [],
                    }
                pain = {
                    "dissatisfaction_evidence": episode.get("dissatisfaction_evidence") or [],
                    "initial_instruction_clarity": episode.get("initial_query_clarity"),
                    "initial_instruction_clarity_reason": episode.get(
                        "initial_query_clarity_reason"
                    ),
                    "user_pain": True,
                    "user_pain_reason": episode.get("user_pain_reason"),
                    **agent,
                }
            else:
                pain = episode.get("pain_assessment")
            if not isinstance(pain, dict):
                continue
            pain_id = stable_id("pain", trace_id, episode_id)
            pains.append({
                **episode_common,
                "pain_episode_id": pain_id,
                **pain,
            })
            if pain.get("agent_related") is True:
                for gap in pain.get("capability_gaps") or []:
                    if not isinstance(gap, dict):
                        continue
                    gaps.append({
                        **episode_common,
                        "pain_episode_id": pain_id,
                        "gap_id": gap.get("gap_id"),
                        "capability": gap.get("capability"),
                        "reason": gap.get("reason"),
                        "attribution": pain.get("attribution"),
                        "evidence": gap.get("evidence") or [],
                    })

    output_dir = run / "03_export"
    output_dir.mkdir(parents=True, exist_ok=True)
    requirement_path = output_dir / "requirements.jsonl"
    pain_path = output_dir / "pain_episodes.jsonl"
    gap_path = output_dir / "capability_gaps.jsonl"
    requirement_distribution_path = output_dir / "capability_requirement_distribution.jsonl"
    gap_distribution_path = output_dir / "capability_gap_distribution.jsonl"
    configuration_path = output_dir / "model_harness_provider_summary.jsonl"
    summary_path = output_dir / "summary.jsonl"
    verified_trace_path = (
        output_dir / f"agent_verified_traces__{batch_id}__{run_id}.jsonl"
    )
    verified_user_path = (
        output_dir / f"agent_verified_user_turns__{batch_id}__{run_id}.jsonl"
    )
    _write_jsonl(requirement_path, requirements)
    _write_jsonl(pain_path, pains)
    _write_jsonl(gap_path, gaps)
    _write_jsonl(requirement_distribution_path, _distribution(
        requirements, label_field="capability", id_field="requirement_id",
        total_label="total_requirement_count",
    ))
    _write_jsonl(gap_distribution_path, _distribution(
        gaps, label_field="capability", id_field="gap_id",
        total_label="total_agent_related_gap_count",
    ))
    _write_jsonl(verified_trace_path, agent_verified_traces)
    _write_jsonl(verified_user_path, _verified_user_turn_views(
        batch / "unified_turns.jsonl",
        {str(row.get("trace_id")) for row in agent_verified_traces},
        metadata, batch_id, run_id,
    ))

    ConfigurationKey = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    configuration_groups: dict[ConfigurationKey, dict[str, Any]] = {}
    traces_by_configuration: dict[ConfigurationKey, set[str]] = defaultdict(set)
    requirements_by_configuration: dict[ConfigurationKey, set[str]] = defaultdict(set)
    pains_by_configuration: dict[ConfigurationKey, set[str]] = defaultdict(set)
    for trace_id, trace_meta in metadata.items():
        if trace_id not in analyzed_trace_ids:
            continue
        key = (
            tuple(trace_meta["models"]), tuple(trace_meta["harnesses"]),
            tuple(trace_meta["api_providers"]),
        )
        traces_by_configuration[key].add(trace_id)
    for row in requirements:
        key = (
            tuple(row["models"]), tuple(row["harnesses"]),
            tuple(row["api_providers"]),
        )
        traces_by_configuration[key].add(str(row["trace_id"]))
        requirements_by_configuration[key].add(str(row["requirement_id"]))
    for row in pains:
        key = (
            tuple(row["models"]), tuple(row["harnesses"]),
            tuple(row["api_providers"]),
        )
        traces_by_configuration[key].add(str(row["trace_id"]))
        if row.get("agent_related") is True:
            pains_by_configuration[key].add(str(row["pain_episode_id"]))
    for key in set(traces_by_configuration) | set(requirements_by_configuration) | set(pains_by_configuration):
        models, harnesses, api_providers = key
        configuration_groups[key] = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "configuration_id": stable_id(
                "configuration", models, harnesses, api_providers
            ),
            "models": list(models),
            "harnesses": list(harnesses),
            "api_providers": list(api_providers),
            "metadata_complete": bool(models and harnesses and api_providers),
            "trace_count": len(traces_by_configuration[key]),
            "requirement_count": len(requirements_by_configuration[key]),
            "agent_related_pain_count": len(pains_by_configuration[key]),
            "note": "descriptive counts only; they do not establish model or harness causality",
        }
    _write_jsonl(configuration_path, sorted(
        configuration_groups.values(), key=lambda row: (-row["trace_count"], row["configuration_id"])
    ))

    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "batch_id": batch_id,
        "analysis_run_id": run_id,
        "trace_count": trace_count,
        "needs_review_trace_count": review_count,
        "requirement_count": len(requirements),
        "pain_episode_count": len(pains),
        "agent_related_pain_count": sum(row.get("agent_related") is True for row in pains),
        "capability_gap_count": len(gaps),
        "generated_at": _now(),
    }
    _write_jsonl(summary_path, [summary])
    manifest = {
        "manifest_version": "1.0",
        "stage": "03_export",
        "status": (
            "completed" if analyze_manifest.get("status") == "completed" else "partial"
        ),
        **summary,
        "input_path": str(analysis_file),
        "output_path": str(output_dir),
        "artifacts": [
            str(requirement_path), str(pain_path), str(gap_path),
            str(requirement_distribution_path), str(gap_distribution_path),
            str(configuration_path), str(summary_path),
            str(verified_trace_path), str(verified_user_path),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    registry_record = None
    if manifest["status"] == "completed":
        registry_id = f"{run_id}__03_export__v{EXPORT_PRODUCT_VERSION.replace('.', '_')}"
        registry_record = write_immutable_record(
            "analysis-runs", registry_id,
            {
                "analysis_run_id": registry_id,
                "product_id": "trace_analysis_exports",
                "product_version": EXPORT_PRODUCT_VERSION,
                "input_batches": [batch_id],
                "output_path": str(output_dir),
                "status": "completed",
                "record_count": len(requirements) + len(pains) + len(gaps),
                "error_count": 0,
                "source_analysis_run_id": run_id,
            },
            artifacts=tuple(Path(path) for path in manifest["artifacts"]) + (manifest_path,),
            root=registry_root_for_output(run.parent.parent),
        )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "registry_record_path": str(registry_record.resolve()) if registry_record else None,
    }


__all__ = [
    "EXPORT_PRODUCT_VERSION", "EXPORT_SCHEMA_VERSION", "export_trace_analysis",
]
