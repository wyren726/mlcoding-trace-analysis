#!/usr/bin/env python3
"""Paired full-evidence vs indexed-evidence validation for Stage 02 candidates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from trace_analysis.model_api import OpenAICompatibleClient, load_provider
from trace_analysis.pipeline.stage_02_analyze.compact import payload_chars
from trace_analysis.pipeline.stage_02_analyze.runner import (
    TraceBundle,
    UserScreenResult,
    candidate_agent_payload,
    iter_trace_bundles,
    verify_screened_trace,
)


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                rows[str(row["trace_id"])] = row
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["candidate_chars"], row["trace_id"]))
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indexes]


def _episode_view(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for episode in record.get("task_episodes") or []:
        route = episode.get("route") if isinstance(episode, dict) else None
        if not isinstance(route, dict) or route.get("judgment") != "analyze":
            continue
        assessment = episode.get("agent_assessment") or {}
        gaps = assessment.get("capability_gaps") or []
        evidence_ids = sorted({
            str(evidence.get("event_id"))
            for gap in gaps if isinstance(gap, dict)
            for evidence in gap.get("evidence") or [] if isinstance(evidence, dict)
            if evidence.get("event_id")
        })
        result[str(episode.get("query_turn_id"))] = {
            "pain_confirmed": episode.get("pain_confirmed"),
            "agent_related": assessment.get("agent_related"),
            "attribution": assessment.get("attribution"),
            "capabilities": sorted({
                str(gap.get("capability")) for gap in gaps
                if isinstance(gap, dict) and gap.get("capability")
            }),
            "gap_evidence_event_ids": evidence_ids,
        }
    return result


def _comparison(trace_id: str, direct: dict[str, Any], indexed: dict[str, Any],
                event_types: dict[str, str]) -> dict[str, Any]:
    direct_episodes = _episode_view(direct)
    indexed_episodes = _episode_view(indexed)
    query_ids = sorted(set(direct_episodes) | set(indexed_episodes))
    selected = indexed.get("analysis_provenance", {}).get("selected_tool_event_ids") or []
    selected_parents = {str(value).split("#chunk_", 1)[0] for value in selected}
    direct_tool_evidence = {
        event_id for episode in direct_episodes.values()
        for event_id in episode["gap_evidence_event_ids"]
        if event_types.get(event_id) in {"tool_call", "tool_result"}
    }
    recalled = direct_tool_evidence & selected_parents
    episode_rows = []
    for query_id in query_ids:
        left = direct_episodes.get(query_id)
        right = indexed_episodes.get(query_id)
        episode_rows.append({
            "query_turn_id": query_id,
            "direct": left,
            "indexed": right,
            "pain_agreement": bool(left and right and left["pain_confirmed"] == right["pain_confirmed"]),
            "attribution_agreement": bool(left and right and left["attribution"] == right["attribution"]),
        })
    return {
        "trace_id": trace_id,
        "episode_comparisons": episode_rows,
        "pain_agreement": all(row["pain_agreement"] for row in episode_rows),
        "attribution_agreement": all(row["attribution_agreement"] for row in episode_rows),
        "direct_tool_evidence_event_ids": sorted(direct_tool_evidence),
        "indexed_selected_tool_event_ids": selected,
        "direct_tool_evidence_recalled": sorted(recalled),
        "tool_evidence_recall": (
            len(recalled) / len(direct_tool_evidence) if direct_tool_evidence else None
        ),
        "indexed_evidence_review": indexed.get("analysis_provenance", {}).get("evidence_review"),
        "indexed_retrieval_rounds": indexed.get("analysis_provenance", {}).get("evidence_retrieval_rounds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provider-config", default="configs/providers.toml", type=Path)
    parser.add_argument("--provider", default="ailab_direct")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--max-direct-chars", type=int, default=500_000)
    parser.add_argument("--request-max-chars", type=int, default=800_000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    stage = args.run_dir / "02_analyze"
    run_id = args.run_dir.name
    batch_id = args.batch.name
    screen_path = stage / f"user_screen__{batch_id}__{run_id}.jsonl"
    final_path = stage / f"trace_analysis__{batch_id}__{run_id}.jsonl"
    screens = _load_jsonl(screen_path)
    historical = _load_jsonl(final_path)
    bundles = {bundle.trace_id: bundle for bundle in iter_trace_bundles(
        args.batch / "unified_turns.jsonl"
    )}

    candidates = []
    for trace_id, screen in screens.items():
        if screen.get("route", {}).get("judgment") != "analyze":
            continue
        bundle = bundles.get(trace_id)
        if bundle is None:
            continue
        payload = candidate_agent_payload(bundle, screen)
        chars = payload_chars(payload)
        tool_count = sum(
            event.get("type") in {"tool_call", "tool_result"}
            for turn in payload.get("episode_turns") or []
            for event in turn.get("events") or []
        )
        if not tool_count or chars > args.max_direct_chars:
            continue
        known_pain = any(
            episode.get("pain_confirmed") is True
            for episode in historical.get(trace_id, {}).get("task_episodes") or []
        )
        candidates.append({
            "trace_id": trace_id,
            "candidate_chars": chars,
            "tool_event_count": tool_count,
            "known_pain": known_pain,
        })
    pain = [row for row in candidates if row["known_pain"]]
    no_pain = [row for row in candidates if not row["known_pain"]]
    pain_count = min(len(pain), (args.size + 1) // 2)
    no_pain_count = min(len(no_pain), args.size - pain_count)
    sample = _spread(pain, pain_count) + _spread(no_pain, no_pain_count)
    if len(sample) < args.size:
        chosen = {row["trace_id"] for row in sample}
        sample += _spread(
            [row for row in candidates if row["trace_id"] not in chosen],
            args.size - len(sample),
        )
    # Run cheaper candidates first so a slow long-context request cannot
    # prevent a small pilot from producing useful partial comparisons.
    sample.sort(key=lambda row: (row["candidate_chars"], row["trace_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    sample_path = args.output_dir / "sample.jsonl"
    comparisons_path = args.output_dir / "comparisons.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    for row in sample:
        _append(sample_path, row)
    manifest = {
        "validation_version": "indexed-evidence-recall-v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "batch_id": batch_id,
        "source_analysis_run_id": run_id,
        "provider": args.provider,
        "model": args.model,
        "sample_size": len(sample),
        "status": "preview" if not args.execute else "running",
        "sample_path": str(sample_path.resolve()),
        "comparisons_path": str(comparisons_path.resolve()),
        "errors_path": str(errors_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    client = OpenAICompatibleClient(
        load_provider(args.provider_config, args.provider), model=args.model,
        cache_dir=args.output_dir / "cache", timeout_seconds=args.timeout_seconds,
        max_retries=0, thinking_type="disabled", max_completion_tokens=8192,
    )
    completed = 0
    for sample_row in sample:
        trace_id = sample_row["trace_id"]
        bundle: TraceBundle = bundles[trace_id]
        screen = UserScreenResult(screens[trace_id], "existing_user_screen", 0, {})
        try:
            direct = verify_screened_trace(
                bundle, client, screen, args.max_direct_chars, args.request_max_chars
            ).record
            indexed = verify_screened_trace(
                bundle, client, screen, 0, args.request_max_chars
            ).record
            event_types = {
                str(event.get("event_id")): str(event.get("type"))
                for turn in bundle.turns for event in turn.get("events") or []
                if event.get("event_id")
            }
            row = {
                **sample_row,
                **_comparison(trace_id, direct, indexed, event_types),
                "direct_record": direct,
                "indexed_record": indexed,
            }
            _append(comparisons_path, row)
            completed += 1
        except Exception as exc:  # keep a partial, resumable audit packet
            _append(errors_path, {
                "trace_id": trace_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        manifest.update({
            "status": "running",
            "completed_count": completed,
            "attempted_count": completed + sum(1 for _ in errors_path.open()) if errors_path.exists() else completed,
            "updated_at": dt.datetime.now().astimezone().isoformat(),
        })
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    errors = sum(1 for _ in errors_path.open()) if errors_path.exists() else 0
    manifest.update({
        "status": "completed" if errors == 0 else "partial",
        "completed_count": completed,
        "error_count": errors,
        "completed_at": dt.datetime.now().astimezone().isoformat(),
    })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
