#!/usr/bin/env python3
"""Offline audit of indexed evidence preservation against existing Stage 03 evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterator

from trace_analysis.pipeline.stage_02_analyze.compact import (
    indexed_candidate_payload,
    payload_chars,
    selected_evidence_payload,
)
from trace_analysis.pipeline.stage_02_analyze.runner import (
    candidate_agent_payload,
    iter_trace_bundles,
)


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                rows[str(row["trace_id"])] = row
    return rows


def _serialized(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _evidence_references(value: Any, path: str = "") -> Iterator[dict[str, str]]:
    if isinstance(value, dict):
        if value.get("event_id") and isinstance(value.get("quote"), str):
            yield {
                "path": path,
                "event_id": str(value["event_id"]),
                "quote": value["quote"],
            }
        for key, child in value.items():
            yield from _evidence_references(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _evidence_references(child, f"{path}/{index}")


def _event_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(event["event_id"]): event
        for turn in payload.get("episode_turns") or []
        for event in turn.get("events") or []
        if event.get("event_id")
    }


def _selection_for_quote(event: dict[str, Any], quote: str,
                         chunk_chars: int = 12_000) -> str | None:
    event_id = str(event.get("event_id") or "")
    data_text = _serialized(event.get("data"))
    if len(data_text) <= chunk_chars:
        return event_id
    position = data_text.find(quote)
    if position < 0:
        return None
    return f"{event_id}#chunk_{position // chunk_chars:04d}"


def _semantic_anchors(quote: str) -> list[str]:
    """Return compact exact cues a selector could use; not a recall verdict."""
    anchors = []
    patterns = (
        r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s\\]+",
        r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)\b",
        r'"tool_name"\s*:\s*"([^"]+)"',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, quote):
            value = match.group(1) if match.lastindex else match.group(0)
            if value not in anchors:
                anchors.append(value)
    if not anchors:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./:+-]{9,}", quote):
            if not token.startswith("call_") and token not in anchors:
                anchors.append(token)
            if len(anchors) >= 5:
                break
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    stage = args.run_dir / "02_analyze"
    batch_id = args.batch.name
    run_id = args.run_dir.name
    screens = _load_jsonl(stage / f"user_screen__{batch_id}__{run_id}.jsonl")
    results = _load_jsonl(stage / f"trace_analysis__{batch_id}__{run_id}.jsonl")
    bundles = {bundle.trace_id: bundle for bundle in iter_trace_bundles(
        args.batch / "unified_turns.jsonl"
    )}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = args.output_dir / "evidence_coverage.jsonl"
    counts = {
        "analyzed_trace_count": 0,
        "trace_with_historical_assessment_count": 0,
        "evidence_reference_count": 0,
        "source_event_found_count": 0,
        "quote_found_in_source_count": 0,
        "visible_or_retrievable_count": 0,
        "tool_evidence_reference_count": 0,
        "tool_source_quote_count": 0,
        "tool_event_indexed_count": 0,
        "tool_quote_exposed_in_index_count": 0,
        "tool_anchor_exposed_in_index_count": 0,
        "tool_source_quote_anchor_exposed_count": 0,
        "tool_exact_quote_retrievable_count": 0,
    }
    trace_summaries = []
    with rows_path.open("w", encoding="utf-8") as target:
        for trace_id, screen in screens.items():
            if screen.get("route", {}).get("judgment") != "analyze":
                continue
            counts["analyzed_trace_count"] += 1
            record = results.get(trace_id) or {}
            assessments = [
                episode.get("agent_assessment")
                for episode in record.get("task_episodes") or []
                if isinstance(episode, dict) and isinstance(episode.get("agent_assessment"), dict)
            ]
            if not assessments:
                continue
            counts["trace_with_historical_assessment_count"] += 1
            bundle = bundles[trace_id]
            candidate = candidate_agent_payload(bundle, screen)
            indexed = indexed_candidate_payload(candidate)
            source_events = _event_map(candidate)
            indexed_events = _event_map(indexed)
            auto_visible = _serialized(selected_evidence_payload(candidate, []))
            references = {
                (ref["path"], ref["event_id"], ref["quote"]): ref
                for assessment in assessments
                for ref in _evidence_references(assessment, "/agent_assessment")
            }.values()
            trace_rows = []
            for ref in references:
                event_id = ref["event_id"]
                quote = ref["quote"]
                event = source_events.get(event_id)
                event_type = str(event.get("type") or "") if event else None
                source_found = event is not None
                quote_in_source = bool(event and quote in _serialized(event))
                is_tool = event_type in {"tool_call", "tool_result"}
                indexed_event = indexed_events.get(event_id)
                index_exposes_quote = bool(indexed_event and quote in _serialized(indexed_event))
                anchors = _semantic_anchors(quote) if is_tool else []
                index_text = _serialized(indexed_event) if indexed_event else ""
                exposed_anchors = [anchor for anchor in anchors if anchor in index_text]
                selection_id = _selection_for_quote(event, quote) if is_tool and event else None
                exact_retrievable = False
                if selection_id:
                    restored = selected_evidence_payload(candidate, [selection_id])
                    exact_retrievable = quote in _serialized(restored)
                elif not is_tool:
                    exact_retrievable = quote in auto_visible
                row = {
                    "trace_id": trace_id,
                    **ref,
                    "event_type": event_type,
                    "source_event_found": source_found,
                    "quote_found_in_source": quote_in_source,
                    "tool_event_indexed": bool(is_tool and indexed_event),
                    "tool_quote_exposed_in_index": bool(is_tool and index_exposes_quote),
                    "semantic_anchors": anchors,
                    "tool_anchors_exposed_in_index": exposed_anchors,
                    "selection_id": selection_id,
                    "exact_quote_visible_or_retrievable": exact_retrievable,
                }
                target.write(_serialized(row) + "\n")
                trace_rows.append(row)
                counts["evidence_reference_count"] += 1
                counts["source_event_found_count"] += int(source_found)
                counts["quote_found_in_source_count"] += int(quote_in_source)
                counts["visible_or_retrievable_count"] += int(exact_retrievable)
                if is_tool:
                    counts["tool_evidence_reference_count"] += 1
                    counts["tool_source_quote_count"] += int(quote_in_source)
                    counts["tool_event_indexed_count"] += int(indexed_event is not None)
                    counts["tool_quote_exposed_in_index_count"] += int(index_exposes_quote)
                    counts["tool_anchor_exposed_in_index_count"] += int(bool(exposed_anchors))
                    counts["tool_source_quote_anchor_exposed_count"] += int(
                        quote_in_source and bool(exposed_anchors)
                    )
                    counts["tool_exact_quote_retrievable_count"] += int(exact_retrievable)
            trace_summaries.append({
                "trace_id": trace_id,
                "candidate_chars": payload_chars(candidate),
                "evidence_reference_count": len(trace_rows),
                "tool_evidence_reference_count": sum(
                    row["event_type"] in {"tool_call", "tool_result"} for row in trace_rows
                ),
                "all_source_events_found": all(row["source_event_found"] for row in trace_rows),
                "all_exact_quotes_visible_or_retrievable": all(
                    row["exact_quote_visible_or_retrievable"] for row in trace_rows
                ),
            })

    def ratio(numerator: str, denominator: str) -> float | None:
        total = counts[denominator]
        return counts[numerator] / total if total else None

    report = {
        "audit_version": "indexed-evidence-offline-coverage-v2",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "batch_id": batch_id,
        "source_analysis_run_id": run_id,
        "scope": (
            "existing agent_assessment evidence references; validates identity, "
            "index preservation and exact retrieval, not model semantic selection"
        ),
        "counts": counts,
        "rates": {
            "source_event_found": ratio("source_event_found_count", "evidence_reference_count"),
            "quote_found_in_source": ratio("quote_found_in_source_count", "evidence_reference_count"),
            "visible_or_retrievable": ratio("visible_or_retrievable_count", "evidence_reference_count"),
            "tool_event_indexed": ratio("tool_event_indexed_count", "tool_evidence_reference_count"),
            "tool_quote_exposed_in_index": ratio(
                "tool_quote_exposed_in_index_count", "tool_evidence_reference_count"
            ),
            "tool_anchor_exposed_in_index": ratio(
                "tool_anchor_exposed_in_index_count", "tool_evidence_reference_count"
            ),
            "valid_tool_quote_anchor_exposed_in_index": ratio(
                "tool_source_quote_anchor_exposed_count", "tool_source_quote_count"
            ),
            "tool_exact_quote_retrievable": ratio(
                "tool_exact_quote_retrievable_count", "tool_evidence_reference_count"
            ),
            "valid_tool_quote_exact_retrievable": ratio(
                "tool_exact_quote_retrievable_count", "tool_source_quote_count"
            ),
        },
        "trace_summaries": trace_summaries,
        "rows_path": str(rows_path.resolve()),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
