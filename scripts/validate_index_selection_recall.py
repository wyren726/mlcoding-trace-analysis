#!/usr/bin/env python3
"""One-call-per-Trace recall pilot for the indexed Tool evidence selector."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trace_analysis.model_api import OpenAICompatibleClient, load_provider
from trace_analysis.pipeline.stage_02_analyze.compact import (
    INDEX_SELECTION_PROMPT_VERSION,
    INDEX_SELECTION_SYSTEM_PROMPT,
    deterministic_strong_evidence_ids,
    evidence_selection_prompt,
    indexed_candidate_payload,
    normalise_selected_event_ids,
    partition_indexed_candidate_payload,
)
from trace_analysis.pipeline.stage_02_analyze.runner import (
    candidate_agent_payload,
    complete_index_partition_with_fallback,
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


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = dt.datetime.now().astimezone().isoformat()
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provider-config", default="configs/providers.toml", type=Path)
    parser.add_argument("--provider", default="ailab_direct")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--request-max-chars", type=int, default=800_000)
    parser.add_argument("--partition-max-chars", type=int, default=50_000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--partition-workers", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    batch_id = args.batch.name
    run_id = args.run_dir.name
    stage = args.run_dir / "02_analyze"
    screens = _load_jsonl(stage / f"user_screen__{batch_id}__{run_id}.jsonl")
    bundles = {bundle.trace_id: bundle for bundle in iter_trace_bundles(
        args.batch / "unified_turns.jsonl"
    )}
    expected: dict[str, set[str]] = defaultdict(set)
    with args.oracle.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("event_type") in {"tool_call", "tool_result"}
                and row.get("quote_found_in_source") is True
                and row.get("selection_id")
            ):
                expected[str(row["trace_id"])].add(str(row["selection_id"]))

    jobs = []
    for trace_id, expected_ids in expected.items():
        candidate = candidate_agent_payload(bundles[trace_id], screens[trace_id])
        indexed = indexed_candidate_payload(candidate)
        partitions = partition_indexed_candidate_payload(
            indexed, max_prompt_chars=min(args.request_max_chars, args.partition_max_chars)
        )
        jobs.append({
            "trace_id": trace_id,
            "expected_ids": sorted(expected_ids),
            "candidate": candidate,
            "partitions": partitions,
            "prompt_chars": sum(len(evidence_selection_prompt(p)) for p in partitions),
            "max_partition_chars": max(len(evidence_selection_prompt(p)) for p in partitions),
        })
    jobs.sort(key=lambda job: (job["prompt_chars"], job["trace_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    results_path = args.output_dir / "selection_results.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    sample_path = args.output_dir / "sample.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    # Keep the output contract stable even when a stream has zero rows.
    results_path.touch()
    errors_path.touch()
    sample_path.touch()
    for job in jobs:
        _append(sample_path, {
            "trace_id": job["trace_id"],
            "expected_ids": job["expected_ids"],
            "prompt_chars": job["prompt_chars"],
            "partition_count": len(job["partitions"]),
            "max_partition_chars": job["max_partition_chars"],
        })
    manifest = {
        "validation_version": "index-selection-recall-v1",
        "prompt_version": INDEX_SELECTION_PROMPT_VERSION,
        "batch_id": batch_id,
        "source_analysis_run_id": run_id,
        "provider": args.provider,
        "model": args.model,
        "status": "running",
        "trace_count": len(jobs),
        "expected_evidence_count": sum(len(job["expected_ids"]) for job in jobs),
        "completed_trace_count": 0,
        "error_count": 0,
        "results_path": str(results_path.resolve()),
        "errors_path": str(errors_path.resolve()),
    }
    _write_manifest(manifest_path, manifest)

    client = OpenAICompatibleClient(
        load_provider(args.provider_config, args.provider),
        model=args.model,
        cache_dir=args.cache_dir or args.output_dir / "cache",
        timeout_seconds=args.timeout_seconds,
        max_retries=1,
        thinking_type="disabled",
        max_completion_tokens=args.max_completion_tokens,
    )
    completed = 0
    hits = 0
    try:
        for job in jobs:
            trace_id = job["trace_id"]
            if job["max_partition_chars"] > args.request_max_chars:
                raise ValueError(
                    f"{trace_id} index prompt has {job['prompt_chars']} chars; "
                    f"limit is {args.request_max_chars}"
                )
            try:
                deterministic = deterministic_strong_evidence_ids(job["candidate"])
                selected = list(deterministic)
                reasons = []
                usages = []
                thinking_verified = True
                def select_partition(partition: dict[str, Any]):
                    return complete_index_partition_with_fallback(
                        client, partition
                    )

                partition_results = []
                with ThreadPoolExecutor(max_workers=args.partition_workers) as pool:
                    futures = [pool.submit(select_partition, p) for p in job["partitions"]]
                    for future in as_completed(futures):
                        partition_results.extend(future.result())
                partition_results.sort(key=lambda item: (
                    item[0]["partition"].get("parent_index")
                    or item[0]["partition"]["index"],
                    item[0]["partition"].get("fallback_depth", 0),
                    item[0]["partition"]["index"],
                ))
                for partition, raw, metadata in partition_results:
                    limit = int(partition["partition"]["selection_limit"])
                    selected.extend(normalise_selected_event_ids(
                        raw, job["candidate"], limit=limit, excluded=set(selected)
                    ))
                    reasons.append({
                        "partition": partition["partition"],
                        "reason": str(raw.get("reason") or ""),
                    })
                    usages.append(metadata.get("usage"))
                    thinking_verified = thinking_verified and bool(
                        metadata.get("thinking_mode_verified")
                    )
                expected_ids = set(job["expected_ids"])
                selected_ids = set(selected)
                matched = sorted(expected_ids & selected_ids)
                missed = sorted(expected_ids - selected_ids)
                hits += len(matched)
                _append(results_path, {
                    "trace_id": trace_id,
                    "prompt_chars": job["prompt_chars"],
                    "partition_count": len(job["partitions"]),
                    "effective_partition_count": len(partition_results),
                    "fallback_partition_count": sum(
                        bool(partition["partition"].get("fallback_depth"))
                        for partition, _, _ in partition_results
                    ),
                    "max_partition_chars": job["max_partition_chars"],
                    "expected_ids": sorted(expected_ids),
                    "deterministic_selected_ids": deterministic,
                    "selected_ids": selected,
                    "matched_ids": matched,
                    "missed_ids": missed,
                    "recall": len(matched) / len(expected_ids),
                    "selection_reasons": reasons,
                    "usage_by_partition": usages,
                    "thinking_mode_verified": thinking_verified,
                })
                completed += 1
            except Exception as exc:
                _append(errors_path, {
                    "trace_id": trace_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                manifest["error_count"] += 1
                manifest["status"] = "partial"
                continue
            manifest["completed_trace_count"] = completed
            manifest["matched_evidence_count"] = hits
            _write_manifest(manifest_path, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_trace_count"] = completed
        manifest["matched_evidence_count"] = hits
        _write_manifest(manifest_path, manifest)
        raise

    attempted_expected = sum(
        len(job["expected_ids"]) for job in jobs[:completed]
    )
    manifest.update({
        "completed_trace_count": completed,
        "matched_evidence_count": hits,
        "attempted_expected_evidence_count": attempted_expected,
        "micro_recall": hits / attempted_expected if attempted_expected else None,
    })
    if manifest["status"] == "running":
        manifest["status"] = "completed"
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
