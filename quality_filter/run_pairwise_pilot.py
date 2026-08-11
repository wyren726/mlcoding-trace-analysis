#!/usr/bin/env python3
"""Run resumable blind pairwise judging for the 50-trace pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
WORKSPACE = Path(__file__).resolve().parents[1]
PAIR_DIR = WORKSPACE / "quality_filter/pairwise_pilot_50"
OUTPUT_SCHEMA = WORKSPACE / "quality_filter/docs/pairwise_output.schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--judge-version", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--pair-ids",
        nargs="+",
        metavar="PAIR_ID",
        help="evaluate only the specified pair IDs",
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--force-reverse",
        action="store_true",
        help="reverse every selected pair instead of only tie/low-confidence pairs",
    )
    parser.add_argument(
        "--reverse-closest",
        type=int,
        default=0,
        metavar="N",
        help="reverse the N pairs with the smallest prior absolute-quality gap",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def validate(path: Path, pair: dict, reverse: bool) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_a = pair["candidate_b_index"] if reverse else pair["candidate_a_index"]
    expected_b = pair["candidate_a_index"] if reverse else pair["candidate_b_index"]
    if value.get("pair_id") != pair["pair_id"]:
        raise ValueError("pair_id mismatch")
    if value.get("candidate_a_index") != expected_a or value.get("candidate_b_index") != expected_b:
        raise ValueError("candidate index mismatch")
    if not value.get("evidence_a") or not value.get("evidence_b"):
        raise ValueError("both candidates require evidence indices")
    if len(value["evidence_a"]) != len(set(value["evidence_a"])) or len(value["evidence_b"]) != len(set(value["evidence_b"])):
        raise ValueError("evidence indices must be unique")
    return value


def run_one(
    pair: dict,
    reverse: bool,
    output_dir: Path,
    log_dir: Path,
    prompt_doc: str,
    output_schema: Path,
) -> tuple[str, str]:
    output_path = output_dir / f'{pair["pair_id"]}.json'
    if output_path.exists():
        validate(output_path, pair, reverse)
        return pair["pair_id"], "skipped_valid"
    a = pair["candidate_b_index"] if reverse else pair["candidate_a_index"]
    b = pair["candidate_a_index"] if reverse else pair["candidate_b_index"]
    prompt = (
        "The user has confirmed this blind pairwise evaluation and authorized data transfer. "
        f"Read {prompt_doc}. "
        f'Evaluate pair_id {pair["pair_id"]}: candidate A has pilot_index {a}; candidate B has pilot_index {b}. '
        "Select records by their internal pilot_index fields. Do not read any existing scores, decisions, reports, or judge outputs. "
        "Do not ask for confirmation and do not modify source data."
    )
    command = [
        "codex", "exec", "-m", MODEL,
        "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "-s", "read-only", "--ephemeral", "--skip-git-repo-check",
        "-C", str(WORKSPACE), "--output-schema", str(output_schema),
        "-o", str(output_path), prompt,
    ]
    log_path = log_dir / f'{pair["pair_id"]}.log'
    for attempt in range(1, 3):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} ===\n")
            result = subprocess.run(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and output_path.exists():
            try:
                validate(output_path, pair, reverse)
                return pair["pair_id"], f"completed_attempt_{attempt}"
            except Exception as exc:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\nvalidation_error: {exc}\n")
        if output_path.exists():
            output_path.rename(output_dir / f'{pair["pair_id"]}.invalid_attempt_{attempt}.json')
    return pair["pair_id"], "failed"


def main() -> int:
    args = parse_args()
    pairs = read_jsonl(PAIR_DIR / "pairs.jsonl")
    all_pairs = list(pairs)
    if args.pair_ids:
        requested_ids = set(args.pair_ids)
        known_ids = {pair["pair_id"] for pair in pairs}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"unknown pair IDs: {unknown_ids}")
        pairs = [pair for pair in pairs if pair["pair_id"] in requested_ids]
    if args.judge_version == "v2":
        result_root = PAIR_DIR / "v2"
        prompt_doc = "quality_filter/docs/pairwise_judge_prompt_v2.md"
        output_schema = WORKSPACE / "quality_filter/docs/pairwise_output_v2.schema.json"
    else:
        result_root = PAIR_DIR
        prompt_doc = "quality_filter/docs/pairwise_judge_prompt.md"
        output_schema = OUTPUT_SCHEMA
    reverse_flags = sum((bool(args.reverse), bool(args.force_reverse), args.reverse_closest > 0))
    if reverse_flags > 1:
        raise ValueError("use only one of --reverse, --force-reverse, or --reverse-closest")
    reverse_mode = args.reverse or args.force_reverse or args.reverse_closest > 0
    if reverse_mode:
        initial_by_id = {row["pair_id"]: row for row in read_jsonl(result_root / "pairwise_results.jsonl")}
        if args.force_reverse:
            pass
        elif args.reverse_closest:
            enriched_by_id = {row["pair_id"]: row for row in read_jsonl(result_root / "pairwise_results.enriched.jsonl")}
            pairs = sorted(
                pairs,
                key=lambda pair: abs(
                    enriched_by_id[pair["pair_id"]]["a_quality_score"]
                    - enriched_by_id[pair["pair_id"]]["b_quality_score"]
                ),
            )[: args.reverse_closest]
        else:
            pairs = [pair for pair in pairs if initial_by_id[pair["pair_id"]]["winner"] == "tie" or initial_by_id[pair["pair_id"]]["confidence"] < 0.70]
        output_dir = result_root / "reversed"
        combined_path = result_root / "reversed_results.jsonl"
    else:
        output_dir = result_root / "initial"
        combined_path = result_root / "pairwise_results.jsonl"
    log_dir = result_root / ("reversed_logs" if reverse_mode else "logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    statuses = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, pair, reverse_mode, output_dir, log_dir, prompt_doc, output_schema): pair
            for pair in pairs
        }
        for future in as_completed(futures):
            pair_id, status = future.result()
            statuses[pair_id] = status
            print(json.dumps({"pair_id": pair_id, "status": status}), flush=True)
    results = []
    failures = []
    for pair in pairs:
        path = output_dir / f'{pair["pair_id"]}.json'
        if path.exists():
            results.append(validate(path, pair, reverse_mode))
        else:
            failures.append(pair["pair_id"])
    combined_results = results
    if reverse_mode:
        combined_results = []
        for pair in all_pairs:
            path = output_dir / f'{pair["pair_id"]}.json'
            if path.exists():
                combined_results.append(validate(path, pair, True))
    with combined_path.open("w", encoding="utf-8") as handle:
        for value in combined_results:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"reverse": reverse_mode, "reverse_closest": args.reverse_closest, "requested": len(pairs), "completed": len(results), "failures": failures, "statuses": statuses}
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / ("reversed_run_summary.json" if reverse_mode else "initial_run_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
