#!/usr/bin/env python3
"""Run the fixed Codex judge over the reproducible 50-trace pilot sample."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
EXPECTED_QUALITY_IDS = {f"Q{i}_" for i in range(1, 9)}
EXPECTED_VALUE_IDS = {f"V{i}_" for i in range(1, 7)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("quality_filter/pilot_50"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=50)
    return parser.parse_args()


def validate_score(path: Path, expected_index: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("pilot_index") != expected_index:
        raise ValueError(f"pilot index mismatch: {value.get('pilot_index')} != {expected_index}")
    if not value.get("sample_id"):
        raise ValueError("missing sample_id")
    quality = value.get("quality_rubric_results") or []
    training_value = value.get("training_value_rubric_results") or []
    if len(quality) != 8:
        raise ValueError(f"expected 8 quality rubrics, found {len(quality)}")
    if len(training_value) != 6:
        raise ValueError(f"expected 6 training-value rubrics, found {len(training_value)}")
    if not all(any(item.get("rubric_id", "").startswith(prefix) for item in quality) for prefix in EXPECTED_QUALITY_IDS):
        raise ValueError("quality rubric IDs are incomplete")
    if not all(any(item.get("rubric_id", "").startswith(prefix) for item in training_value) for prefix in EXPECTED_VALUE_IDS):
        raise ValueError("training-value rubric IDs are incomplete")
    return value


def run_one(index: int, workspace: Path, pilot_dir: Path, score_dir: Path, log_dir: Path) -> tuple[int, str]:
    output_path = score_dir / f"score_{index:03d}.json"
    if output_path.exists():
        validate_score(output_path, index)
        return index, "skipped_valid"
    prompt = (
        "The user has already confirmed this exact read-only evaluation and authorized the data transfer. "
        f"Read quality_filter/docs/judge_prompt.md and evaluate only pilot_index {index} now. "
        "Do not ask for confirmation. Do not modify any source or data file."
    )
    command = [
        "codex", "exec",
        "-m", MODEL,
        "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "-s", "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "-C", str(workspace),
        "--output-schema", str(workspace / "quality_filter/docs/judge_output.schema.json"),
        "-o", str(output_path),
        prompt,
    ]
    log_path = log_dir / f"judge_{index:03d}.log"
    for attempt in range(1, 3):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} ===\n")
            result = subprocess.run(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and output_path.exists():
            try:
                validate_score(output_path, index)
                return index, f"completed_attempt_{attempt}"
            except Exception as exc:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\nvalidation_error: {exc}\n")
        if output_path.exists():
            output_path.rename(score_dir / f"invalid_{index:03d}_attempt_{attempt}.json")
    return index, "failed"


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[1]
    pilot_dir = (workspace / args.pilot_dir).resolve() if not args.pilot_dir.is_absolute() else args.pilot_dir
    score_dir = pilot_dir / "scores"
    log_dir = pilot_dir / "logs"
    score_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    smoke = pilot_dir / "smoke_001_retry.json"
    first_score = score_dir / "score_001.json"
    if smoke.exists() and not first_score.exists():
        validate_score(smoke, 1)
        shutil.copy2(smoke, first_score)

    indices = list(range(args.start, args.end + 1))
    statuses: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, index, workspace, pilot_dir, score_dir, log_dir): index
            for index in indices
        }
        for future in as_completed(futures):
            index, status = future.result()
            statuses[index] = status
            print(json.dumps({"pilot_index": index, "status": status}), flush=True)

    records = []
    failures = []
    for index in indices:
        path = score_dir / f"score_{index:03d}.json"
        if path.exists():
            records.append(validate_score(path, index))
        else:
            failures.append(index)
    records.sort(key=lambda item: item["pilot_index"])
    with (pilot_dir / "quality_scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "judge": {"model": MODEL, "reasoning_effort": REASONING_EFFORT, "prompt_version": "v1", "rubric_schema_version": "1.0.0"},
        "requested_count": len(indices),
        "completed_count": len(records),
        "failed_indices": failures,
        "statuses": {str(index): statuses.get(index, "missing") for index in indices},
    }
    (pilot_dir / "judge_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
