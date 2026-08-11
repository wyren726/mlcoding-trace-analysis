#!/usr/bin/env python3
"""Summarize validated pilot judge outputs."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PILOT_DIR = Path("quality_filter/pilot_50")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def grouped_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: {
            "count": len(values),
            "quality_score_mean": round(statistics.mean(v["quality_score"] for v in values), 2),
            "training_value_score_mean": round(statistics.mean(v["training_value_score"] for v in values), 2),
            "decisions": dict(Counter(v["decision"] for v in values)),
        }
        for name, values in sorted(groups.items())
    }


def main() -> int:
    manifest = read_jsonl(PILOT_DIR / "sample_manifest.jsonl")
    scores = read_jsonl(PILOT_DIR / "quality_scores.jsonl")
    pilot_inputs = read_jsonl(PILOT_DIR / "pilot_input.jsonl")
    manifest_by_index = {row["pilot_index"]: row for row in manifest}
    input_by_index = {row["pilot_index"]: row for row in pilot_inputs}
    if len(manifest_by_index) != 50 or len(scores) != 50:
        raise ValueError(f"expected 50 manifest and score records, got {len(manifest_by_index)} and {len(scores)}")
    joined = []
    for score in scores:
        meta = manifest_by_index[score["pilot_index"]]
        source_sample = input_by_index[score["pilot_index"]]
        valid_identifiers = {source_sample.get("id"), source_sample.get("sample_id")}
        if score["sample_id"] not in valid_identifiers:
            raise ValueError(f"sample mismatch at pilot_index {score['pilot_index']}")
        normalized_score = {
            **score,
            "sample_id": source_sample.get("id"),
            "source_sample_id": source_sample.get("sample_id"),
        }
        joined.append({**meta, **normalized_score})

    quality_values: dict[str, list[int]] = defaultdict(list)
    training_values: dict[str, list[int]] = defaultdict(list)
    gate_counts: Counter[str] = Counter()
    for row in joined:
        for item in row["quality_rubric_results"]:
            if item["applicability"] == "applicable" and item["raw_score"] is not None:
                quality_values[item["rubric_id"]].append(item["raw_score"])
        for item in row["training_value_rubric_results"]:
            if item["applicability"] == "applicable" and item["raw_score"] is not None:
                training_values[item["rubric_id"]].append(item["raw_score"])
        for gate in row["hard_gate_results"]:
            if gate["triggered"]:
                gate_counts[gate["gate_id"]] += 1

    report = {
        "pilot": {
            "sample_count": len(joined),
            "seed": 20260811,
            "judge_model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
            "judge_count": 1,
            "purpose": "rubric and prompt pilot; not a production filtering decision",
        },
        "validation": {
            "manifest_score_match": True,
            "failed_judge_count": 0,
            "all_quality_rubrics_present": all(len(row["quality_rubric_results"]) == 8 for row in joined),
            "all_training_value_rubrics_present": all(len(row["training_value_rubric_results"]) == 6 for row in joined),
        },
        "overall": {
            "decision_counts": dict(Counter(row["decision"] for row in joined)),
            "hard_gate_status_counts": dict(Counter(row["hard_gate_status"] for row in joined)),
            "quality_score_mean": round(statistics.mean(row["quality_score"] for row in joined), 2),
            "quality_score_median": round(statistics.median(row["quality_score"] for row in joined), 2),
            "quality_score_min": min(row["quality_score"] for row in joined),
            "quality_score_max": max(row["quality_score"] for row in joined),
            "training_value_score_mean": round(statistics.mean(row["training_value_score"] for row in joined), 2),
            "training_value_score_median": round(statistics.median(row["training_value_score"] for row in joined), 2),
            "judge_confidence_mean": round(statistics.mean(row["judge_confidence"] for row in joined), 3),
        },
        "triggered_hard_gates": dict(gate_counts.most_common()),
        "quality_rubric_means_0_to_4": {
            rubric_id: {"mean": round(statistics.mean(values), 3), "applicable_count": len(values)}
            for rubric_id, values in sorted(quality_values.items())
        },
        "training_value_rubric_means_0_to_4": {
            rubric_id: {"mean": round(statistics.mean(values), 3), "applicable_count": len(values)}
            for rubric_id, values in sorted(training_values.items())
        },
        "by_source_file": grouped_summary(joined, "source_file"),
        "by_length_bucket": grouped_summary(joined, "length_bucket"),
        "by_has_tool_error": grouped_summary(joined, "has_tool_error"),
        "by_mixed_model": grouped_summary(joined, "mixed_model"),
    }
    (PILOT_DIR / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (PILOT_DIR / "quality_scores.normalized.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(joined, key=lambda item: item["pilot_index"]):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    review_rows = [row for row in joined if row["decision"] == "review"]
    with (PILOT_DIR / "review_queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
