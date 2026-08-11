#!/usr/bin/env python3
"""Measure A/B order stability for reversed pairwise controls."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


PAIR_DIR = Path("quality_filter/pairwise_pilot_50")
DIMS = ("task_result", "instruction_compliance", "tool_evidence", "recovery_efficiency", "training_value")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def actual_choice(row: dict, field: str = "winner") -> int | str | None:
    choice = row[field]
    if choice == "A":
        return row["candidate_a_index"]
    if choice == "B":
        return row["candidate_b_index"]
    return choice


def main() -> int:
    initial = {row["pair_id"]: row for row in read_jsonl(PAIR_DIR / "pairwise_results.jsonl")}
    reversed_rows = read_jsonl(PAIR_DIR / "reversed_results.jsonl")
    details = []
    for reverse in reversed_rows:
        first = initial[reverse["pair_id"]]
        dimension_stability = {}
        for dim in DIMS:
            first_choice = first["dimension_winners"][dim]
            reverse_choice = reverse["dimension_winners"][dim]
            first_actual = first["candidate_a_index"] if first_choice == "A" else first["candidate_b_index"] if first_choice == "B" else first_choice
            reverse_actual = reverse["candidate_a_index"] if reverse_choice == "A" else reverse["candidate_b_index"] if reverse_choice == "B" else reverse_choice
            dimension_stability[dim] = first_actual == reverse_actual
        details.append({
            "pair_id": reverse["pair_id"],
            "initial_order": [first["candidate_a_index"], first["candidate_b_index"]],
            "reversed_order": [reverse["candidate_a_index"], reverse["candidate_b_index"]],
            "initial_winner_pilot_index": actual_choice(first),
            "reversed_winner_pilot_index": actual_choice(reverse),
            "winner_stable": actual_choice(first) == actual_choice(reverse),
            "initial_confidence": first["confidence"],
            "reversed_confidence": reverse["confidence"],
            "dimension_stability": dimension_stability,
            "initial_reason": first["reason"],
            "reversed_reason": reverse["reason"],
        })
    stable = sum(row["winner_stable"] for row in details)
    report = {
        "pair_count": len(details),
        "winner_stable_count": stable,
        "winner_flip_count": len(details) - stable,
        "winner_stability_rate": round(stable / len(details), 4),
        "confidence": {
            "initial_mean": round(statistics.mean(row["initial_confidence"] for row in details), 4),
            "reversed_mean": round(statistics.mean(row["reversed_confidence"] for row in details), 4),
        },
        "dimension_stability_rate": {
            dim: round(sum(row["dimension_stability"][dim] for row in details) / len(details), 4)
            for dim in DIMS
        },
        "details": details,
    }
    output = PAIR_DIR / "reversal_stability_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
