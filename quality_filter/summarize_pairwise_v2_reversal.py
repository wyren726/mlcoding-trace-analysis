#!/usr/bin/env python3
"""Summarize winner, gate, and rubric stability for v2 reversed controls."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "quality_filter/pairwise_pilot_50/v2"
RUBRICS = tuple([f"Q{i}" for i in range(1, 9)] + [f"V{i}" for i in range(1, 7)])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def actual_choice(row: dict, value: str) -> int | str:
    if value == "A":
        return row["candidate_a_index"]
    if value == "B":
        return row["candidate_b_index"]
    return value


def mapped_gates(row: dict) -> dict[int, tuple[str, tuple[str, ...]]]:
    return {
        row["candidate_a_index"]: (row["hard_gate_status"]["A"], tuple(row["triggered_gates"]["A"])),
        row["candidate_b_index"]: (row["hard_gate_status"]["B"], tuple(row["triggered_gates"]["B"])),
    }


def mapped_gate_outcomes(row: dict) -> dict[int, str]:
    return {
        row["candidate_a_index"]: row["hard_gate_status"]["A"],
        row["candidate_b_index"]: row["hard_gate_status"]["B"],
    }


def main() -> int:
    initial = {row["pair_id"]: row for row in read_jsonl(V2_DIR / "pairwise_results.jsonl")}
    reversed_rows = read_jsonl(V2_DIR / "reversed_results.jsonl")
    details = []
    for reverse in reversed_rows:
        first = initial[reverse["pair_id"]]
        first_winner = actual_choice(first, first["winner"])
        reverse_winner = actual_choice(reverse, reverse["winner"])
        rubric_stability = {
            rubric: actual_choice(first, first["rubric_preferences"][rubric])
            == actual_choice(reverse, reverse["rubric_preferences"][rubric])
            for rubric in RUBRICS
        }
        details.append({
            "pair_id": reverse["pair_id"],
            "initial_order": [first["candidate_a_index"], first["candidate_b_index"]],
            "reversed_order": [reverse["candidate_a_index"], reverse["candidate_b_index"]],
            "initial_winner_pilot_index": first_winner,
            "reversed_winner_pilot_index": reverse_winner,
            "winner_stable": first_winner == reverse_winner,
            "initial_confidence": first["confidence"],
            "reversed_confidence": reverse["confidence"],
            "hard_gates_stable": mapped_gates(first) == mapped_gates(reverse),
            "hard_gate_outcomes_stable": mapped_gate_outcomes(first) == mapped_gate_outcomes(reverse),
            "initial_hard_gates_by_pilot": mapped_gates(first),
            "reversed_hard_gates_by_pilot": mapped_gates(reverse),
            "rubric_stability": rubric_stability,
            "initial_decisive_reason": first["reason"]["decisive_reason"],
            "reversed_decisive_reason": reverse["reason"]["decisive_reason"],
        })
    report = {
        "pair_count": len(details),
        "winner_stable_count": sum(row["winner_stable"] for row in details),
        "winner_stability_rate": round(sum(row["winner_stable"] for row in details) / len(details), 4),
        "hard_gate_stable_count": sum(row["hard_gates_stable"] for row in details),
        "hard_gate_stability_rate": round(sum(row["hard_gates_stable"] for row in details) / len(details), 4),
        "hard_gate_outcome_stable_count": sum(row["hard_gate_outcomes_stable"] for row in details),
        "hard_gate_outcome_stability_rate": round(sum(row["hard_gate_outcomes_stable"] for row in details) / len(details), 4),
        "confidence": {
            "initial_mean": round(statistics.mean(row["initial_confidence"] for row in details), 4),
            "reversed_mean": round(statistics.mean(row["reversed_confidence"] for row in details), 4),
        },
        "rubric_stability_rate": {
            rubric: round(sum(row["rubric_stability"][rubric] for row in details) / len(details), 4)
            for rubric in RUBRICS
        },
        "details": details,
    }
    output = V2_DIR / "reversal_stability_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
