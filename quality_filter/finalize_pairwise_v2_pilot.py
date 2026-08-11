#!/usr/bin/env python3
"""Produce conservative trace-level pilot decisions from v2 pairwise controls."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIR_DIR = ROOT / "quality_filter/pairwise_pilot_50"
V2_DIR = PAIR_DIR / "v2"
PILOT_DIR = ROOT / "quality_filter/pilot_50"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def actual_choice(row: dict) -> int | str:
    if row["winner"] == "A":
        return row["candidate_a_index"]
    if row["winner"] == "B":
        return row["candidate_b_index"]
    return row["winner"]


def gate_outcomes(row: dict) -> dict[int, str]:
    return {
        row["candidate_a_index"]: row["hard_gate_status"]["A"],
        row["candidate_b_index"]: row["hard_gate_status"]["B"],
    }


def main() -> int:
    pairs = {row["pair_id"]: row for row in read_jsonl(PAIR_DIR / "pairs.jsonl")}
    initial = {row["pair_id"]: row for row in read_jsonl(V2_DIR / "pairwise_results.jsonl")}
    reversed_rows = {row["pair_id"]: row for row in read_jsonl(V2_DIR / "reversed_results.jsonl")}
    inputs = {row["pilot_index"]: row for row in read_jsonl(PILOT_DIR / "pilot_input.jsonl")}
    trace_decisions = []
    pair_decisions = []
    for pair_id, pair in pairs.items():
        first = initial[pair_id]
        reverse = reversed_rows.get(pair_id)
        winner = actual_choice(first)
        winner_stable = reverse is None or actual_choice(reverse) == winner
        gate_stable = reverse is None or gate_outcomes(reverse) == gate_outcomes(first)
        needs_review = not winner_stable or not gate_stable or first["winner"] in ("tie", "both_bad")
        candidates = (pair["candidate_a_index"], pair["candidate_b_index"])
        if needs_review:
            for index in candidates:
                trace_decisions.append({
                    "pilot_index": index,
                    "sample_id": inputs[index]["id"],
                    "pair_id": pair_id,
                    "decision": "review",
                    "reason": "winner_flip" if not winner_stable else "hard_gate_outcome_flip" if not gate_stable else first["winner"],
                })
            pair_action = "human_review"
        else:
            loser = candidates[1] if winner == candidates[0] else candidates[0]
            first_gates = gate_outcomes(first)
            loser_decision = "reject" if first_gates[loser] == "reject" else "reserve"
            trace_decisions.extend([
                {
                    "pilot_index": winner,
                    "sample_id": inputs[winner]["id"],
                    "pair_id": pair_id,
                    "decision": "keep",
                    "reason": "stable_pairwise_winner" if reverse else "screening_winner_no_review_trigger",
                },
                {
                    "pilot_index": loser,
                    "sample_id": inputs[loser]["id"],
                    "pair_id": pair_id,
                    "decision": loser_decision,
                    "reason": "stable_reject_gate" if loser_decision == "reject" else "pairwise_loser_reserve",
                },
            ])
            pair_action = "accept_pairwise_result"
        pair_decisions.append({
            "pair_id": pair_id,
            "initial_winner_pilot_index": winner,
            "reversed_winner_pilot_index": actual_choice(reverse) if reverse else None,
            "was_reversed": reverse is not None,
            "winner_stable": winner_stable,
            "hard_gate_outcomes_stable": gate_stable,
            "action": pair_action,
        })
    trace_decisions.sort(key=lambda row: row["pilot_index"])
    counts = Counter(row["decision"] for row in trace_decisions)
    report = {
        "trace_count": len(trace_decisions),
        "pair_count": len(pair_decisions),
        "decision_counts": dict(counts),
        "accepted_pair_count": sum(row["action"] == "accept_pairwise_result" for row in pair_decisions),
        "human_review_pair_count": sum(row["action"] == "human_review" for row in pair_decisions),
        "human_review_pair_ids": [row["pair_id"] for row in pair_decisions if row["action"] == "human_review"],
        "policy_note": "reserve is not reject; it is the lower-ranked trace in an accepted pair and may still be usable after dataset-level allocation.",
        "pair_decisions": pair_decisions,
    }
    (V2_DIR / "pilot_final_decision_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (V2_DIR / "pilot_trace_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in trace_decisions:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
