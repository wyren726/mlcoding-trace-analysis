#!/usr/bin/env python3
"""Validate the five-pair v2 pilot and compare it with v1 decisions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIR_DIR = ROOT / "quality_filter/pairwise_pilot_50"
PILOT_DIR = ROOT / "quality_filter/pilot_50"
RUBRICS = tuple([f"Q{i}" for i in range(1, 9)] + [f"V{i}" for i in range(1, 7)])
REASON_FIELDS = ("hard_gate_comparison", "quality_comparison", "training_value_comparison", "decisive_reason")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def actual_winner(row: dict) -> int | str:
    if row["winner"] == "A":
        return row["candidate_a_index"]
    if row["winner"] == "B":
        return row["candidate_b_index"]
    return row["winner"]


def has_true_reasoning(trace: dict) -> bool:
    return any(
        isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip()
        for message in trace["messages"]
    )


def main() -> int:
    v1 = {row["pair_id"]: row for row in read_jsonl(PAIR_DIR / "pairwise_results.jsonl")}
    v2 = read_jsonl(PAIR_DIR / "v2/pairwise_results.jsonl")
    inputs = {row["pilot_index"]: row for row in read_jsonl(PILOT_DIR / "pilot_input.jsonl")}
    details = []
    errors = []
    warnings = []
    for row in v2:
        pair_id = row["pair_id"]
        first = v1[pair_id]
        for side, index in (("A", row["candidate_a_index"]), ("B", row["candidate_b_index"])):
            size = len(inputs[index]["messages"])
            evidence = row[f"evidence_{side.lower()}"]
            if not evidence or any(value < 0 or value >= size for value in evidence):
                errors.append(f"{pair_id}: invalid evidence for {side}")
        if set(row["rubric_preferences"]) != set(RUBRICS):
            errors.append(f"{pair_id}: rubric keys incomplete")
        if set(row["reason"]) != set(REASON_FIELDS):
            errors.append(f"{pair_id}: reason fields incomplete")
        both_reasoning = all(has_true_reasoning(inputs[row[f"candidate_{side.lower()}_index"]]) for side in ("A", "B"))
        if both_reasoning and row["rubric_preferences"]["Q7"] == "NA":
            warnings.append(f"{pair_id}: Q7 marked NA although both traces contain reasoning_content")
        details.append({
            "pair_id": pair_id,
            "v1_winner": first["winner"],
            "v1_winner_pilot_index": actual_winner(first),
            "v1_confidence": first["confidence"],
            "v2_winner": row["winner"],
            "v2_winner_pilot_index": actual_winner(row),
            "v2_confidence": row["confidence"],
            "winner_agrees": actual_winner(first) == actual_winner(row),
            "hard_gate_status": row["hard_gate_status"],
            "triggered_gates": row["triggered_gates"],
            "rubric_preferences": row["rubric_preferences"],
            "reason": row["reason"],
            "evidence_a": row["evidence_a"],
            "evidence_b": row["evidence_b"],
        })
    agreement = sum(row["winner_agrees"] for row in details)
    report = {
        "pair_count": len(details),
        "schema_and_index_errors": errors,
        "semantic_validation_warnings": warnings,
        "v1_v2_winner_agreement_count": agreement,
        "v1_v2_winner_agreement_rate": round(agreement / len(details), 4),
        "v2_winner_counts": dict(Counter(row["v2_winner"] for row in details)),
        "details": details,
    }
    output = PAIR_DIR / "v2/v1_v2_comparison_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
