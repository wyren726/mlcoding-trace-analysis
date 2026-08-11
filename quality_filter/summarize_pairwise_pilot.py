#!/usr/bin/env python3
"""Summarize blind pairwise results and compare them with prior absolute scores."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest, spearmanr


PAIR_DIR = Path("quality_filter/pairwise_pilot_50")
PILOT_DIR = Path("quality_filter/pilot_50")
DECISION_RANK = {"reject": 0, "review": 1, "usable": 2, "high_quality": 3}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def preference(value_a, value_b, winner: str) -> bool | None:
    if value_a == value_b:
        return None
    expected = "A" if value_a > value_b else "B"
    return winner == expected


def main() -> int:
    pairs = {row["pair_id"]: row for row in read_jsonl(PAIR_DIR / "pairs.jsonl")}
    results = read_jsonl(PAIR_DIR / "pairwise_results.jsonl")
    scores = {row["pilot_index"]: row for row in read_jsonl(PILOT_DIR / "quality_scores.normalized.jsonl")}
    inputs = {row["pilot_index"]: row for row in read_jsonl(PILOT_DIR / "pilot_input.jsonl")}
    if len(pairs) != 25 or len(results) != 25:
        raise ValueError("expected 25 complete pairs and results")

    enriched = []
    winners = []
    reserves = []
    quality_agreement = []
    value_agreement = []
    decision_agreement = []
    hard_gate_preference = []
    quality_gaps = []
    for result in results:
        pair = pairs[result["pair_id"]]
        a_index = pair["candidate_a_index"]
        b_index = pair["candidate_b_index"]
        if result["candidate_a_index"] != a_index or result["candidate_b_index"] != b_index:
            raise ValueError(f'candidate mismatch for {result["pair_id"]}')
        a_score = scores[a_index]
        b_score = scores[b_index]
        winner_index = a_index if result["winner"] == "A" else b_index if result["winner"] == "B" else None
        loser_index = b_index if result["winner"] == "A" else a_index if result["winner"] == "B" else None
        q_agree = preference(a_score["quality_score"], b_score["quality_score"], result["winner"])
        v_agree = preference(a_score["training_value_score"], b_score["training_value_score"], result["winner"])
        d_agree = preference(DECISION_RANK[a_score["decision"]], DECISION_RANK[b_score["decision"]], result["winner"])
        if q_agree is not None:
            quality_agreement.append(q_agree)
        if v_agree is not None:
            value_agreement.append(v_agree)
        if d_agree is not None:
            decision_agreement.append(d_agree)
        if a_score["hard_gate_status"] != b_score["hard_gate_status"]:
            gate_rank = {"reject": 0, "review": 1, "pass": 2}
            hard_gate_preference.append(preference(gate_rank[a_score["hard_gate_status"]], gate_rank[b_score["hard_gate_status"]], result["winner"]))
        quality_gaps.append(abs(a_score["quality_score"] - b_score["quality_score"]))
        enriched.append({
            **pair,
            **result,
            "winner_pilot_index": winner_index,
            "reserve_pilot_index": loser_index,
            "a_quality_score": a_score["quality_score"],
            "b_quality_score": b_score["quality_score"],
            "a_training_value_score": a_score["training_value_score"],
            "b_training_value_score": b_score["training_value_score"],
            "pairwise_agrees_with_higher_quality": q_agree,
            "pairwise_agrees_with_higher_training_value": v_agree,
            "pairwise_agrees_with_higher_absolute_decision": d_agree,
        })
        if winner_index is not None:
            winners.append(inputs[winner_index])
            reserves.append(inputs[loser_index])

    write_jsonl(PAIR_DIR / "pairwise_results.enriched.jsonl", enriched)
    write_jsonl(PAIR_DIR / "winners.jsonl", winners)
    write_jsonl(PAIR_DIR / "reserve.jsonl", reserves)
    dimension_counts = {
        dimension: dict(Counter(row["dimension_winners"][dimension] for row in results))
        for dimension in ("task_result", "instruction_compliance", "tool_evidence", "recovery_efficiency", "training_value")
    }
    winner_counts = Counter(row["winner"] for row in results)
    a_or_b = winner_counts["A"] + winner_counts["B"]
    report = {
        "pair_count": len(results),
        "trace_count": 50,
        "winner_counts": dict(winner_counts),
        "decisive_pair_rate": round(a_or_b / len(results), 4),
        "confidence": {
            "mean": round(statistics.mean(row["confidence"] for row in results), 4),
            "median": round(statistics.median(row["confidence"] for row in results), 4),
            "minimum": min(row["confidence"] for row in results),
            "maximum": max(row["confidence"] for row in results),
            "below_0_70": sum(row["confidence"] < 0.70 for row in results),
        },
        "position_balance": {
            "a_wins": winner_counts["A"],
            "b_wins": winner_counts["B"],
            "two_sided_binomial_p": round(float(binomtest(winner_counts["A"], a_or_b, 0.5).pvalue), 4),
            "note": "This detects gross imbalance only; reversed-order controls are still needed for reliability."
        },
        "dimension_winner_counts": dimension_counts,
        "agreement_with_prior_absolute_scoring": {
            "higher_quality_score": {"comparable_pairs": len(quality_agreement), "agreement_rate": round(sum(quality_agreement) / len(quality_agreement), 4)},
            "higher_training_value_score": {"comparable_pairs": len(value_agreement), "agreement_rate": round(sum(value_agreement) / len(value_agreement), 4)},
            "higher_decision_class": {"comparable_pairs": len(decision_agreement), "agreement_rate": round(sum(decision_agreement) / len(decision_agreement), 4)},
            "better_hard_gate_status": {"comparable_pairs": len(hard_gate_preference), "agreement_rate": round(sum(hard_gate_preference) / len(hard_gate_preference), 4) if hard_gate_preference else None},
        },
        "quality_score_gap": {
            "mean": round(statistics.mean(quality_gaps), 3),
            "median": round(statistics.median(quality_gaps), 3),
            "pairs_gap_below_5": sum(gap < 5 for gap in quality_gaps),
            "pairs_gap_below_10": sum(gap < 10 for gap in quality_gaps),
        },
        "reverse_review_required_by_protocol": [],
        "winner_count": len(winners),
        "reserve_count": len(reserves),
        "limitations": [
            "Each trace was compared once, so this is not a global ranking.",
            "No low-confidence or tie result triggered protocol reversal; order stability remains unmeasured.",
            "Prior absolute scores are not human ground truth and were used only after blind judging for comparison."
        ],
    }
    (PAIR_DIR / "pairwise_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
