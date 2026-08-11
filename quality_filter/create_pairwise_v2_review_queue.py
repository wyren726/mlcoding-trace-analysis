#!/usr/bin/env python3
"""Create a deduplicated v2 reversal/review queue from pilot diagnostics."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "quality_filter/pairwise_pilot_50/v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    report = json.loads((V2_DIR / "v1_v2_comparison_report.json").read_text(encoding="utf-8"))
    reversal_path = V2_DIR / "reversal_stability_report.json"
    reversed_details = {}
    if reversal_path.exists():
        reversed_details = {
            row["pair_id"]: row
            for row in json.loads(reversal_path.read_text(encoding="utf-8"))["details"]
        }
    queue = []
    resolved = []
    for row in report["details"]:
        reasons = []
        if row["v2_confidence"] < 0.85:
            reasons.append("confidence_below_0_85")
        if row["v2_winner"] in ("tie", "both_bad"):
            reasons.append(f"winner_{row['v2_winner']}")
        if any(status != "pass" for status in row["hard_gate_status"].values()):
            reasons.append("hard_gate_triggered")
        if not row["winner_agrees"]:
            reasons.append("v1_v2_winner_disagreement")
        if not reasons:
            continue
        previous = reversed_details.get(row["pair_id"])
        entry = {
            "pair_id": row["pair_id"],
            "reasons": reasons,
            "v2_winner_pilot_index": row["v2_winner_pilot_index"],
            "v2_confidence": row["v2_confidence"],
            "hard_gate_status": row["hard_gate_status"],
            "triggered_gates": row["triggered_gates"],
        }
        if previous:
            entry["already_reversed"] = True
            entry["winner_stable"] = previous["winner_stable"]
            entry["hard_gates_stable"] = previous["hard_gates_stable"]
            entry["recommended_action"] = "accept_stable" if previous["winner_stable"] and previous["hard_gates_stable"] else "human_review"
            resolved.append(entry)
        else:
            entry["already_reversed"] = False
            entry["recommended_action"] = "reverse_once"
            queue.append(entry)
    output = {
        "policy": {
            "confidence_threshold": 0.85,
            "triggers": ["low confidence", "tie/both_bad", "hard gate", "v1/v2 disagreement"],
        },
        "new_reverse_count": len(queue),
        "new_reverse_pair_ids": [row["pair_id"] for row in queue],
        "already_reversed_count": len(resolved),
        "already_reversed_diagnostics": resolved,
        "queue": queue,
    }
    (V2_DIR / "reverse_review_queue.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
