#!/usr/bin/env python3
"""Create a blinded human-audit packet for unstable pairs plus stable controls."""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "quality_filter"
PAIR_DIR = QUALITY / "pairwise_pilot_50"
V2_DIR = PAIR_DIR / "v2"
PILOT_DIR = QUALITY / "pilot_50"
OUT = QUALITY / "human_audit_12"
SEED = 20260813
CONTROL_COUNT = 5


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def render_trace(trace: dict, label: str) -> str:
    lines = [
        f"# Candidate {label}",
        "",
        f"- Message count: {len(trace['messages'])}",
        "- Trace content below is rendered without rewriting",
        "",
    ]
    for number, message in enumerate(trace["messages"]):
        lines.extend([
            f"## Message {number} — {message.get('role', 'unknown')}",
            "",
            "```json",
            json.dumps(message, ensure_ascii=False, indent=2),
            "```",
            "",
        ])
    return "\n".join(lines)


def actual_winner(row: dict) -> int | str:
    if row["winner"] == "A":
        return row["candidate_a_index"]
    if row["winner"] == "B":
        return row["candidate_b_index"]
    return row["winner"]


def main() -> int:
    rng = random.Random(SEED)
    pairs = {row["pair_id"]: row for row in read_jsonl(PAIR_DIR / "pairs.jsonl")}
    traces = {row["pilot_index"]: row for row in read_jsonl(PILOT_DIR / "pilot_input.jsonl")}
    initial = {row["pair_id"]: row for row in read_jsonl(V2_DIR / "pairwise_results.jsonl")}
    reversed_rows = {row["pair_id"]: row for row in read_jsonl(V2_DIR / "reversed_results.jsonl")}
    decisions = json.loads((V2_DIR / "pilot_final_decision_report.json").read_text(encoding="utf-8"))
    unstable = list(decisions["human_review_pair_ids"])
    stable = [row["pair_id"] for row in decisions["pair_decisions"] if row["action"] == "accept_pairwise_result"]
    gate_controls = [pair_id for pair_id in stable if any(v != "pass" for v in initial[pair_id]["hard_gate_status"].values())]
    clean_controls = [pair_id for pair_id in stable if pair_id not in gate_controls]
    rng.shuffle(gate_controls)
    rng.shuffle(clean_controls)
    controls = gate_controls[:2] + clean_controls[: CONTROL_COUNT - min(2, len(gate_controls))]
    controls = controls[:CONTROL_COUNT]
    selected = [(pair_id, "unstable") for pair_id in unstable] + [(pair_id, "stable_control") for pair_id in controls]
    rng.shuffle(selected)

    raw_dir = OUT / "raw"
    readable_dir = OUT / "readable"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    answer_key = []
    for audit_number, (pair_id, audit_group) in enumerate(selected, 1):
        pair = pairs[pair_id]
        candidates = [pair["candidate_a_index"], pair["candidate_b_index"]]
        rng.shuffle(candidates)
        audit_id = f"audit_{audit_number:02d}"
        labels = {"A": candidates[0], "B": candidates[1]}
        pair_folder = readable_dir / audit_id
        pair_folder.mkdir(parents=True, exist_ok=True)
        for label, index in labels.items():
            trace = traces[index]
            (pair_folder / f"candidate_{label}.md").write_text(render_trace(trace, label), encoding="utf-8")
            (raw_dir / f"{audit_id}_candidate_{label}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        manifest.append({
            "audit_id": audit_id,
            "candidate_a_readable": f"readable/{audit_id}/candidate_A.md",
            "candidate_b_readable": f"readable/{audit_id}/candidate_B.md",
            "candidate_a_raw": f"raw/{audit_id}_candidate_A.json",
            "candidate_b_raw": f"raw/{audit_id}_candidate_B.json",
        })
        reverse = reversed_rows.get(pair_id)
        answer_key.append({
            "audit_id": audit_id,
            "pair_id": pair_id,
            "audit_group": audit_group,
            "display_mapping": labels,
            "initial_winner_pilot_index": actual_winner(initial[pair_id]),
            "reversed_winner_pilot_index": actual_winner(reverse) if reverse else None,
            "initial_hard_gate_status": initial[pair_id]["hard_gate_status"],
            "initial_triggered_gates": initial[pair_id]["triggered_gates"],
        })

    with (OUT / "audit_manifest.blinded.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (OUT / "answer_key.private.json").write_text(
        json.dumps({"seed": SEED, "items": answer_key}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_buffer = io.StringIO()
    fields = [
        "audit_id", "human_winner", "confidence", "candidate_a_gate", "candidate_b_gate",
        "evidence_a", "evidence_b", "decisive_reason", "reviewer",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for row in manifest:
        writer.writerow({"audit_id": row["audit_id"]})
    (OUT / "review_form.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")

    readme = [
        "# 人工盲审包——12 对 Trace",
        "",
        "填写审核表前，请勿打开 `answer_key.private.json`。",
        "12 对样本包含全部不稳定案例和 5 对稳定对照，顺序已经随机打乱。",
        "Judge Winner、分数、置信度、Reason、pair ID 和 pilot ID 均已隐藏。",
        "",
        "## 审核说明",
        "",
        "逐对阅读两条完整 Trace，并在 `review_form.csv` 中填写一行：",
        "",
        "- `human_winner`：A、B、tie 或 both_bad",
        "- `confidence`：0 到 1",
        "- `candidate_a_gate` / `candidate_b_gate`：pass、review 或 reject",
        "- `evidence_a` / `evidence_b`：零基 message 索引",
        "- `decisive_reason`：用一句话说明决定性差异",
        "- `reviewer`：审核人姓名或标识",
        "",
        "## 待审 Pair",
        "",
    ]
    for row in manifest:
        readme.extend([
            f"### {row['audit_id']}",
            "",
            f"- [Candidate A]({row['candidate_a_readable']}) · [raw JSON]({row['candidate_a_raw']})",
            f"- [Candidate B]({row['candidate_b_readable']}) · [raw JSON]({row['candidate_b_raw']})",
            "",
        ])
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "pair_count": len(selected),
        "unstable_count": len(unstable),
        "stable_control_count": len(controls),
        "seed": SEED,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
