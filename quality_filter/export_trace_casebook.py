#!/usr/bin/env python3
"""Export exact pilot traces and human-readable views for pairwise case review."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "quality_filter/pilot_50"
PAIR = ROOT / "quality_filter/pairwise_pilot_50"
OUT = ROOT / "quality_filter/trace_casebook"
CASE_PAIRS = ("pair_015", "pair_024", "pair_010")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def render_message(message: dict, number: int) -> str:
    role = message.get("role", "unknown")
    name = message.get("name")
    title = f"## Message {number}: {role}" + (f" ({name})" if name else "")
    return f"{title}\n\n```json\n{json.dumps(message, ensure_ascii=False, indent=2)}\n```\n"


def actual_winner(result: dict) -> int | str:
    winner = result["winner"]
    if winner == "A":
        return result["candidate_a_index"]
    if winner == "B":
        return result["candidate_b_index"]
    return winner


def main() -> int:
    traces = {row["pilot_index"]: row for row in read_jsonl(PILOT / "pilot_input.jsonl")}
    manifest = {row["pilot_index"]: row for row in read_jsonl(PILOT / "sample_manifest.jsonl")}
    pairs = {row["pair_id"]: row for row in read_jsonl(PAIR / "pairs.jsonl")}
    initial = {row["pair_id"]: row for row in read_jsonl(PAIR / "pairwise_results.jsonl")}
    reversed_results = {row["pair_id"]: row for row in read_jsonl(PAIR / "reversed_results.jsonl")}
    raw_dir = OUT / "raw"
    readable_dir = OUT / "readable"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for pair_id in CASE_PAIRS:
        pair = pairs[pair_id]
        selected.extend((pair["candidate_a_index"], pair["candidate_b_index"]))
    for index in dict.fromkeys(selected):
        trace = traces[index]
        info = manifest[index]
        stem = f"pilot_{index:02d}"
        (raw_dir / f"{stem}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        header = [
            f"# Pilot {index}: complete trace",
            "",
            f"- ID: `{trace['id']}`",
            f"- Source file: `{info['source_file']}`",
            f"- Source line: `{info['source_line']}`",
            f"- Message count: {len(trace['messages'])}",
            "- Integrity: every message below is rendered from the pilot JSON without rewriting",
            "",
        ]
        body = "\n".join(header) + "\n" + "\n".join(
            render_message(message, number) for number, message in enumerate(trace["messages"], 1)
        )
        (readable_dir / f"{stem}.md").write_text(body, encoding="utf-8")

    lines = [
        "# Pairwise trace casebook",
        "",
        "This casebook exposes the complete source traces behind three representative pairwise decisions.",
        "The raw JSON and readable Markdown are generated from `pilot_input.jsonl`; no trace content is rewritten.",
        "",
    ]
    labels = {
        "pair_015": "Stable strong winner",
        "pair_024": "Evidence-grounding difference",
        "pair_010": "Unstable boundary pair",
    }
    for pair_id in CASE_PAIRS:
        pair = pairs[pair_id]
        first = initial[pair_id]
        reverse = reversed_results[pair_id]
        a, b = pair["candidate_a_index"], pair["candidate_b_index"]
        lines.extend([
            f"## {pair_id}: {labels[pair_id]}",
            "",
            f"- Candidate A: pilot {a} — [readable](readable/pilot_{a:02d}.md) · [raw JSON](raw/pilot_{a:02d}.json)",
            f"- Candidate B: pilot {b} — [readable](readable/pilot_{b:02d}.md) · [raw JSON](raw/pilot_{b:02d}.json)",
            f"- Initial result: pilot {actual_winner(first)}, confidence {first['confidence']}",
            f"- Reversed result: pilot {actual_winner(reverse)}, confidence {reverse['confidence']}",
            "",
            "Initial judge reason:",
            "",
            f"> {first['reason']}",
            "",
            "Reversed-order judge reason:",
            "",
            f"> {reverse['reason']}",
            "",
        ])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "pairs": list(CASE_PAIRS), "traces": list(dict.fromkeys(selected))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
