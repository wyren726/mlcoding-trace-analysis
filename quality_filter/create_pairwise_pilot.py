#!/usr/bin/env python3
"""Create 25 reproducible, length-stratified pairs from the validated pilot."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 20260812
SOURCE = Path("quality_filter/pilot_50/sample_manifest.jsonl")
OUTPUT_DIR = Path("quality_filter/pairwise_pilot_50")


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    if len(rows) != 50:
        raise ValueError(f"expected 50 manifest rows, found {len(rows)}")
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["length_bucket"]].append(row)
    pairs = []
    leftovers = []
    for bucket in ("short", "medium", "long", "extra_long"):
        values = groups[bucket]
        rng.shuffle(values)
        while len(values) >= 2:
            a = values.pop()
            b = values.pop()
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append((a, b, bucket))
        leftovers.extend(values)
    rng.shuffle(leftovers)
    while len(leftovers) >= 2:
        a = leftovers.pop()
        b = leftovers.pop()
        if rng.random() < 0.5:
            a, b = b, a
        pairs.append((a, b, f'{a["length_bucket"]}+{b["length_bucket"]}'))
    if leftovers or len(pairs) != 25:
        raise ValueError(f"pairing failed: pairs={len(pairs)}, leftovers={len(leftovers)}")
    rng.shuffle(pairs)
    records = []
    for number, (a, b, stratum) in enumerate(pairs, 1):
        records.append({
            "pair_id": f"pair_{number:03d}",
            "candidate_a_index": a["pilot_index"],
            "candidate_b_index": b["pilot_index"],
            "pairing_stratum": stratum,
            "seed": SEED,
        })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"seed": SEED, "pair_count": len(records), "trace_count": 50}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
