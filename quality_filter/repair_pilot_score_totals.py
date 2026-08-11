#!/usr/bin/env python3
"""Repair deterministic total-score arithmetic while preserving judge rubric outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from analyze_rubric_discrimination import QUALITY_WEIGHTS, VALUE_WEIGHTS, weighted_score


PILOT_DIR = Path("quality_filter/pilot_50")
REPAIRS = {
    26: "quality_score",
    50: "training_value_score",
}


def main() -> int:
    audit = []
    for index, field in REPAIRS.items():
        path = PILOT_DIR / "scores" / f"score_{index:03d}.json"
        backup = PILOT_DIR / "scores" / f"score_{index:03d}.before_formula_fix.json"
        if not backup.exists():
            shutil.copy2(path, backup)
        value = json.loads(path.read_text(encoding="utf-8"))
        if field == "quality_score":
            corrected = round(weighted_score(value["quality_rubric_results"], QUALITY_WEIGHTS), 2)
        else:
            corrected = round(weighted_score(value["training_value_rubric_results"], VALUE_WEIGHTS), 2)
        previous = value[field]
        value[field] = corrected
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        audit.append({
            "pilot_index": index,
            "field": field,
            "previous": previous,
            "corrected": corrected,
            "method": "deterministic_weighted_recalculation_from_unchanged_raw_rubric_scores",
            "backup_file": backup.name,
        })
    (PILOT_DIR / "score_repair_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
