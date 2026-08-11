"""VSCode 可选审核入口：编辑最新运行生成的 CSV 后，直接运行本文件。"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_analysis.env import load_project_env
from trace_analysis.orchestration import run_update


REVIEWER = "wyren726"
ALLOW_SINGLE_CASE_PUBLICATION_FOR_SMOKE = False


def value_list(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    load_project_env(PROJECT_ROOT / ".env")
    latest_path = PROJECT_ROOT / "analysis" / "latest_run.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    review_csv = Path(latest["review_csv"])
    decisions = []
    with review_csv.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            action = (row.get("action") or "").strip()
            if not action:
                continue
            decision = {
                "capability_id": row["capability_id"], "action": action,
                "reason": (row.get("reason") or "").strip(),
            }
            if action == "rename":
                decision["name"] = (row.get("final_name") or "").strip()
            if action in {"merge", "assign_existing"}:
                decision["target_capability_id"] = (row.get("target_capability_id") or "").strip()
            for field in ("definition",):
                if (row.get(field) or "").strip():
                    decision[field] = row[field].strip()
            for field in ("inclusion_criteria", "exclusion_criteria"):
                if value_list(row.get(field) or ""):
                    decision[field] = value_list(row[field])
            decisions.append(decision)
    if not decisions:
        raise SystemExit(f"审核表尚未填写 action：{review_csv}")
    review = {
        "reviewer": REVIEWER, "reviewed_at": datetime.now().astimezone().isoformat(),
        "decisions": decisions,
    }
    if ALLOW_SINGLE_CASE_PUBLICATION_FOR_SMOKE:
        review["publication_policy"] = {
            "min_independent_cases": 1, "min_independent_traces": 1, "min_evidence_count": 1,
            "require_definition": True, "require_boundaries": True,
        }
    run_dir = Path(latest["manifest_path"]).parent
    decisions_path = run_dir / "review_decisions.generated.json"
    decisions_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = run_update(PROJECT_ROOT / "configs" / ".vscode_run.generated.json",
                        resume_run_id=latest["run_id"], review_decisions=decisions_path)
    print("\n审核处理完成")
    print(f"状态：{result['status']}")
    print(f"运行清单：{result['manifest_path']}")


if __name__ == "__main__":
    main()
