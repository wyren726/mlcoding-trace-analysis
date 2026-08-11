"""VSCode 日常入口：修改下方配置，然后点击“运行 Python 文件”。"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trace_analysis.env import load_project_env
from trace_analysis.orchestration import run_update


# ===== 日常只需要修改这里 =====
SOURCE = "raw_data/sls-logs.json"
RAW_RECORD_LIMIT = 100
SEMANTIC_TURN_LIMIT = 5
MODEL = "deepseek-v4-flash-0731"
RUN_LLM_FEATURES = True
RUN_EXTENSIONS = True
# ============================


def build_config() -> Path:
    base_path = PROJECT_ROOT / "configs" / "update.demo100.json"
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config.update({
        "sources": [SOURCE],
        "limit": RAW_RECORD_LIMIT,
        "semantic_limit": SEMANTIC_TURN_LIMIT,
        "semantic_features": (["user_feedback", "task_outcome", "difficulty", "behavior_quality",
                               "impact", "demand_capability_instances"] if RUN_LLM_FEATURES else []),
        "extensions": (["model_comparison", "harness_comparison", "training_selection",
                        "unmet_need_priority"] if RUN_EXTENSIONS else []),
        "enable_human_review": False,
    })
    config.setdefault("llm", {})["model"] = MODEL
    output = PROJECT_ROOT / "configs" / ".vscode_run.generated.json"
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_review_template(manifest: dict, core_dir: Path) -> Path:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    output = Path(manifest["manifest_path"]).parent / "capability_review.csv"
    fields = ["capability_id", "current_name", "action", "final_name", "target_capability_id",
              "reason", "definition", "inclusion_criteria", "exclusion_criteria"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in taxonomy["root"].get("children") or []:
            writer.writerow({"capability_id": node["capability_id"], "current_name": node["name"]})
    return output


def main() -> None:
    os.chdir(PROJECT_ROOT)
    load_project_env(PROJECT_ROOT / ".env")
    result = run_update(build_config())
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    core_dir = Path(manifest["final_core_path"] if manifest.get("final_core_path") else
                    manifest["stages"]["candidate_core"]["result"]["output_path"])
    review_csv = write_review_template({**result, **manifest}, core_dir)
    latest = PROJECT_ROOT / "analysis" / "latest_run.json"
    latest.write_text(json.dumps({
        "run_id": result["run_id"], "status": result["status"],
        "manifest_path": result["manifest_path"], "core_path": str(core_dir),
        "extensions_path": str(core_dir.parent / "extensions"), "review_csv": str(review_csv),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n日常分析完成")
    print(f"状态：{result['status']}")
    print(f"核心结果：{core_dir}")
    print(f"扩展结果：{core_dir.parent / 'extensions'}")
    print(f"可选审核表：{review_csv}")
    print(f"最新运行索引：{latest}")


if __name__ == "__main__":
    main()
