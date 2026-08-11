from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..shared import ExtensionContext, case_feature_records, case_value


def run(context: ExtensionContext, output: Path) -> dict[str, Any]:
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in context.cases:
        by_trace[str(case.get("trace_id"))].append(case)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    count = 0
    records: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as handle:
        for trace_id, cases in sorted(by_trace.items()):
            turn_ids = sorted({str(turn_id) for case in cases for turn_id in case.get("turn_ids") or []})
            basic_covered = sum((turn_id, "basic_turn_features") in context.features for turn_id in turn_ids)
            outcomes = [case_value(context, case, "task_outcome", "outcome") for case in cases]
            verified = [case_value(context, case, "task_outcome", "verified") for case in cases]
            feedback = [case_value(context, case, "user_feedback", "sentiment") for case in cases]
            difficulty = [case_value(context, case, "difficulty", "intrinsic_difficulty") for case in cases]
            unmet = any(case.get("fulfillment") in {"unmet", "partially_met"} for case in cases)
            failed = any(value == "failed" for value in outcomes)
            known_outcomes = [value for value in outcomes if value is not None]
            completed = bool(known_outcomes) and all(value == "completed" for value in known_outcomes)
            externally_verified = any(value is True for value in verified)
            positive = any(value == "positive" for value in feedback)
            negative = any(value == "negative" for value in feedback)
            medium_or_hard = any(value in {"medium", "hard"} for value in difficulty)
            hard = any(value == "hard" for value in difficulty)
            data_complete = bool(turn_ids) and basic_covered == len(turn_ids)
            behavior = [case_value(context, case, "behavior_quality", "quality") for case in cases]
            good_behavior = any(value in {"high", "exemplary", "good", "pass"} for value in behavior)
            if not data_complete:
                use = "exclude"
                reasons = ["缺少部分 Turn 的基础特征"]
            elif completed and externally_verified and positive and medium_or_hard and good_behavior and not unmet:
                use = "positive_sft"
                reasons = ["完成且有验证证据", "存在正向用户反馈", "任务为中高难度"]
            elif (failed or unmet) and (negative or any(case.get("unmet_need") for case in cases)):
                use = "repair_trace"
                reasons = ["存在失败或未满足需求", "包含可定位的负向反馈或能力缺口"]
            else:
                use = "evaluation_candidate"
                reasons = ["不满足严格正向训练标准，保留用于评测或人工复核"]
            record = {
                "taxonomy_version": context.taxonomy_version, "trace_id": trace_id,
                "turn_ids": turn_ids, "case_ids": sorted(case["case_id"] for case in cases),
                "capability_ids": sorted({case["capability_id"] for case in cases}),
                "recommended_use": use, "reasons": reasons,
                "signals": {"data_complete": data_complete, "completed": completed,
                            "verified": externally_verified, "positive_feedback": positive,
                            "negative_feedback": negative, "medium_or_hard": medium_or_hard,
                            "hard": hard, "has_unmet_need": unmet, "has_failure": failed,
                            "behavior_quality": "high" if good_behavior else ("unavailable" if not behavior else "not_high")},
                "coverage": {"turn_count": len(turn_ids), "basic_feature_count": basic_covered,
                             "outcome_label_count": sum(value is not None for value in outcomes),
                             "feedback_label_count": sum(value is not None for value in feedback),
                             "difficulty_label_count": sum(value is not None for value in difficulty)},
                "limitations": (["缺少 behavior_quality 标签，不能进入 positive_sft"]
                                if not any(case_feature_records(context, case, "behavior_quality") for case in cases) else []),
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records.append(record)
            count += 1
        positives = [record for record in records if record["recommended_use"] == "positive_sft"]
        repairs = [record for record in records if record["recommended_use"] == "repair_trace"]
        used_pairs: set[tuple[str, str]] = set()
        for chosen in positives:
            for rejected in repairs:
                shared = sorted(set(chosen["capability_ids"]) & set(rejected["capability_ids"]))
                pair_key = (chosen["trace_id"], rejected["trace_id"])
                if not shared or pair_key in used_pairs:
                    continue
                pair = {
                    "taxonomy_version": context.taxonomy_version,
                    "pair_id": f"preference_{chosen['trace_id']}_{rejected['trace_id']}",
                    "recommended_use": "preference_pair", "capability_ids": shared,
                    "chosen_trace_id": chosen["trace_id"], "rejected_trace_id": rejected["trace_id"],
                    "reasons": ["同能力下存在高质量完成轨迹与可定位的修复轨迹"],
                    "limitations": ["发布到训练集前仍需人工确认任务可比性与偏好方向"],
                }
                handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
                used_pairs.add(pair_key)
                count += 1
                break
    return {"status": "success", "trace_count": count, "output_path": str(output.resolve())}


__all__ = ["run"]
