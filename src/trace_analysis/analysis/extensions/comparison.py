from __future__ import annotations

import csv
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from .shared import ExtensionContext, case_dimension, case_value, ratio


def write_comparison(context: ExtensionContext, output: Path, dimension: str) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    field = "agent_models" if dimension == "model" else "harness"
    for case in context.cases:
        groups[(case["capability_id"], case_dimension(context, case, field))].append(case)
    rows = []
    for (capability_id, value), cases in sorted(groups.items()):
        outcomes = [case_value(context, case, "task_outcome", "outcome") for case in cases]
        feedback = [case_value(context, case, "user_feedback", "sentiment") for case in cases]
        difficulty = [case_value(context, case, "difficulty", "intrinsic_difficulty") for case in cases]
        outcome_labeled = sum(item in {"completed", "partially_completed", "failed"} for item in outcomes)
        feedback_labeled = sum(item in {"positive", "negative", "mixed"} for item in feedback)
        difficulty_labeled = sum(item in {"easy", "medium", "hard"} for item in difficulty)
        agent_models = [case_dimension(context, case, "agent_models") for case in cases]
        harnesses = [case_dimension(context, case, "harness") for case in cases]
        source_batches = [case_dimension(context, case, "source_batch") for case in cases]
        unmet = sum(case.get("fulfillment") in {"unmet", "partially_met"} for case in cases)
        node = context.nodes[capability_id]
        rows.append({
            "taxonomy_version": context.taxonomy_version, "capability_id": capability_id,
            "capability_path": node["capability_path"], dimension: value,
            "case_count": len({case["case_id"] for case in cases}),
            "trace_count": len({case.get("trace_id") for case in cases}),
            "outcome_labeled_count": outcome_labeled,
            "completed_count": sum(item == "completed" for item in outcomes),
            "completion_rate": ratio(sum(item == "completed" for item in outcomes), outcome_labeled),
            "unmet_count": unmet, "unmet_rate": ratio(unmet, len(cases)),
            "feedback_labeled_count": feedback_labeled,
            "negative_feedback_count": sum(item == "negative" for item in feedback),
            "negative_feedback_rate": ratio(sum(item == "negative" for item in feedback), feedback_labeled),
            "difficulty_labeled_count": difficulty_labeled,
            "hard_count": sum(item == "hard" for item in difficulty),
            "hard_rate": ratio(sum(item == "hard" for item in difficulty), difficulty_labeled),
            "agent_model_distribution": json.dumps(Counter(agent_models), ensure_ascii=False, sort_keys=True),
            "harness_distribution": json.dumps(Counter(harnesses), ensure_ascii=False, sort_keys=True),
            "source_batch_distribution": json.dumps(Counter(source_batches), ensure_ascii=False, sort_keys=True),
            "difficulty_distribution": json.dumps(Counter(str(item or "unknown") for item in difficulty),
                                                    ensure_ascii=False, sort_keys=True),
            "interpretation": "observational_only; inspect sample size and label coverage before comparison",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fields = list(rows[0]) if rows else ["taxonomy_version", "capability_id", "capability_path", dimension,
                                        "case_count", "trace_count"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "success", "row_count": len(rows), "output_path": str(output.resolve())}
