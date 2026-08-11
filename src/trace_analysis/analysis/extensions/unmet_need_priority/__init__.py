from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..shared import ExtensionContext, case_dimension, case_value, ratio


def run(context: ExtensionContext, output: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in context.cases:
        groups[case["capability_id"]].append(case)
    rows = []
    for capability_id, cases in sorted(groups.items()):
        unmet_cases = [case for case in cases if case.get("fulfillment") in {"unmet", "partially_met"}]
        if not unmet_cases:
            continue
        feedback = [case_value(context, case, "user_feedback", "sentiment") for case in unmet_cases]
        outcomes = [case_value(context, case, "task_outcome", "outcome") for case in unmet_cases]
        difficulty = [case_value(context, case, "difficulty", "intrinsic_difficulty") for case in unmet_cases]
        impacts = [case_value(context, case, "impact", "severity") for case in unmet_cases]
        traces = {case.get("trace_id") for case in unmet_cases}
        trace_count = len(traces)
        if trace_count >= 5 and len(unmet_cases) / len(cases) >= 0.5:
            tier = "high"
        elif trace_count >= 3:
            tier = "medium"
        else:
            tier = "insufficient_evidence"
        node = context.nodes[capability_id]
        rows.append({
            "taxonomy_version": context.taxonomy_version, "capability_id": capability_id,
            "capability_path": node["capability_path"], "total_case_count": len(cases),
            "unmet_case_count": len(unmet_cases), "unmet_trace_count": trace_count,
            "unmet_rate": ratio(len(unmet_cases), len(cases)),
            "negative_feedback_count": sum(value == "negative" for value in feedback),
            "feedback_label_coverage": ratio(sum(value is not None for value in feedback), len(unmet_cases)),
            "failed_count": sum(value == "failed" for value in outcomes),
            "outcome_label_coverage": ratio(sum(value is not None for value in outcomes), len(unmet_cases)),
            "hard_count": sum(value == "hard" for value in difficulty),
            "difficulty_label_coverage": ratio(sum(value is not None for value in difficulty), len(unmet_cases)),
            "model_coverage_count": len({case_dimension(context, case, "agent_models") for case in unmet_cases}),
            "harness_coverage_count": len({case_dimension(context, case, "harness") for case in unmet_cases}),
            "evidence_count": sum(len(case.get("evidence") or []) for case in unmet_cases),
            "gap_clarity_rate": ratio(sum(bool(str(case.get("unmet_need") or "").strip())
                                           for case in unmet_cases), len(unmet_cases)),
            "evidence_case_coverage": ratio(sum(bool(case.get("evidence")) for case in unmet_cases),
                                             len(unmet_cases)),
            "source_coverage_count": len({case_dimension(context, case, "source_batch") for case in unmet_cases}),
            "high_impact_count": sum(value in {"high", "critical"} for value in impacts),
            "impact_label_coverage": ratio(sum(value is not None for value in impacts), len(unmet_cases)),
            "priority_tier": tier,
            "interpretation": "priority is evidence-gated; dimensions are not collapsed into an opaque score",
        })
    rows.sort(key=lambda row: ({"high": 0, "medium": 1, "insufficient_evidence": 2}[row["priority_tier"]],
                               -row["unmet_trace_count"], -row["unmet_case_count"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fields = list(rows[0]) if rows else ["taxonomy_version", "capability_id", "capability_path",
                                        "unmet_case_count", "unmet_trace_count", "priority_tier"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "success", "row_count": len(rows), "output_path": str(output.resolve())}


__all__ = ["run"]
