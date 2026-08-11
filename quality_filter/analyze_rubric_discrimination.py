#!/usr/bin/env python3
"""Validate 50 pilot scores and plot rubric-level score discrimination."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlcoding-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


PILOT_DIR = Path("quality_filter/pilot_50")
PLOT_DIR = PILOT_DIR / "plots"
INDIVIDUAL_PLOT_DIR = PLOT_DIR / "by_rubric"
QUALITY_WEIGHTS = {
    "Q1_TASK_COMPLETION_CORRECTNESS": 30,
    "Q2_INSTRUCTION_CONSTRAINT_COMPLIANCE": 15,
    "Q3_TOOL_ACTION_CORRECTNESS": 15,
    "Q4_EVIDENCE_GROUNDING": 10,
    "Q5_ERROR_RECOVERY": 10,
    "Q6_EFFICIENCY_NON_REDUNDANCY": 10,
    "Q7_REASONING_DECISION_QUALITY": 5,
    "Q8_FINAL_RESPONSE_QUALITY": 5,
}
VALUE_WEIGHTS = {
    "V1_TASK_COMPLEXITY": 25,
    "V2_LEARNABLE_AGENT_BEHAVIOR": 25,
    "V3_TARGET_CAPABILITY_RELEVANCE": 20,
    "V4_SKILL_TOOL_RARITY": 15,
    "V5_INFORMATION_DENSITY": 10,
    "V6_NOVELTY": 5,
}
SHORT_NAMES = {
    "Q1_TASK_COMPLETION_CORRECTNESS": "Task completion",
    "Q2_INSTRUCTION_CONSTRAINT_COMPLIANCE": "Constraint compliance",
    "Q3_TOOL_ACTION_CORRECTNESS": "Tool/action correctness",
    "Q4_EVIDENCE_GROUNDING": "Evidence grounding",
    "Q5_ERROR_RECOVERY": "Error recovery",
    "Q6_EFFICIENCY_NON_REDUNDANCY": "Efficiency",
    "Q7_REASONING_DECISION_QUALITY": "Reasoning/decisions",
    "Q8_FINAL_RESPONSE_QUALITY": "Final response",
    "V1_TASK_COMPLEXITY": "Task complexity",
    "V2_LEARNABLE_AGENT_BEHAVIOR": "Learnable behavior",
    "V3_TARGET_CAPABILITY_RELEVANCE": "Target relevance",
    "V4_SKILL_TOOL_RARITY": "Skill/tool rarity",
    "V5_INFORMATION_DENSITY": "Information density",
    "V6_NOVELTY": "Novelty",
}
DECISION_COLORS = {
    "high_quality": "#237a57",
    "usable": "#3572b0",
    "review": "#d28b26",
    "reject": "#b64242",
}
DECISION_MARKERS = {"high_quality": "o", "usable": "s", "review": "^", "reject": "X"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def weighted_score(results: list[dict[str, Any]], weights: dict[str, int], exclude: str | None = None) -> float:
    applicable = [
        item for item in results
        if item["applicability"] == "applicable"
        and item["raw_score"] is not None
        and item["rubric_id"] != exclude
    ]
    denominator = sum(weights[item["rubric_id"]] for item in applicable)
    if not denominator:
        return math.nan
    numerator = sum(item["raw_score"] / 4 * 100 * weights[item["rubric_id"]] for item in applicable)
    return numerator / denominator


def deterministic_jitter(index: int, rubric_id: str) -> float:
    digest = hashlib.sha256(f"{index}:{rubric_id}".encode()).digest()
    return ((int.from_bytes(digest[:2], "big") / 65535) - 0.5) * 0.24


def discrimination_label(unique_count: int, dominant_share: float, std: float) -> str:
    if unique_count >= 4 and dominant_share < 0.60 and std >= 0.75:
        return "high"
    if unique_count >= 3 and dominant_share < 0.80 and std >= 0.45:
        return "medium"
    return "low"


def validate(rows: list[dict[str, Any]], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    input_by_index = {row["pilot_index"]: row for row in inputs}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    score_delta_quality: list[float] = []
    score_delta_value: list[float] = []
    decision_mismatches: list[int] = []
    for row in rows:
        index = row["pilot_index"]
        source = input_by_index.get(index)
        if source is None:
            errors.append({"pilot_index": index, "error": "missing_source_sample"})
            continue
        if index in seen_indices:
            errors.append({"pilot_index": index, "error": "duplicate_pilot_index"})
        seen_indices.add(index)
        if row["sample_id"] != source.get("id"):
            errors.append({"pilot_index": index, "error": "delivery_id_mismatch"})
        if row["sample_id"] in seen_ids:
            errors.append({"pilot_index": index, "error": "duplicate_delivery_id"})
        seen_ids.add(row["sample_id"])
        quality_ids = {item["rubric_id"] for item in row["quality_rubric_results"]}
        value_ids = {item["rubric_id"] for item in row["training_value_rubric_results"]}
        if quality_ids != set(QUALITY_WEIGHTS):
            errors.append({"pilot_index": index, "error": "quality_rubric_set_mismatch"})
        if value_ids != set(VALUE_WEIGHTS):
            errors.append({"pilot_index": index, "error": "training_value_rubric_set_mismatch"})
        if len(row["hard_gate_results"]) != 13:
            errors.append({"pilot_index": index, "error": "hard_gate_count_not_13"})
        message_count = len(source.get("messages") or [])
        for section in ("hard_gate_results", "quality_rubric_results", "training_value_rubric_results"):
            for item in row[section]:
                for message_index in item.get("evidence_message_indices", []):
                    if message_index < 0 or message_index >= message_count:
                        errors.append({
                            "pilot_index": index,
                            "error": "evidence_index_out_of_range",
                            "section": section,
                            "message_index": message_index,
                            "message_count": message_count,
                        })
        recomputed_quality = weighted_score(row["quality_rubric_results"], QUALITY_WEIGHTS)
        recomputed_value = weighted_score(row["training_value_rubric_results"], VALUE_WEIGHTS)
        score_delta_quality.append(abs(recomputed_quality - row["quality_score"]))
        score_delta_value.append(abs(recomputed_value - row["training_value_score"]))
        if abs(recomputed_quality - row["quality_score"]) > 0.02:
            errors.append({"pilot_index": index, "error": "quality_score_formula_mismatch", "delta": round(abs(recomputed_quality-row["quality_score"]), 4)})
        if abs(recomputed_value - row["training_value_score"]) > 0.02:
            errors.append({"pilot_index": index, "error": "training_value_score_formula_mismatch", "delta": round(abs(recomputed_value-row["training_value_score"]), 4)})

        expected = "review"
        if row["hard_gate_status"] == "reject" or row["quality_score"] < 55:
            expected = "reject"
        elif row["hard_gate_status"] == "review" or row["quality_score"] < 70 or row["judge_confidence"] < 0.70:
            expected = "review"
        elif row["quality_score"] >= 85 and row["training_value_score"] >= 50:
            expected = "high_quality"
        elif row["quality_score"] >= 70 and row["training_value_score"] >= 50:
            expected = "usable"
        if row["decision"] != expected:
            decision_mismatches.append(index)
            warnings.append({"pilot_index": index, "warning": "decision_policy_mismatch", "reported": row["decision"], "expected": expected})
    return {
        "sample_count": len(rows),
        "source_sample_count": len(inputs),
        "unique_pilot_indices": len(seen_indices),
        "unique_delivery_ids": len(seen_ids),
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "max_quality_score_formula_delta": round(max(score_delta_quality, default=0), 6),
        "max_training_value_score_formula_delta": round(max(score_delta_value, default=0), 6),
        "decision_policy_mismatch_indices": decision_mismatches,
        "validation_passed": not errors,
    }


def analyze_group(rows: list[dict[str, Any]], result_key: str, weights: dict[str, int]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for rubric_id in weights:
        observations = []
        for row in rows:
            item = next(value for value in row[result_key] if value["rubric_id"] == rubric_id)
            if item["applicability"] != "applicable" or item["raw_score"] is None:
                continue
            observations.append((row, item["raw_score"], weighted_score(row[result_key], weights, exclude=rubric_id)))
        raw = np.array([value[1] for value in observations], dtype=float)
        leave_one_out = np.array([value[2] for value in observations], dtype=float)
        counts = Counter(int(value) for value in raw)
        probabilities = np.array(list(counts.values()), dtype=float) / len(raw)
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        normalized_entropy = entropy / math.log2(5)
        rho = float(spearmanr(raw, leave_one_out).statistic) if len(set(raw)) > 1 else math.nan
        unique_count = len(counts)
        dominant_share = max(counts.values()) / len(raw)
        std = float(np.std(raw, ddof=0))
        metrics.append({
            "rubric_id": rubric_id,
            "name": SHORT_NAMES[rubric_id],
            "applicable_count": len(raw),
            "na_count": len(rows) - len(raw),
            "score_counts": {str(score): counts.get(score, 0) for score in range(5)},
            "unique_score_count": unique_count,
            "mean": round(float(np.mean(raw)), 3),
            "std": round(std, 3),
            "dominant_score_share": round(dominant_share, 3),
            "normalized_entropy": round(normalized_entropy, 3),
            "spearman_vs_leave_one_out": None if math.isnan(rho) else round(rho, 3),
            "discrimination": discrimination_label(unique_count, dominant_share, std),
            "observations": observations,
        })
    return metrics


def plot_group(metrics: list[dict[str, Any]], filename: str, y_label: str, columns: int = 2) -> None:
    rows_count = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(13, rows_count * 4.0), sharex=True, sharey=True)
    axes_array = np.array(axes).reshape(-1)
    for axis, metric in zip(axes_array, metrics):
        for decision in DECISION_COLORS:
            subset = [value for value in metric["observations"] if value[0]["decision"] == decision]
            if not subset:
                continue
            x = [value[1] + deterministic_jitter(value[0]["pilot_index"], metric["rubric_id"]) for value in subset]
            y = [value[2] for value in subset]
            axis.scatter(x, y, s=40, alpha=0.78, color=DECISION_COLORS[decision], marker=DECISION_MARKERS[decision], label=decision)
        axis.set_title(f'{metric["rubric_id"].split("_")[0]} · {metric["name"]}', fontsize=11)
        axis.set_xlim(-0.35, 4.35)
        axis.set_ylim(0, 102)
        axis.set_xticks(range(5))
        axis.grid(True, alpha=0.22, linewidth=0.7)
        rho = metric["spearman_vs_leave_one_out"]
        annotation = f'n={metric["applicable_count"]}, NA={metric["na_count"]}\nρ={rho if rho is not None else "NA"}, {metric["discrimination"]}'
        axis.text(0.02, 0.98, annotation, transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#cccccc"})
        axis.set_xlabel("Rubric raw score (0–4; jittered)")
        axis.set_ylabel(y_label)
    for axis in axes_array[len(metrics):]:
        axis.set_visible(False)
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, frameon=False)
    fig.suptitle("Rubric score discrimination across the 50-trace pilot", fontsize=15, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for metric in metrics:
        fig, axis = plt.subplots(figsize=(7.2, 5.2))
        for decision in DECISION_COLORS:
            subset = [value for value in metric["observations"] if value[0]["decision"] == decision]
            if not subset:
                continue
            x = [value[1] + deterministic_jitter(value[0]["pilot_index"], metric["rubric_id"]) for value in subset]
            y = [value[2] for value in subset]
            axis.scatter(x, y, s=52, alpha=0.8, color=DECISION_COLORS[decision], marker=DECISION_MARKERS[decision], label=decision)
        axis.set_title(f'{metric["rubric_id"]} · {metric["name"]}')
        axis.set_xlim(-0.35, 4.35)
        axis.set_ylim(0, 102)
        axis.set_xticks(range(5))
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.set_xlabel("Rubric raw score (0–4; jittered)")
        axis.set_ylabel(y_label)
        rho = metric["spearman_vs_leave_one_out"]
        axis.text(0.02, 0.98,
                  f'n={metric["applicable_count"]}, NA={metric["na_count"]} | ρ={rho if rho is not None else "NA"} | spread={metric["discrimination"]}',
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"})
        axis.legend(loc="lower right", frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(INDIVIDUAL_PLOT_DIR / f'{metric["rubric_id"].lower()}.png', dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(PILOT_DIR / "quality_scores.normalized.jsonl")
    inputs = read_jsonl(PILOT_DIR / "pilot_input.jsonl")
    validation = validate(rows, inputs)
    (PILOT_DIR / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    quality_metrics = analyze_group(rows, "quality_rubric_results", QUALITY_WEIGHTS)
    value_metrics = analyze_group(rows, "training_value_rubric_results", VALUE_WEIGHTS)
    serializable = [
        {key: value for key, value in metric.items() if key != "observations"}
        for metric in quality_metrics + value_metrics
    ]
    (PILOT_DIR / "rubric_discrimination.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (PILOT_DIR / "rubric_discrimination.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["rubric_id", "name", "applicable_count", "na_count", "unique_score_count", "mean", "std", "dominant_score_share", "normalized_entropy", "spearman_vs_leave_one_out", "discrimination"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in serializable:
            writer.writerow({key: metric[key] for key in fields})

    plot_group(quality_metrics, "quality_rubrics_scatter.png", "Other quality rubrics score (0–100)")
    plot_group(value_metrics, "training_value_rubrics_scatter.png", "Other training-value rubrics score (0–100)")
    print(json.dumps({
        "validation_passed": validation["validation_passed"],
        "validation_error_count": validation["error_count"],
        "validation_warning_count": validation["warning_count"],
        "quality_discrimination": {item["rubric_id"]: item["discrimination"] for item in quality_metrics},
        "training_value_discrimination": {item["rubric_id"]: item["discrimination"] for item in value_metrics},
    }, ensure_ascii=False, indent=2))
    return 0 if validation["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
