#!/usr/bin/env python3
"""Plot aggregate score structure for the validated 50-trace pilot."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlcoding-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PILOT_DIR = Path("quality_filter/pilot_50")
PLOT_DIR = PILOT_DIR / "plots" / "score_structure"
QUALITY_IDS = [
    "Q1_TASK_COMPLETION_CORRECTNESS",
    "Q2_INSTRUCTION_CONSTRAINT_COMPLIANCE",
    "Q3_TOOL_ACTION_CORRECTNESS",
    "Q4_EVIDENCE_GROUNDING",
    "Q5_ERROR_RECOVERY",
    "Q6_EFFICIENCY_NON_REDUNDANCY",
    "Q7_REASONING_DECISION_QUALITY",
    "Q8_FINAL_RESPONSE_QUALITY",
]
VALUE_IDS = [
    "V1_TASK_COMPLEXITY",
    "V2_LEARNABLE_AGENT_BEHAVIOR",
    "V3_TARGET_CAPABILITY_RELEVANCE",
    "V4_SKILL_TOOL_RARITY",
    "V5_INFORMATION_DENSITY",
    "V6_NOVELTY",
]
RUBRIC_IDS = QUALITY_IDS + VALUE_IDS
SHORT_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "V1", "V2", "V3", "V4", "V5", "V6"]
SOURCE_LABELS = {
    "mlcoding_delivery_20260805_205233.jsonl": "legacy API logs",
    "mlcoding_delivery_claude-4.6-opus-20260205_20260807_155721.jsonl": "Claude 4.6 Opus",
    "mlcoding_delivery_claude-4.8-opus-20260528_20260807_155721.jsonl": "Claude 4.8 Opus dated",
    "mlcoding_delivery_claude-opus-4.8_20260807_155721.jsonl": "Claude Opus 4.8",
}
SOURCE_COLORS = {
    "legacy API logs": "#5b76b2",
    "Claude 4.6 Opus": "#b44e4e",
    "Claude 4.8 Opus dated": "#4d9777",
    "Claude Opus 4.8": "#d08b32",
}
GATE_MARKERS = {"pass": "o", "review": "^", "reject": "X"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def rubric_vector(row: dict[str, Any]) -> list[float]:
    results = row["quality_rubric_results"] + row["training_value_rubric_results"]
    by_id = {item["rubric_id"]: item for item in results}
    return [
        np.nan if by_id[rubric_id]["raw_score"] is None else float(by_id[rubric_id]["raw_score"])
        for rubric_id in RUBRIC_IDS
    ]


def source_label(row: dict[str, Any]) -> str:
    return SOURCE_LABELS.get(row["source_file"], row["source_file"])


def plot_quality_value(rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 7.2))
    for source in SOURCE_COLORS:
        for gate_status in GATE_MARKERS:
            subset = [row for row in rows if source_label(row) == source and row["hard_gate_status"] == gate_status]
            if not subset:
                continue
            axis.scatter(
                [row["quality_score"] for row in subset],
                [row["training_value_score"] for row in subset],
                s=72,
                alpha=0.82,
                color=SOURCE_COLORS[source],
                marker=GATE_MARKERS[gate_status],
                edgecolors="black" if gate_status != "pass" else "none",
                linewidths=0.7,
            )
    for x, label in ((55, "reject/review"), (70, "review/usable"), (85, "usable/high")):
        axis.axvline(x, color="#777777", linewidth=0.9, linestyle="--", alpha=0.7)
        axis.text(x + 0.5, 99, label, rotation=90, va="top", fontsize=8, color="#555555")
    axis.axhline(50, color="#777777", linewidth=0.9, linestyle=":", alpha=0.75)
    axis.text(99, 51, "training-value threshold", ha="right", va="bottom", fontsize=8, color="#555555")
    for row in rows:
        if row["hard_gate_status"] != "pass" or row["decision"] == "reject":
            axis.annotate(str(row["pilot_index"]), (row["quality_score"], row["training_value_score"]),
                          xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_xlim(20, 102)
    axis.set_ylim(35, 102)
    axis.set_xlabel("Behavior quality score (0–100)")
    axis.set_ylabel("Training value score (0–100)")
    axis.set_title("Quality and training value for 50 validated traces")
    axis.grid(True, alpha=0.2)

    source_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=8)
        for label, color in SOURCE_COLORS.items()
    ]
    gate_handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color="#555555", label=f"gate: {status}", markersize=8)
        for status, marker in GATE_MARKERS.items()
    ]
    axis.legend(handles=source_handles + gate_handles, loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "quality_vs_training_value.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def prepare_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.array([rubric_vector(row) for row in rows], dtype=float)
    column_means = np.nanmean(raw, axis=0)
    imputed = np.where(np.isnan(raw), column_means, raw)
    standardized = StandardScaler().fit_transform(imputed)
    return raw, imputed, standardized


def plot_heatmap(rows: list[dict[str, Any]], raw: np.ndarray, standardized: np.ndarray) -> list[int]:
    order = leaves_list(linkage(standardized, method="ward")).tolist()
    ordered = raw[order]
    masked = np.ma.masked_invalid(ordered)
    cmap = plt.get_cmap("viridis", 5).copy()
    cmap.set_bad("#d9d9d9")
    fig, axis = plt.subplots(figsize=(13.5, 12.5))
    image = axis.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=4.5)
    axis.set_xticks(range(len(SHORT_LABELS)), SHORT_LABELS)
    axis.set_yticks(range(len(order)), [f'{rows[index]["pilot_index"]:02d}' for index in order], fontsize=7)
    axis.set_xlabel("Rubric (raw score 0–4; gray = NA)")
    axis.set_ylabel("Trace pilot_index, Ward-clustered")
    axis.set_title("Rubric score profiles across the 50-trace pilot")
    axis.axvline(7.5, color="white", linewidth=2.2)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02, ticks=range(5))
    colorbar.set_label("Raw rubric score")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rubric_heatmap_clustered.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return order


def plot_pca(rows: list[dict[str, Any]], standardized: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(standardized)
    fig, axis = plt.subplots(figsize=(10.5, 7.4))
    for source in SOURCE_COLORS:
        for gate_status in GATE_MARKERS:
            indices = [i for i, row in enumerate(rows) if source_label(row) == source and row["hard_gate_status"] == gate_status]
            if not indices:
                continue
            axis.scatter(coordinates[indices, 0], coordinates[indices, 1], s=72, alpha=0.82,
                         color=SOURCE_COLORS[source], marker=GATE_MARKERS[gate_status],
                         edgecolors="black" if gate_status != "pass" else "none", linewidths=0.7)
    for i, row in enumerate(rows):
        if row["hard_gate_status"] != "pass" or row["decision"] == "reject":
            axis.annotate(str(row["pilot_index"]), coordinates[i], xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.axhline(0, color="#888888", linewidth=0.7, alpha=0.5)
    axis.axvline(0, color="#888888", linewidth=0.7, alpha=0.5)
    axis.grid(True, alpha=0.18)
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    axis.set_title("PCA of 14-rubric score profiles (NA mean-imputed)")
    source_handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=8) for label, color in SOURCE_COLORS.items()]
    gate_handles = [plt.Line2D([], [], marker=marker, linestyle="", color="#555555", label=f"gate: {status}", markersize=8) for status, marker in GATE_MARKERS.items()]
    axis.legend(handles=source_handles + gate_handles, loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rubric_pca.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pca, coordinates


def cluster_diagnostics(standardized: np.ndarray) -> dict[str, Any]:
    scores = {}
    for clusters in range(2, 7):
        labels = KMeans(n_clusters=clusters, random_state=20260811, n_init=30).fit_predict(standardized)
        scores[str(clusters)] = round(float(silhouette_score(standardized, labels)), 4)
    best = max(scores, key=scores.get)
    return {"silhouette_by_k": scores, "best_k": int(best), "best_silhouette": scores[best]}


def main() -> int:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(PILOT_DIR / "quality_scores.normalized.jsonl")
    validation = json.loads((PILOT_DIR / "validation_report.json").read_text(encoding="utf-8"))
    if not validation.get("validation_passed") or len(rows) != 50:
        raise ValueError("score data must contain 50 fully validated records")
    raw, imputed, standardized = prepare_matrix(rows)
    plot_quality_value(rows)
    heatmap_order = plot_heatmap(rows, raw, standardized)
    pca, coordinates = plot_pca(rows, standardized)
    diagnostics = cluster_diagnostics(standardized)
    report = {
        "sample_count": len(rows),
        "quality_score": {
            "min": min(row["quality_score"] for row in rows),
            "max": max(row["quality_score"] for row in rows),
            "mean": round(float(np.mean([row["quality_score"] for row in rows])), 3),
            "std": round(float(np.std([row["quality_score"] for row in rows])), 3),
        },
        "training_value_score": {
            "min": min(row["training_value_score"] for row in rows),
            "max": max(row["training_value_score"] for row in rows),
            "mean": round(float(np.mean([row["training_value_score"] for row in rows])), 3),
            "std": round(float(np.std([row["training_value_score"] for row in rows])), 3),
        },
        "hard_gate_status_counts": dict(Counter(row["hard_gate_status"] for row in rows)),
        "source_counts": dict(Counter(source_label(row) for row in rows)),
        "pca": {
            "explained_variance_ratio": [round(float(value), 4) for value in pca.explained_variance_ratio_],
            "explained_variance_total": round(float(pca.explained_variance_ratio_.sum()), 4),
            "na_handling": "column-mean imputation before standardization",
            "loadings": {
                rubric_id: [round(float(pca.components_[0, i]), 4), round(float(pca.components_[1, i]), 4)]
                for i, rubric_id in enumerate(RUBRIC_IDS)
            },
        },
        "unsupervised_cluster_diagnostics": diagnostics,
        "heatmap_order_pilot_indices": [rows[index]["pilot_index"] for index in heatmap_order],
        "interpretation_warning": "Clusters are exploratory; final decisions are partly derived from quality scores and cannot serve as independent ground truth.",
    }
    (PILOT_DIR / "score_structure_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
