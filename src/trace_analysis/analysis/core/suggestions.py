from __future__ import annotations

import datetime as dt
import difflib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalized(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def _terms(value: str) -> set[str]:
    normalized = _normalized(value)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index:index + 2] for index in range(len(normalized) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _case_text(case: dict[str, Any]) -> str:
    return " ".join(str(case.get(field) or "") for field in
                    ("user_need", "expected_outcome", "unmet_need"))


def _connected_components(case_ids: list[str], similarity: dict[tuple[str, str], float],
                          threshold: float) -> list[list[str]]:
    unseen = set(case_ids)
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = {candidate for candidate in list(unseen)
                         if similarity.get(tuple(sorted((current, candidate))), 0.0) >= threshold}
            unseen -= neighbors
            stack.extend(neighbors)
        components.append(sorted(component))
    return sorted(components, key=lambda group: (-len(group), group))


def suggest_reviews(draft_core: Path, output_path: Path, merge_threshold: float = 0.65,
                    split_similarity_threshold: float = 0.32,
                    split_min_cases: int = 4, split_min_cluster_size: int = 2) -> dict[str, Any]:
    if not 0 <= merge_threshold <= 1 or not 0 <= split_similarity_threshold <= 1:
        raise ValueError("Similarity thresholds must be between 0 and 1")
    taxonomy = json.loads((draft_core / "capability_taxonomy.json").read_text(encoding="utf-8"))
    nodes = taxonomy["root"].get("children") or []
    cases = _jsonl(draft_core / "capability_cases.jsonl")
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_capability[case["capability_id"]].append(case)

    merge_candidates = []
    for left, right in itertools.combinations(nodes, 2):
        left_cases = by_capability[left["capability_id"]]
        right_cases = by_capability[right["capability_id"]]
        name_score = difflib.SequenceMatcher(None, _normalized(left["name"]), _normalized(right["name"])).ratio()
        left_terms = _terms(" ".join(_case_text(case) for case in left_cases))
        right_terms = _terms(" ".join(_case_text(case) for case in right_cases))
        case_score = _jaccard(left_terms, right_terms)
        combined = 0.55 * name_score + 0.45 * case_score
        if combined < merge_threshold:
            continue
        merge_candidates.append({
            "left_capability_id": left["capability_id"], "left_name": left["name"],
            "right_capability_id": right["capability_id"], "right_name": right["name"],
            "combined_score": round(combined, 4), "name_score": round(name_score, 4),
            "case_content_score": round(case_score, 4),
            "shared_terms": sorted(left_terms & right_terms)[:30],
            "left_case_ids": sorted(case["case_id"] for case in left_cases),
            "right_case_ids": sorted(case["case_id"] for case in right_cases),
            "suggested_action": "review_merge", "reason": "名称和/或案例内容相似，需人工核对纳入与排除边界。",
        })
    merge_candidates.sort(key=lambda item: item["combined_score"], reverse=True)

    split_candidates = []
    for node in nodes:
        node_cases = by_capability[node["capability_id"]]
        if len(node_cases) < split_min_cases:
            continue
        pair_scores: dict[tuple[str, str], float] = {}
        case_map = {case["case_id"]: case for case in node_cases}
        for left, right in itertools.combinations(node_cases, 2):
            pair_scores[tuple(sorted((left["case_id"], right["case_id"])))] = _jaccard(
                _terms(_case_text(left)), _terms(_case_text(right)))
        components = _connected_components(list(case_map), pair_scores, split_similarity_threshold)
        meaningful = [component for component in components if len(component) >= split_min_cluster_size]
        if len(meaningful) < 2 or sum(map(len, meaningful)) != len(node_cases):
            continue
        split_candidates.append({
            "capability_id": node["capability_id"], "name": node["name"],
            "case_count": len(node_cases), "suggested_action": "review_split",
            "reason": "节点案例形成多个低相似度连通组，可能包含不同的可独立验证能力。",
            "clusters": [{"case_ids": group,
                          "representative_needs": [case_map[case_id].get("user_need") for case_id in group[:3]]}
                         for group in meaningful],
        })
    result = {
        "suggestion_version": "v1", "generated_at": dt.datetime.now().astimezone().isoformat(),
        "source_taxonomy_version": taxonomy.get("taxonomy_version"),
        "method": "deterministic_name_and_case_content_similarity",
        "thresholds": {"merge": merge_threshold, "split_case_similarity": split_similarity_threshold,
                       "split_min_cases": split_min_cases, "split_min_cluster_size": split_min_cluster_size},
        "llm_review": {"status": "not_run"},
        "merge_candidates": merge_candidates, "split_candidates": split_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Suggestion output already exists: {output_path}")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "success", "merge_candidate_count": len(merge_candidates),
            "split_candidate_count": len(split_candidates), "output_path": str(output_path.resolve())}
