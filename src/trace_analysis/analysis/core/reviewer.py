from __future__ import annotations

import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...preprocessing.adapters.common import stable_id


ACTIONS = {"accept", "reject", "rename", "merge", "assign_existing", "split"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _node(base: dict[str, Any], version: str, **updates: Any) -> dict[str, Any]:
    value = {**base, **updates}
    value["taxonomy_version"] = version
    value["status"] = "published"
    value.pop("match_score", None)
    value.pop("suggested_existing_capability_id", None)
    return value


def _refresh_counts(node: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    node_cases = [case for case in cases if case.get("capability_id") == node["capability_id"]
                  and case.get("assignment_status") == "published"]
    node["case_count"] = len({case["case_id"] for case in node_cases})
    node["trace_count"] = len({case.get("trace_id") for case in node_cases})
    node["evidence_count"] = sum(len(case.get("evidence") or []) for case in node_cases)


def _publication_gaps(node: dict[str, Any], cases: list[dict[str, Any]], policy: dict[str, Any]) -> list[str]:
    node_cases = [case for case in cases if case.get("capability_id") == node["capability_id"]
                  and case.get("assignment_status") == "published"]
    gaps: list[str] = []
    if len({case.get("case_id") for case in node_cases}) < int(policy["min_independent_cases"]):
        gaps.append("independent_case_count")
    if len({case.get("trace_id") for case in node_cases}) < int(policy["min_independent_traces"]):
        gaps.append("independent_trace_count")
    evidence_count = sum(len(case.get("evidence") or []) for case in node_cases)
    if evidence_count < int(policy["min_evidence_count"]):
        gaps.append("direct_evidence_count")
    if policy["require_definition"] and not str(node.get("definition") or "").strip():
        gaps.append("definition")
    if policy["require_boundaries"] and (not node.get("inclusion_criteria") or not node.get("exclusion_criteria")):
        gaps.append("inclusion_and_exclusion_criteria")
    return gaps


def publish_reviewed_core(draft_core: Path, decisions_path: Path, output_root: Path,
                          taxonomy_version: str) -> dict[str, Any]:
    draft_taxonomy = json.loads((draft_core / "capability_taxonomy.json").read_text(encoding="utf-8"))
    cases = _jsonl(draft_core / "capability_cases.jsonl")
    review = json.loads(decisions_path.read_text(encoding="utf-8"))
    policy = {
        "min_independent_cases": 2, "min_independent_traces": 2, "min_evidence_count": 2,
        "require_definition": True, "require_boundaries": True,
        **(review.get("publication_policy") or {}),
    }
    reviewer = str(review.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("reviewer is required")
    draft_nodes = {node["capability_id"]: node for node in draft_taxonomy["root"].get("children") or []}
    decisions: dict[str, dict[str, Any]] = {}
    for decision in review.get("decisions") or []:
        capability_id = decision.get("capability_id")
        action = decision.get("action")
        if capability_id not in draft_nodes:
            raise ValueError(f"Decision references unknown capability: {capability_id}")
        if action not in ACTIONS:
            raise ValueError(f"Unsupported review action: {action}")
        if capability_id in decisions:
            raise ValueError(f"Duplicate decision for capability: {capability_id}")
        if not str(decision.get("reason") or "").strip():
            raise ValueError(f"Review reason is required for capability: {capability_id}")
        decisions[capability_id] = decision

    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_capability[case["capability_id"]].append(case)
        case["taxonomy_version"] = taxonomy_version
        case["assignment_status"] = "candidate_unreviewed"

    published: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    deferred_merges: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for capability_id, base in draft_nodes.items():
        decision = decisions.get(capability_id)
        if decision is None:
            candidates.append({**base, "status": "candidate_unreviewed"})
            continue
        action = decision["action"]
        if action == "reject":
            candidates.append({**base, "status": "rejected", "review_reason": decision["reason"]})
            for case in by_capability[capability_id]:
                case["assignment_status"] = "rejected"
            continue
        if action in {"merge", "assign_existing"}:
            deferred_merges.append((base, decision))
            continue
        if action == "split":
            original_ids = {case["case_id"] for case in by_capability[capability_id]}
            assigned: list[str] = []
            children = decision.get("children") or []
            if len(children) < 2:
                raise ValueError(f"split requires at least two children: {capability_id}")
            for child in children:
                child_ids = child.get("case_ids") or []
                assigned.extend(child_ids)
                child_name = str(child.get("name") or "").strip()
                if not child_name:
                    raise ValueError(f"split child name is required: {capability_id}")
                child_id = child.get("capability_id") or stable_id("cap", capability_id, child_name)
                published[child_id] = _node(base, taxonomy_version, capability_id=child_id,
                                             name=child_name, path=["Coding Agent 能力", child_name],
                                             definition=child.get("definition") or "",
                                             inclusion_criteria=child.get("inclusion_criteria") or [],
                                             exclusion_criteria=child.get("exclusion_criteria") or [],
                                             review_action="split", split_from=capability_id)
                for case in by_capability[capability_id]:
                    if case["case_id"] in child_ids:
                        case["capability_id"] = child_id
                        case["assignment_status"] = "published"
            if len(assigned) != len(set(assigned)) or set(assigned) != original_ids:
                raise ValueError(f"split must assign every case exactly once: {capability_id}")
            continue
        name = decision.get("name") if action == "rename" else decision.get("name", base["name"])
        if not str(name or "").strip():
            raise ValueError(f"Published capability name is required: {capability_id}")
        published[capability_id] = _node(
            base, taxonomy_version, name=name, path=["Coding Agent 能力", name], review_action=action,
            definition=decision.get("definition", base.get("definition", "")),
            inclusion_criteria=decision.get("inclusion_criteria", base.get("inclusion_criteria", [])),
            exclusion_criteria=decision.get("exclusion_criteria", base.get("exclusion_criteria", [])),
        )
        for case in by_capability[capability_id]:
            case["assignment_status"] = "published"

    for base, decision in deferred_merges:
        source_id = base["capability_id"]
        target_id = decision.get("target_capability_id")
        if target_id not in published:
            raise ValueError(f"Merge target must be accepted in this review: {target_id}")
        for case in by_capability[source_id]:
            case["capability_id"] = target_id
            case["assignment_status"] = "published"
        candidates.append({**base, "status": "merged", "merged_into": target_id,
                           "review_reason": decision["reason"]})

    published_nodes = sorted(published.values(), key=lambda item: item["name"])
    eligible_nodes: list[dict[str, Any]] = []
    for node in published_nodes:
        _refresh_counts(node, cases)
        gaps = _publication_gaps(node, cases, policy)
        if gaps:
            node_id = node["capability_id"]
            candidates.append({**node, "status": "candidate_insufficient_evidence",
                               "publication_gaps": gaps})
            for case in cases:
                if case.get("capability_id") == node_id and case.get("assignment_status") == "published":
                    case["assignment_status"] = "candidate_insufficient_evidence"
        else:
            eligible_nodes.append(node)
    published_nodes = eligible_nodes
    now = dt.datetime.now().astimezone()
    taxonomy = {
        "taxonomy_version": taxonomy_version, "status": "published", "generated_at": now.isoformat(),
        "base_draft_version": draft_taxonomy.get("taxonomy_version"),
        "review": {"reviewer": reviewer, "reviewed_at": review.get("reviewed_at") or now.isoformat(),
                   "decision_count": len(decisions), "publication_policy": policy},
        "root": {"capability_id": "capability_root", "name": "Coding Agent 能力",
                 "children": published_nodes},
        "candidates": candidates,
    }
    run_dir = output_root / f"version_{taxonomy_version}" / "core"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "capability_taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "capability_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            if case.get("assignment_status") != "published":
                continue
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    for filename, statuses in (
        ("capability_candidate_cases.jsonl", {"candidate_unreviewed", "candidate_insufficient_evidence"}),
        ("capability_rejected_cases.jsonl", {"rejected"}),
    ):
        with (run_dir / filename).open("w", encoding="utf-8") as handle:
            for case in cases:
                if case.get("assignment_status") in statuses:
                    handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    fields = ["capability_id", "parent_id", "level", "capability_path", "name", "is_leaf",
              "status", "case_count", "trace_count", "evidence_count"]
    with (run_dir / "capability_distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in published_nodes:
            writer.writerow({**{key: node[key] for key in fields if key != "capability_path"},
                             "capability_path": " > ".join(node["path"])})
    audit = {**review, "source_decisions_path": str(decisions_path.resolve()),
             "published_at": now.isoformat()}
    (run_dir / "review_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"taxonomy_version": taxonomy_version, "status": "published",
            "published_capability_count": len(published_nodes), "candidate_count": len(candidates),
            "published_case_count": sum(case["assignment_status"] == "published" for case in cases),
            "unpublished_case_count": sum(case["assignment_status"] != "published" for case in cases),
            "output_path": str(run_dir.resolve())}
