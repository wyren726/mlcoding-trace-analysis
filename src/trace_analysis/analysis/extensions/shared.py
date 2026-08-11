from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExtensionContext:
    taxonomy_version: str
    cases: list[dict[str, Any]]
    nodes: dict[str, dict[str, Any]]
    features: dict[tuple[str, str], dict[str, Any]]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_context(core_dir: Path, feature_paths: list[Path]) -> ExtensionContext:
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    nodes: dict[str, dict[str, Any]] = {}

    def visit(node: dict[str, Any], path: list[str]) -> None:
        node_path = path + [node["name"]]
        nodes[node["capability_id"]] = {**node, "capability_path": " > ".join(node_path)}
        for child in node.get("children") or []:
            visit(child, node_path)

    for child in taxonomy["root"].get("children") or []:
        visit(child, ["Coding Agent 能力"])
    cases = [case for case in _jsonl(core_dir / "capability_cases.jsonl")
             if case.get("assignment_status", "published") == "published"]
    features: dict[tuple[str, str], dict[str, Any]] = {}
    for path in feature_paths:
        for record in _jsonl(path):
            key = (str(record.get("turn_id")), str(record.get("feature_set")))
            if key in features and features[key].get("values") != record.get("values"):
                raise ValueError(f"Conflicting feature records for turn and feature set: {key}")
            features[key] = record
    return ExtensionContext(taxonomy.get("taxonomy_version", "unknown"), cases, nodes, features)


def case_feature_records(context: ExtensionContext, case: dict[str, Any], feature_set: str) -> list[dict[str, Any]]:
    return [context.features[(str(turn_id), feature_set)] for turn_id in case.get("turn_ids") or []
            if (str(turn_id), feature_set) in context.features]


def case_value(context: ExtensionContext, case: dict[str, Any], feature_set: str, field: str) -> Any:
    values = [record.get("values", {}).get(field) for record in case_feature_records(context, case, feature_set)
              if record.get("values", {}).get(field) is not None]
    unique = list(dict.fromkeys(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values))
    if not unique:
        return None
    if len(unique) > 1:
        return "mixed"
    return json.loads(unique[0])


def case_dimension(context: ExtensionContext, case: dict[str, Any], field: str) -> str:
    values = case_value(context, case, "basic_turn_features", field)
    if field == "agent_models":
        return values[0] if isinstance(values, list) and len(values) == 1 else ("mixed" if values else "unknown")
    return str(values or "unknown")


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
