from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_CORE_FILES = (
    "capability_taxonomy.json",
    "capability_distribution.csv",
    "capability_cases.jsonl",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _walk(node: dict[str, Any]):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def _requirement_quotes(case: dict[str, Any]) -> list[str]:
    quotes = []
    for evidence in case.get("evidence", []):
        if evidence.get("evidence_type") in {"requirement", "requirement_acceptance"}:
            quote = str(evidence.get("quote", "")).strip()
            if quote and quote not in quotes:
                quotes.append(quote)
    return quotes


def _public_case(case: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for item in case.get("evidence", []):
        evidence.append({
            "event_id": item.get("event_id"),
            "evidence_type": item.get("evidence_type"),
            "quote": item.get("quote", ""),
        })
    support = case.get("assignment_support") or {}
    return {
        "case_id": case.get("case_id"),
        "source_case_id": case.get("source_case_id"),
        "user_query": _requirement_quotes(case),
        "normalized_need": case.get("user_need", ""),
        "expected_outcome": case.get("expected_outcome", ""),
        "constraints": case.get("constraints", []),
        "fulfillment": case.get("fulfillment", "unknown"),
        "unmet_need": case.get("unmet_need", ""),
        "requirement_origin": support.get("support_type", "unknown"),
        "entailment_audit": support.get("entailment_audit"),
        "evidence": evidence,
        "trace_id": case.get("trace_id"),
        "session_id": case.get("session_id"),
        "turn_ids": case.get("turn_ids", []),
    }


def publish_capability_site(core_dir: Path, output_dir: Path, title: str) -> dict[str, Any]:
    core_dir = core_dir.resolve()
    missing = [name for name in REQUIRED_CORE_FILES if not (core_dir / name).is_file()]
    if missing:
        raise ValueError(f"Core directory is missing required files: {', '.join(missing)}")

    taxonomy = _read_json(core_dir / "capability_taxonomy.json")
    root = taxonomy.get("root")
    if not isinstance(root, dict):
        raise ValueError("capability_taxonomy.json must contain an object field named 'root'")

    cases_by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (core_dir / "capability_cases.jsonl").open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            capability_id = case.get("capability_id")
            if capability_id:
                cases_by_capability[str(capability_id)].append(_public_case(case))

    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = Path(__file__).with_name("static")
    for name in ("index.html", "app.js", "styles.css"):
        shutil.copyfile(assets / name, output_dir / name)

    nodes = list(_walk(root))
    leaves = [node for node in nodes if node.get("is_leaf") or not node.get("children")]
    total_queries = sum(int(node.get("query_count", 0) or 0) for node in leaves)
    metadata = {
        "site_title": title,
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "taxonomy_status": taxonomy.get("status"),
        "analysis_generated_at": taxonomy.get("generated_at"),
        "published_at": datetime.now().astimezone().isoformat(),
        "root_name": root.get("name"),
        "node_count": len(nodes),
        "leaf_count": len(leaves),
        "query_count": total_queries,
        "case_count": sum(len(value) for value in cases_by_capability.values()),
    }
    _write_json(output_dir / "data" / "metadata.json", metadata)
    _write_json(output_dir / "data" / "taxonomy.json", {"root": root})

    search_index = []
    evidence_index = []
    for node in leaves:
        capability_id = str(node.get("capability_id"))
        cases = cases_by_capability.get(capability_id, [])
        _write_json(output_dir / "data" / "cases" / f"{capability_id}.json", cases)
        search_index.append({
            "capability_id": capability_id,
            "name": node.get("name", ""),
            "path": node.get("path", []),
            "queries": [quote for case in cases for quote in case.get("user_query", [])],
        })
        for case in cases:
            evidence_index.append({
                "capability_id": capability_id,
                "capability_name": node.get("name", ""),
                "path": node.get("path", []),
                "case_id": case.get("case_id"),
                "user_query": case.get("user_query", []),
                "normalized_need": case.get("normalized_need", ""),
                "requirement_origin": case.get("requirement_origin", "unknown"),
                "fulfillment": case.get("fulfillment", "unknown"),
                "trace_id": case.get("trace_id"),
                "turn_ids": case.get("turn_ids", []),
            })
    _write_json(output_dir / "data" / "search-index.json", search_index)
    _write_json(output_dir / "data" / "evidence-index.json", evidence_index)
    _write_json(output_dir / "site-manifest.json", {
        "schema_version": "v1",
        "metadata": "data/metadata.json",
        "taxonomy": "data/taxonomy.json",
        "search_index": "data/search-index.json",
        "evidence_index": "data/evidence-index.json",
        "cases_pattern": "data/cases/{capability_id}.json",
    })
    (output_dir / "README.txt").write_text(
        "Capability research static site\n\n"
        "Preview locally:\n"
        f"  python3 -m http.server 8000 --directory {output_dir}\n"
        "  then open http://localhost:8000\n\n"
        "The directory is self-contained and may be uploaded to any static hosting service.\n",
        encoding="utf-8",
    )
    return {"status": "success", "output_dir": str(output_dir.resolve()), **metadata}
