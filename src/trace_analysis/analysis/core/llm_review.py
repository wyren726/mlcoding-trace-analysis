from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from ...features.custom.semantic import redact_for_api
from ...preprocessing.adapters.common import stable_id


class JSONClient(Protocol):
    model: str

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]: ...


SCHEMAS = {
    "merge": {
        "decision": "merge|keep_separate|insufficient_evidence", "reasoning": "string",
        "boundary_difference": "string", "suggested_name": "string",
        "inclusion_criteria": ["string"], "exclusion_criteria": ["string"],
        "cited_case_ids": ["string"],
    },
    "split": {
        "decision": "split|keep_together|insufficient_evidence", "reasoning": "string",
        "proposed_children": [{"name": "string", "definition": "string", "case_ids": ["string"]}],
        "cited_case_ids": ["string"],
    },
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate(kind: str, value: dict[str, Any], allowed_case_ids: set[str]) -> None:
    missing = set(SCHEMAS[kind]) - set(value)
    if missing:
        raise ValueError(f"Missing LLM review fields: {sorted(missing)}")
    allowed_decisions = ({"merge", "keep_separate", "insufficient_evidence"} if kind == "merge"
                         else {"split", "keep_together", "insufficient_evidence"})
    if value["decision"] not in allowed_decisions:
        raise ValueError(f"Invalid {kind} review decision: {value['decision']!r}")
    cited = value.get("cited_case_ids")
    if not isinstance(cited, list) or not set(cited).issubset(allowed_case_ids):
        raise ValueError("LLM review cites cases outside the candidate")
    if kind == "split" and value["decision"] == "split":
        children = value.get("proposed_children")
        if not isinstance(children, list) or len(children) < 2:
            raise ValueError("A split recommendation requires at least two children")
        assigned = [case_id for child in children for case_id in child.get("case_ids") or []]
        if len(assigned) != len(set(assigned)) or set(assigned) != allowed_case_ids:
            raise ValueError("A split recommendation must assign every candidate case exactly once")


def _candidate_rows(suggestions: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return ([(("merge"), item) for item in suggestions.get("merge_candidates") or []]
            + [(("split"), item) for item in suggestions.get("split_candidates") or []])


def _request(kind: str, candidate: dict[str, Any], cases: dict[str, dict[str, Any]],
             max_input_chars: int) -> tuple[str, set[str]]:
    case_ids = set(candidate.get("left_case_ids") or []) | set(candidate.get("right_case_ids") or [])
    if kind == "split":
        case_ids = {case_id for cluster in candidate.get("clusters") or [] for case_id in cluster.get("case_ids") or []}
    selected = [{key: case.get(key) for key in ("case_id", "user_need", "expected_outcome", "unmet_need",
                                                "fulfillment", "evidence")}
                for case_id in sorted(case_ids) if (case := cases.get(case_id))]
    payload = json.dumps({"candidate_type": kind, "rule_candidate": candidate, "cases": selected},
                         ensure_ascii=False, separators=(",", ":"))
    if len(payload) > max_input_chars:
        payload = payload[:max_input_chars] + "\n[INPUT_TRUNCATED]"
    return redact_for_api(payload), case_ids


def review_suggestions_with_llm(draft_core: Path, suggestions_path: Path, output_root: Path,
                                client: JSONClient, provider: str, dry_run: bool = False,
                                limit: int | None = None, max_input_chars: int = 24000) -> dict[str, Any]:
    suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in _jsonl(draft_core / "capability_cases.jsonl")}
    now = dt.datetime.now().astimezone()
    fingerprint = hashlib.sha256(
        f"{suggestions_path.resolve()}|{client.model}|v1".encode()).hexdigest()[:8]
    run_id = f"{'dry_run' if dry_run else 'run'}_{now.strftime('%Y%m%d_%H%M%S')}_{fingerprint}"
    run_dir = output_root / "llm_review_suggestions" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / ("requests.jsonl" if dry_run else "reviews.jsonl")
    errors = run_dir / "errors.jsonl"
    processed = error_count = 0
    with output.open("w", encoding="utf-8") as target, errors.open("w", encoding="utf-8") as error_out:
        for kind, candidate in _candidate_rows(suggestions):
            if limit is not None and processed + error_count >= limit:
                break
            candidate_id = stable_id("reviewcand", suggestions.get("source_taxonomy_version"), kind, candidate)
            user, allowed_case_ids = _request(kind, candidate, cases, max_input_chars)
            system = ("你是 Coding Agent 能力体系审核助手。只评估给定候选，不得修改或发布能力树。"
                      "必须返回 JSON；只能引用输入中存在的 case_id。证据不足时选择 insufficient_evidence。"
                      f"输出结构：{json.dumps(SCHEMAS[kind], ensure_ascii=False)}")
            if dry_run:
                target.write(json.dumps({"candidate_id": candidate_id, "candidate_type": kind,
                                         "system": system, "user": user}, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
                processed += 1
                continue
            try:
                value, api_meta = client.complete_json(system, user)
                _validate(kind, value, allowed_case_ids)
                target.write(json.dumps({"candidate_id": candidate_id, "candidate_type": kind,
                                         "source_taxonomy_version": suggestions.get("source_taxonomy_version"),
                                         "rule_candidate": candidate, "llm_review": value,
                                         "generated_by": {"provider": provider, "model": client.model,
                                                          "prompt_version": "v1"},
                                         "api_meta": api_meta}, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
                processed += 1
            except Exception as exc:
                error_out.write(json.dumps({"candidate_id": candidate_id, "candidate_type": kind,
                                            "error": str(exc)}, ensure_ascii=False) + "\n")
                error_count += 1
    status = "dry_run" if dry_run else ("success" if not error_count else ("partial" if processed else "failed"))
    manifest = {"run_id": run_id, "status": status, "source_suggestions": str(suggestions_path.resolve()),
                "source_taxonomy_version": suggestions.get("source_taxonomy_version"), "provider": provider,
                "model": client.model, "prompt_version": "v1", "record_count": processed,
                "error_count": error_count, "output_path": str(output.resolve()), "created_at": now.isoformat()}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    return manifest
