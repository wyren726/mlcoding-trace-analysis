from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .planner import (
    _atomic_json, _atomic_jsonl, _read_jsonl, _result_path, _stable_id,
)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())


def approve_queue_items(
    *,
    cycle_dir: Path,
    reviewer_name: str,
    reason: str,
    queue_ids: set[str] | None = None,
    issue_codes: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Append explicit approval events; never mutate the generated queue."""
    cycle = cycle_dir.expanduser().resolve()
    queue_path = cycle / "04_reanalysis_queue" / "reanalysis_queue.jsonl"
    queue = list(_read_jsonl(queue_path))
    selected = []
    for item in queue:
        if queue_ids is not None and str(item.get("queue_id")) not in queue_ids:
            continue
        if issue_codes is not None and not (
            set(str(value) for value in item.get("issue_codes") or []) & issue_codes
        ):
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError("No queue items matched the approval selector")
    name = reviewer_name.strip()
    why = reason.strip()
    if not name or not why:
        raise ValueError("reviewer_name and reason are required")
    approval_path = cycle / "04_reanalysis_queue" / "queue_approvals.jsonl"
    created_at = _now()
    for item in selected:
        _append_jsonl(approval_path, {
            "schema_version": "review-remediation-approval-v1",
            "approval_id": _stable_id(
                "approval", item["queue_id"], name, created_at
            ),
            "cycle_id": cycle.name,
            "queue_id": item["queue_id"],
            "decision": "approved",
            "reviewer_name": name,
            "reason": why,
            "created_at": created_at,
        })
    return {
        "cycle_id": cycle.name,
        "approved_count": len(selected),
        "queue_ids": [str(value["queue_id"]) for value in selected],
        "approval_path": str(approval_path),
    }


def _latest_approvals(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for value in _read_jsonl(path):
        latest[str(value.get("queue_id") or "")] = value
    return latest


def _screen_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("02_user_screen/user_screen__*.jsonl"))
    candidates.extend(sorted(run_dir.glob("02_analyze/user_screen__*.jsonl")))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot resolve one user_screen JSONL in {run_dir}")
    return candidates[0]


def prepare_approved_reanalysis(
    *, cycle_dir: Path, analysis_root: Path, preprocessed_root: Path
) -> dict[str, Any]:
    """Materialize Stage-03-only repair runs from explicitly approved items."""
    cycle = cycle_dir.expanduser().resolve()
    analysis = analysis_root.expanduser().resolve()
    preprocessed = preprocessed_root.expanduser().resolve()
    queue_path = cycle / "04_reanalysis_queue" / "reanalysis_queue.jsonl"
    approval_path = cycle / "04_reanalysis_queue" / "queue_approvals.jsonl"
    approvals = _latest_approvals(approval_path)
    approved = [
        value for value in _read_jsonl(queue_path)
        if (approvals.get(str(value.get("queue_id") or "")) or {}).get("decision")
        == "approved"
    ]
    if not approved:
        raise ValueError("No approved queue items; planning alone cannot start model calls")
    unsupported_modes = sorted({
        str(value.get("reanalysis_mode")) for value in approved
        if value.get("reanalysis_mode") != "agent_verify_only"
    })
    if unsupported_modes:
        raise ValueError(
            "This executor only prepares agent_verify_only jobs: "
            + ", ".join(unsupported_modes)
        )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in approved:
        groups[(
            str(value["batch_id"]), str(value["baseline_analysis_run_id"])
        )].append(value)

    jobs = []
    for (batch_id, baseline_run_id), items in sorted(groups.items()):
        trace_ids = {str(value["trace_id"]) for value in items}
        baseline_run = analysis / batch_id / baseline_run_id
        source_screen = _screen_path(baseline_run)
        digest = hashlib.sha256(
            "|".join(sorted(trace_ids)).encode("utf-8")
        ).hexdigest()[:8]
        repair_run_id = f"repair_{cycle.name}_{digest}"
        repair_run = cycle / "05_reanalysis" / batch_id / repair_run_id
        target_dir = repair_run / "02_user_screen"
        target_screen = target_dir / f"user_screen__{batch_id}__{repair_run_id}.jsonl"
        records = []
        for value in _read_jsonl(source_screen):
            if str(value.get("trace_id") or "") not in trace_ids:
                continue
            record = dict(value)
            record["analysis_run_id"] = repair_run_id
            provenance = dict(record.get("analysis_provenance") or {})
            provenance.update({
                "review_cycle_id": cycle.name,
                "supersedes_analysis_run_id": baseline_run_id,
                "source_user_screen_path": str(source_screen),
            })
            record["analysis_provenance"] = provenance
            records.append(record)
        found = {str(value.get("trace_id") or "") for value in records}
        missing = sorted(trace_ids - found)
        if missing:
            raise ValueError(f"Approved traces missing from user_screen: {missing[:5]}")
        _atomic_jsonl(target_screen, sorted(
            records, key=lambda value: str(value.get("trace_id") or "")
        ))
        _atomic_json(target_dir / "manifest.json", {
            "manifest_version": "1.0",
            "stage": "02_user_screen",
            "status": "reused_from_baseline",
            "batch_id": batch_id,
            "analysis_run_id": repair_run_id,
            "supersedes_analysis_run_id": baseline_run_id,
            "review_cycle_id": cycle.name,
            "selected_trace_count": len(trace_ids),
            "output_path": str(target_screen),
            "created_at": _now(),
        })
        job = {
            "schema_version": "review-remediation-job-v1",
            "job_id": _stable_id("repair_job", cycle.name, batch_id, repair_run_id),
            "cycle_id": cycle.name,
            "batch_id": batch_id,
            "baseline_analysis_run_id": baseline_run_id,
            "repair_analysis_run_id": repair_run_id,
            "batch_dir": str(preprocessed / batch_id),
            "repair_run_dir": str(repair_run),
            "input_user_screen": str(target_screen),
            "trace_ids": sorted(trace_ids),
            "queue_ids": sorted(str(value["queue_id"]) for value in items),
            "status": "prepared_waiting_for_model_execution",
            "required_prompt_versions": [
                "trace-agent-verification-v4",
                "trace-agent-verification-indexed-v6",
            ],
            "publication_allowed": False,
        }
        _atomic_json(repair_run / "job_spec.json", job)
        jobs.append(job)
    jobs_path = cycle / "05_reanalysis" / "jobs.jsonl"
    _atomic_jsonl(jobs_path, jobs)
    manifest = {
        "schema_version": "review-remediation-execution-v1",
        "cycle_id": cycle.name,
        "status": "prepared_waiting_for_model_execution",
        "job_count": len(jobs),
        "approved_trace_count": len({
            (value["batch_id"], value["trace_id"]) for value in approved
        }),
        "jobs_path": str(jobs_path),
        "model_calls_started": False,
        "publication_allowed": False,
        "created_at": _now(),
    }
    _atomic_json(cycle / "05_reanalysis" / "manifest.json", manifest)
    return manifest


def execute_prepared_reanalysis(
    *,
    cycle_dir: Path,
    provider: str,
    provider_config: Path,
    cache_dir: Path,
    model: str | None = None,
    workers: int = 4,
    direct_max_chars: int = 120_000,
    request_max_chars: int = 800_000,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    thinking_type: str | None = "disabled",
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Execute only prepared jobs; Stage 03 resumes safely on repeated calls."""
    # Lazy imports keep plan/approve/prepare usable in read-only environments
    # where the model client dependencies are intentionally not installed.
    from ..model_api import OpenAICompatibleClient, load_provider
    from ..pipeline.stage_03_agent_verify import run_agent_verify
    from ..pipeline.stage_04_build_results import build_pain_cases

    cycle = cycle_dir.expanduser().resolve()
    jobs_path = cycle / "05_reanalysis" / "jobs.jsonl"
    jobs = list(_read_jsonl(jobs_path))
    if not jobs:
        raise ValueError("No prepared jobs")
    config = load_provider(provider_config.expanduser().resolve(), provider)
    client = OpenAICompatibleClient(
        config,
        model=model or config.default_model,
        cache_dir=cache_dir.expanduser().resolve(),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        thinking_type=thinking_type,
        max_completion_tokens=max_completion_tokens,
    )
    results = []
    for job in jobs:
        if job.get("status") not in {
            "prepared_waiting_for_model_execution", "partial",
        }:
            continue
        run_dir = Path(str(job["repair_run_dir"])).resolve()
        batch_dir = Path(str(job["batch_dir"])).resolve()
        verification = run_agent_verify(
            batch_dir,
            run_dir,
            client,
            provider,
            workers=workers,
            direct_max_chars=direct_max_chars,
            request_max_chars=request_max_chars,
            include_trace_ids=set(str(value) for value in job.get("trace_ids") or []),
        )
        status = str(verification.get("status") or "partial")
        built = None
        if status == "completed":
            built = build_pain_cases(run_dir, batch_dir)
        job["status"] = status
        job["executed_prompt_versions"] = {
            "direct": "trace-agent-verification-v4",
            "indexed": "trace-agent-verification-indexed-v6",
        }
        job["verification_manifest"] = verification
        job["result_manifest"] = built
        job["updated_at"] = _now()
        _atomic_json(run_dir / "job_spec.json", job)
        results.append({
            "job_id": job["job_id"], "batch_id": job["batch_id"],
            "status": status,
            "completed_trace_count": verification.get("completed_trace_count"),
            "pending_trace_count": verification.get("pending_trace_count"),
            "failed_trace_count": verification.get("failed_trace_count"),
        })
    _atomic_jsonl(jobs_path, jobs)
    overall = (
        "completed"
        if jobs and all(value.get("status") == "completed" for value in jobs)
        else "partial"
    )
    manifest = {
        "schema_version": "review-remediation-execution-v1",
        "cycle_id": cycle.name,
        "status": overall,
        "provider": provider,
        "model": client.model,
        "job_results": results,
        "publication_allowed": False,
        "updated_at": _now(),
    }
    _atomic_json(cycle / "05_reanalysis" / "manifest.json", manifest)
    return manifest


def _case_comparison(value: dict[str, Any]) -> dict[str, Any]:
    verification = value.get("agent_verification") or {}
    return {
        "case_id": value.get("case_id"),
        "episode_id": value.get("episode_id"),
        "pain_judgment": value.get("pain_judgment"),
        "judgment_reason_codes": value.get("judgment_reason_codes") or [],
        "outcome": value.get("outcome"),
        "agent_related": verification.get("agent_related"),
        "agent_failure": verification.get("agent_failure"),
        "agent_related_reason": verification.get("agent_related_reason"),
        "user_interruptions": verification.get("user_interruptions") or [],
        "capability_gaps": value.get("capability_gaps") or [],
        "validation_warnings": value.get("validation_warnings") or [],
    }


def build_validated_candidate_runs(
    *, cycle_dir: Path, analysis_root: Path
) -> dict[str, Any]:
    """Merge completed repairs with untouched baselines, but do not publish."""
    cycle = cycle_dir.expanduser().resolve()
    analysis = analysis_root.expanduser().resolve()
    jobs_path = cycle / "05_reanalysis" / "jobs.jsonl"
    jobs = list(_read_jsonl(jobs_path))
    if not jobs:
        raise ValueError("No prepared reanalysis jobs")
    all_diffs = []
    candidate_manifests = []
    for job in jobs:
        batch_id = str(job["batch_id"])
        baseline_run_id = str(job["baseline_analysis_run_id"])
        repair_run = Path(str(job["repair_run_dir"])).resolve()
        repair_run_id = str(job["repair_analysis_run_id"])
        target_trace_ids = set(str(value) for value in job.get("trace_ids") or [])
        baseline_path = _result_path(analysis, batch_id, baseline_run_id)
        repair_path = _result_path(repair_run.parents[1], batch_id, repair_run_id)
        # A repair run lives below <cycle>/05_reanalysis/<batch>/<run>.  The
        # generic resolver above receives <cycle>/05_reanalysis as its root.
        baseline_cases = list(_read_jsonl(baseline_path))
        repair_cases = list(_read_jsonl(repair_path))
        repair_trace_ids = {
            str(value.get("trace_id") or "") for value in repair_cases
        }
        missing = sorted(target_trace_ids - repair_trace_ids)
        unexpected = sorted(repair_trace_ids - target_trace_ids)
        if missing or unexpected:
            raise ValueError(
                f"Repair coverage mismatch for {batch_id}: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        digest = hashlib.sha256(
            f"{cycle.name}|{batch_id}|{baseline_run_id}|{repair_run_id}".encode()
        ).hexdigest()[:8]
        candidate_run_id = f"candidate_{cycle.name}_{digest}"
        candidate_run = (
            cycle / "06_diff_validation" / "candidate-runs"
            / batch_id / candidate_run_id
        )
        output_dir = candidate_run / "04_results"
        output_path = output_dir / f"pain_cases__{batch_id}__{candidate_run_id}.jsonl"
        combined = []
        for source in baseline_cases:
            if str(source.get("trace_id") or "") in target_trace_ids:
                continue
            value = dict(source)
            value["analysis_run_id"] = candidate_run_id
            lineage = dict(value.get("lineage") or {})
            lineage.update({
                "copied_unchanged_from_analysis_run_id": baseline_run_id,
                "review_cycle_id": cycle.name,
            })
            value["lineage"] = lineage
            combined.append(value)
        for source in repair_cases:
            value = dict(source)
            value["analysis_run_id"] = candidate_run_id
            lineage = dict(value.get("lineage") or {})
            lineage.update({
                "repaired_from_analysis_run_id": baseline_run_id,
                "repair_analysis_run_id": repair_run_id,
                "review_cycle_id": cycle.name,
                "trigger_queue_ids": job.get("queue_ids") or [],
            })
            value["lineage"] = lineage
            combined.append(value)
        combined.sort(key=lambda value: (
            str(value.get("trace_id") or ""),
            str(value.get("episode_id") or ""),
        ))
        case_ids = [str(value.get("case_id") or "") for value in combined]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Duplicate case_id in candidate Run: {batch_id}")
        _atomic_jsonl(output_path, combined)
        _atomic_json(output_dir / "manifest.json", {
            "manifest_version": "1.0",
            "stage": "04_results",
            "status": "candidate_needs_human_acceptance",
            "batch_id": batch_id,
            "analysis_run_id": candidate_run_id,
            "supersedes_analysis_run_id": baseline_run_id,
            "repair_analysis_run_id": repair_run_id,
            "review_cycle_id": cycle.name,
            "output_path": str(output_path),
            "case_count": len(combined),
            "repaired_trace_count": len(target_trace_ids),
            "generated_at": _now(),
        })
        before_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        after_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in baseline_cases:
            if str(value.get("trace_id") or "") in target_trace_ids:
                before_by_trace[str(value["trace_id"])].append(_case_comparison(value))
        for value in repair_cases:
            after_by_trace[str(value["trace_id"])].append(_case_comparison(value))
        for trace_id in sorted(target_trace_ids):
            before = sorted(before_by_trace[trace_id], key=lambda value: str(value["case_id"]))
            after = sorted(after_by_trace[trace_id], key=lambda value: str(value["case_id"]))
            all_diffs.append({
                "schema_version": "review-remediation-case-diff-v1",
                "cycle_id": cycle.name,
                "batch_id": batch_id,
                "baseline_analysis_run_id": baseline_run_id,
                "repair_analysis_run_id": repair_run_id,
                "candidate_analysis_run_id": candidate_run_id,
                "trace_id": trace_id,
                "changed": before != after,
                "before": before,
                "after": after,
                "acceptance_status": "pending_human_review",
            })
        candidate_manifests.append({
            "batch_id": batch_id,
            "candidate_analysis_run_id": candidate_run_id,
            "candidate_run_dir": str(candidate_run),
            "case_count": len(combined),
            "repaired_trace_count": len(target_trace_ids),
            "structural_validation": "passed",
            "publication_allowed": False,
        })
    diff_path = cycle / "06_diff_validation" / "case_diffs.jsonl"
    _atomic_jsonl(diff_path, all_diffs)
    summary = {
        "schema_version": "review-remediation-validation-v1",
        "cycle_id": cycle.name,
        "status": "needs_human_acceptance",
        "candidate_runs": candidate_manifests,
        "diff_trace_count": len(all_diffs),
        "changed_trace_count": sum(1 for value in all_diffs if value["changed"]),
        "structural_validation": "passed",
        "publication_allowed": False,
        "diff_path": str(diff_path),
        "created_at": _now(),
    }
    _atomic_json(cycle / "06_diff_validation" / "summary.json", summary)
    return summary


__all__ = [
    "approve_queue_items", "build_validated_candidate_runs",
    "execute_prepared_reanalysis", "prepare_approved_reanalysis",
]
