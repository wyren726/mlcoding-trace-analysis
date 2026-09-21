"""Repair unresolved Trace analyses without rewriting successful records."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from ...registries import registry_root_for_output, write_immutable_record
from .compact import (
    BOUNDARY_REPAIR_PROMPT_VERSION,
    BOUNDARY_REPAIR_SYSTEM_PROMPT,
    COMPACT_USER_PROMPT_VERSION,
    COMPACT_USER_SYSTEM_PROMPT,
    INDEX_SELECTION_PROMPT_VERSION,
    INDEX_SELECTION_SYSTEM_PROMPT,
    INDEXED_ATTRIBUTION_PROMPT_VERSION,
    INDEXED_ATTRIBUTION_SYSTEM_PROMPT,
)
from .prompt import AGENT_ATTRIBUTION_PROMPT_VERSION, AGENT_ATTRIBUTION_SYSTEM_PROMPT
from .runner import (
    JSONClient,
    TraceBundle,
    TraceResult,
    UserScreenResult,
    analyze_user_trace,
    iter_trace_bundles,
    verify_screened_trace,
)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict) and value.get("trace_id"):
                records[str(value["trace_id"])] = value
    return records


def _upsert_record(path: Path, record: dict[str, Any]) -> None:
    """Atomically replace one Trace row while preserving one-row-per-Trace."""
    trace_id = str(record.get("trace_id") or "")
    rows = _read_records(path)
    rows[trace_id] = record
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows.values():
            target.write(_json_line(row))
    temporary.replace(path)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _prompt_snapshot(repair_id: str) -> dict[str, Any]:
    prompts = {
        "user_screen": {
            "version": COMPACT_USER_PROMPT_VERSION,
            "system_prompt": COMPACT_USER_SYSTEM_PROMPT,
        },
        "user_screen_boundary_repair": {
            "version": BOUNDARY_REPAIR_PROMPT_VERSION,
            "system_prompt": BOUNDARY_REPAIR_SYSTEM_PROMPT,
        },
        "direct_agent_verification": {
            "version": AGENT_ATTRIBUTION_PROMPT_VERSION,
            "system_prompt": AGENT_ATTRIBUTION_SYSTEM_PROMPT,
        },
        "evidence_selection": {
            "version": INDEX_SELECTION_PROMPT_VERSION,
            "system_prompt": INDEX_SELECTION_SYSTEM_PROMPT,
        },
        "indexed_agent_verification": {
            "version": INDEXED_ATTRIBUTION_PROMPT_VERSION,
            "system_prompt": INDEXED_ATTRIBUTION_SYSTEM_PROMPT,
        },
    }
    for prompt in prompts.values():
        prompt["sha256"] = _sha256(str(prompt["system_prompt"]))
    return {
        "repair_run_id": repair_id,
        "created_at": _now(),
        "prompts": prompts,
    }


def _append_history(manifest: dict[str, Any], key: str, values: Iterable[str]) -> None:
    history = [str(item) for item in manifest.get(key) or []]
    for value in values:
        if value and value not in history:
            history.append(value)
    manifest[key] = history


def _route_is_analyze(record: dict[str, Any]) -> bool:
    return any(
        isinstance(episode, dict)
        and isinstance(episode.get("route"), dict)
        and episode["route"].get("judgment") == "analyze"
        for episode in record.get("task_episodes") or []
    )


def _register_completed_analysis(
    run: Path, stage: Path, manifest: dict[str, Any],
    screen_output: Path, output: Path, errors: Path, progress: Path,
    manifest_path: Path, record_count: int,
) -> Path:
    run_id = str(manifest["analysis_run_id"])
    registry_id = f"{run_id}__02_analyze__v3_1"
    prompt_snapshots = sorted((stage / "repairs").glob("*/prompt_snapshot.json"))
    return write_immutable_record(
        "analysis-runs", registry_id,
        {
            "analysis_run_id": registry_id,
            "product_id": "trace_analysis",
            "product_version": "3.1",
            "input_batches": [str(manifest["batch_id"])],
            "output_path": str(stage),
            "status": "completed",
            "record_count": record_count,
            "error_count": 0,
            "prompt_versions": manifest.get("prompt_version_history") or [],
            "source_analysis_run_id": run_id,
        },
        artifacts=(
            screen_output, output, errors, progress, manifest_path,
            *prompt_snapshots,
        ),
        root=registry_root_for_output(run),
    )


def repair_trace_analysis(
    batch_dir: Path,
    run_dir: Path,
    client: JSONClient,
    provider_name: str,
    *,
    include_trace_ids: set[str] | None = None,
    workers: int = 2,
    direct_max_chars: int = 120_000,
    request_max_chars: int = 800_000,
    export_after: bool = True,
) -> dict[str, Any]:
    """Append only missing records, keeping every earlier successful row unchanged.

    User screening and Agent verification are scheduled as separate phases.  A
    successful compact screen is therefore durable even if its later Agent call
    times out, and another repair can resume from that JSONL row without paying
    for the first call again.
    """
    batch = batch_dir.expanduser().resolve()
    run = run_dir.expanduser().resolve()
    stage = run / "02_analyze"
    manifest_path = stage / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_id = str(manifest.get("batch_id") or "")
    run_id = str(manifest.get("analysis_run_id") or "")
    if batch.name != batch_id:
        raise ValueError("Analysis run and preprocessing batch do not match")
    if not run_id:
        raise ValueError("Analysis manifest has no analysis_run_id")
    turn_path = batch / "unified_turns.jsonl"
    if not turn_path.is_file():
        raise FileNotFoundError(turn_path)

    output = stage / f"trace_analysis__{batch_id}__{run_id}.jsonl"
    screen_output = stage / f"user_screen__{batch_id}__{run_id}.jsonl"
    main_errors = stage / "errors.jsonl"
    main_progress = stage / "progress.jsonl"
    completed = _read_records(output)
    screens = _read_records(screen_output)
    bundles = {bundle.trace_id: bundle for bundle in iter_trace_bundles(turn_path)}
    source_ids = set(bundles)
    requested = include_trace_ids if include_trace_ids is not None else source_ids
    unknown = requested - source_ids
    if unknown:
        raise ValueError(f"Unknown trace IDs: {sorted(unknown)[:5]}")
    target_ids = requested - set(completed)
    if not target_ids:
        registry_record = None
        if len(completed) == len(source_ids):
            registry_record = _register_completed_analysis(
                run, stage, manifest, screen_output, output, main_errors,
                main_progress, manifest_path, len(completed),
            )
        return {
            "status": "nothing_to_repair",
            "batch_id": batch_id,
            "analysis_run_id": run_id,
            "completed_trace_count": len(completed),
            "pending_trace_count": len(source_ids - set(completed)),
            "registry_record_path": (
                str(registry_record.resolve()) if registry_record else None
            ),
        }

    now = dt.datetime.now().astimezone()
    suffix = _sha256("|".join(sorted(target_ids)) + now.isoformat())[:8]
    repair_id = f"repair_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"
    repair_dir = stage / "repairs" / repair_id
    repair_dir.mkdir(parents=True, exist_ok=False)
    repair_screen = repair_dir / "user_screen.jsonl"
    repair_output = repair_dir / "trace_analysis.jsonl"
    repair_errors = repair_dir / "errors.jsonl"
    repair_progress = repair_dir / "progress.jsonl"
    prompt_snapshot = repair_dir / "prompt_snapshot.json"
    repair_manifest_path = repair_dir / "manifest.json"
    _atomic_json(prompt_snapshot, _prompt_snapshot(repair_id))

    repair_manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "repair_run_id": repair_id,
        "analysis_run_id": run_id,
        "batch_id": batch_id,
        "status": "running",
        "provider": provider_name,
        "model": client.model,
        "thinking_type": getattr(client, "thinking_type", None),
        "max_completion_tokens": getattr(client, "max_completion_tokens", None),
        "target_trace_ids": sorted(target_ids),
        "target_trace_count": len(target_ids),
        "completed_trace_count": 0,
        "failed_trace_count": 0,
        "prompt_snapshot_path": str(prompt_snapshot),
        "output_path": str(repair_output),
        "updated_at": _now(),
    }
    _atomic_json(repair_manifest_path, repair_manifest)

    _append_history(
        manifest, "prompt_version_history",
        [str(manifest.get("prompt_version") or ""), COMPACT_USER_PROMPT_VERSION],
    )
    _append_history(
        manifest, "attribution_prompt_version_history",
        [
            str(manifest.get("attribution_prompt_version") or ""),
            AGENT_ATTRIBUTION_PROMPT_VERSION,
            INDEXED_ATTRIBUTION_PROMPT_VERSION,
        ],
    )
    _append_history(
        manifest, "evidence_selection_prompt_version_history",
        [INDEX_SELECTION_PROMPT_VERSION],
    )
    repair_runs = list(manifest.get("repair_runs") or [])
    repair_runs.append({
        "repair_run_id": repair_id,
        "status": "running",
        "provider": provider_name,
        "model": client.model,
        "target_trace_count": len(target_ids),
        "path": str(repair_dir),
        "started_at": _now(),
    })
    manifest.update({
        "status": "running",
        "mixed_prompt_versions": True,
        "repair_runs": repair_runs,
        "updated_at": _now(),
    })
    _atomic_json(manifest_path, manifest)

    usage: Counter[str] = Counter()
    model_calls = 0
    failures: dict[str, dict[str, Any]] = {}
    screen_results: dict[str, UserScreenResult] = {}
    rescreen_ids: set[str] = set()

    def progress_row(phase: str) -> dict[str, Any]:
        current_completed = len(_read_records(output))
        return {
            "timestamp": _now(),
            "batch_id": batch_id,
            "analysis_run_id": run_id,
            "repair_run_id": repair_id,
            "stage": "02_analyze_repair",
            "phase": phase,
            "target_trace_count": len(target_ids),
            "repair_completed_trace_count": sum(
                trace_id in _read_records(repair_output) for trace_id in target_ids
            ),
            "repair_failed_trace_count": len(failures),
            "total_completed_trace_count": current_completed,
            "total_pending_trace_count": len(source_ids) - current_completed,
        }

    try:
        with repair_screen.open("a", encoding="utf-8") as repair_screen_out, \
                main_errors.open("a", encoding="utf-8") as main_error_out, \
                repair_errors.open("a", encoding="utf-8") as repair_error_out, \
                repair_progress.open("a", encoding="utf-8") as repair_progress_out, \
                main_progress.open("a", encoding="utf-8") as main_progress_out:
            for trace_id in target_ids:
                existing = screens.get(trace_id)
                if existing is not None:
                    provenance = existing.setdefault("analysis_provenance", {})
                    if provenance.get("user_prompt_version") != COMPACT_USER_PROMPT_VERSION:
                        rescreen_ids.add(trace_id)
                        continue
                    origin_repair_id = str(provenance.get("repair_run_id") or "")
                    provenance.setdefault("user_screen_repair_run_id", origin_repair_id or None)
                    if origin_repair_id:
                        origin_manifest_path = (
                            stage / "repairs" / origin_repair_id / "manifest.json"
                        )
                        if origin_manifest_path.is_file():
                            origin_manifest = json.loads(
                                origin_manifest_path.read_text(encoding="utf-8")
                            )
                            provenance.setdefault(
                                "user_screen_provider", origin_manifest.get("provider")
                            )
                            provenance.setdefault(
                                "user_screen_model", origin_manifest.get("model")
                            )
                    screen_results[trace_id] = UserScreenResult(
                        existing, "reused_existing_user_screen", 0, {}
                    )

            missing_screens = sorted(target_ids - set(screen_results))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                future_screens: dict[Future[UserScreenResult], str] = {
                    executor.submit(
                        analyze_user_trace, bundles[trace_id], client, run_id,
                        request_max_chars,
                    ): trace_id
                    for trace_id in missing_screens
                }
                for future in as_completed(future_screens):
                    trace_id = future_screens[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        error = {
                            "timestamp": _now(), "batch_id": batch_id,
                            "analysis_run_id": run_id, "repair_run_id": repair_id,
                            "trace_id": trace_id, "phase": "user_screen",
                            "error_type": type(exc).__name__, "error": str(exc),
                        }
                        failures[trace_id] = error
                        main_error_out.write(_json_line(error)); main_error_out.flush()
                        repair_error_out.write(_json_line(error)); repair_error_out.flush()
                    else:
                        provenance = result.record.setdefault("analysis_provenance", {})
                        provenance.update({
                            "repair_run_id": repair_id,
                            "user_screen_repair_run_id": repair_id,
                            "user_screen_provider": provider_name,
                            "user_screen_model": client.model,
                        })
                        row = _json_line(result.record)
                        if trace_id in rescreen_ids:
                            _upsert_record(screen_output, result.record)
                        else:
                            with screen_output.open("a", encoding="utf-8") as target:
                                target.write(row)
                                target.flush()
                        repair_screen_out.write(row); repair_screen_out.flush()
                        screens[trace_id] = result.record
                        screen_results[trace_id] = result
                        usage.update(result.usage)
                        model_calls += result.model_call_count
                    progress = progress_row("user_screen")
                    repair_progress_out.write(_json_line(progress)); repair_progress_out.flush()
                    main_progress_out.write(_json_line(progress)); main_progress_out.flush()

            final_ready: dict[str, TraceResult] = {}
            to_verify: list[str] = []
            for trace_id, screen in screen_results.items():
                if trace_id in failures:
                    continue
                if _route_is_analyze(screen.record):
                    to_verify.append(trace_id)
                else:
                    final_ready[trace_id] = TraceResult(
                        screen.record, screen.record, "compact_user_screen_only",
                        screen.model_call_count, screen.usage,
                    )

            with output.open("a", encoding="utf-8") as main_output_out, \
                    repair_output.open("a", encoding="utf-8") as repair_output_out:
                def write_final(trace_id: str, result: TraceResult) -> None:
                    nonlocal model_calls
                    provenance = result.record.setdefault("analysis_provenance", {})
                    provenance.update({
                        "last_repair_run_id": repair_id,
                        "agent_verification_repair_run_id": (
                            repair_id if _route_is_analyze(result.record) else None
                        ),
                        "agent_verification_provider": (
                            provider_name if _route_is_analyze(result.record) else None
                        ),
                        "agent_verification_model": (
                            client.model if _route_is_analyze(result.record) else None
                        ),
                    })
                    row = _json_line(result.record)
                    main_output_out.write(row); main_output_out.flush()
                    repair_output_out.write(row); repair_output_out.flush()
                    completed[trace_id] = result.record
                    screen_usage = Counter(screen_results[trace_id].usage)
                    usage.update(Counter(result.usage) - screen_usage)
                    model_calls += (
                        result.model_call_count
                        - screen_results[trace_id].model_call_count
                    )

                for trace_id, result in final_ready.items():
                    write_final(trace_id, result)
                    progress = progress_row("finalize_screen_only")
                    repair_progress_out.write(_json_line(progress)); repair_progress_out.flush()
                    main_progress_out.write(_json_line(progress)); main_progress_out.flush()

                with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                    future_verifications: dict[Future[TraceResult], str] = {
                        executor.submit(
                            verify_screened_trace, bundles[trace_id], client,
                            screen_results[trace_id], direct_max_chars,
                            request_max_chars,
                        ): trace_id
                        for trace_id in sorted(to_verify)
                    }
                    for future in as_completed(future_verifications):
                        trace_id = future_verifications[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            error = {
                                "timestamp": _now(), "batch_id": batch_id,
                                "analysis_run_id": run_id, "repair_run_id": repair_id,
                                "trace_id": trace_id, "phase": "agent_verification",
                                "error_type": type(exc).__name__, "error": str(exc),
                            }
                            failures[trace_id] = error
                            main_error_out.write(_json_line(error)); main_error_out.flush()
                            repair_error_out.write(_json_line(error)); repair_error_out.flush()
                        else:
                            write_final(trace_id, result)
                        progress = progress_row("agent_verification")
                        repair_progress_out.write(_json_line(progress)); repair_progress_out.flush()
                        main_progress_out.write(_json_line(progress)); main_progress_out.flush()
    except BaseException:
        repair_manifest.update({"status": "interrupted", "updated_at": _now()})
        _atomic_json(repair_manifest_path, repair_manifest)
        manifest.update({"status": "interrupted", "updated_at": _now()})
        repair_runs[-1].update({"status": "interrupted", "updated_at": _now()})
        _atomic_json(manifest_path, manifest)
        raise

    unresolved = source_ids - set(completed)
    status = "completed" if not unresolved else "partial"
    repair_completed = len(target_ids & set(completed))
    repair_manifest.update({
        "status": status,
        "completed_trace_count": repair_completed,
        "failed_trace_count": len(target_ids - set(completed)),
        "model_call_count": model_calls,
        "usage": dict(usage),
        "updated_at": _now(),
    })
    _atomic_json(repair_manifest_path, repair_manifest)
    repair_runs[-1].update({
        "status": status,
        "completed_trace_count": repair_completed,
        "failed_trace_count": len(target_ids - set(completed)),
        "completed_at": _now(),
    })
    manifest.update({
        "status": status,
        "completed_trace_count": len(completed),
        "failed_trace_count": len(unresolved),
        "pending_trace_count": len(unresolved),
        "repair_runs": repair_runs,
        "updated_at": _now(),
    })
    _atomic_json(manifest_path, manifest)

    registry_record = None
    if status == "completed":
        registry_record = _register_completed_analysis(
            run, stage, manifest, screen_output, output, main_errors,
            main_progress, manifest_path, len(completed),
        )
    export_result = None
    if export_after:
        from ..stage_03_export import export_trace_analysis

        export_result = export_trace_analysis(run, batch)
    return {
        **repair_manifest,
        "run_dir": str(run),
        "repair_dir": str(repair_dir),
        "total_completed_trace_count": len(completed),
        "total_pending_trace_count": len(unresolved),
        "registry_record_path": (
            str(registry_record.resolve()) if registry_record else None
        ),
        "export": export_result,
    }


__all__ = ["repair_trace_analysis"]
