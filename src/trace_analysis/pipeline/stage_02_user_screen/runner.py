from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .._compat import (
    ACTIVE_USER_PROMPT_VERSION,
    JSONClient,
    TraceBundle,
    UserScreenResult,
    analyze_user_trace,
    iter_trace_bundles,
)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values: set[str] = set()
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                trace_id = json.loads(line).get("trace_id")
            except (json.JSONDecodeError, AttributeError):
                continue
            if trace_id:
                values.add(str(trace_id))
    return values


def _new_run_id(batch_id: str, model: str) -> str:
    now = dt.datetime.now().astimezone()
    suffix = hashlib.sha256(
        f"{batch_id}|{ACTIVE_USER_PROMPT_VERSION}|{model}|{now.isoformat()}".encode()
    ).hexdigest()[:8]
    return f"run_{now:%Y%m%d_%H%M%S}_{suffix}"


def run_user_screen(
    batch_dir: Path,
    analysis_root: Path,
    client: JSONClient,
    provider_name: str,
    *,
    workers: int = 4,
    request_max_chars: int = 800_000,
    run_id: str | None = None,
    include_trace_ids: set[str] | None = None,
    force_hierarchical_trace_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run and persist only the complete-user-turn screen for every Trace.

    This stage never reads Agent messages into the model prompt and never performs
    Agent verification.  Its output is therefore independently resumable and is
    the sole routing input to Stage 03.
    """
    batch = batch_dir.expanduser().resolve()
    turn_path = batch / "unified_turns.jsonl"
    if not turn_path.is_file():
        raise FileNotFoundError(turn_path)
    batch_id = batch.name
    user_path = batch / f"user_turns__{batch_id}.jsonl"
    if user_path.is_file():
        # Lazy import avoids a module cycle: Stage 01's extractor reuses the
        # established real-user-message normalization exposed by the temporary
        # compatibility facade.
        from ..stage_01_preprocess.user_turns import iter_user_trace_bundles
        bundle_iterator = lambda: iter_user_trace_bundles(user_path)
        input_path = user_path
        input_product = "user_turns"
    else:
        bundle_iterator = lambda: iter_trace_bundles(turn_path)
        input_path = turn_path
        input_product = "unified_turns_compatibility_fallback"
    effective_run_id = run_id or _new_run_id(batch_id, client.model)
    run_dir = analysis_root.expanduser().resolve() / batch_id / effective_run_id
    stage_dir = run_dir / "02_user_screen"
    stage_dir.mkdir(parents=True, exist_ok=True)
    output = stage_dir / f"user_screen__{batch_id}__{effective_run_id}.jsonl"
    errors = stage_dir / "errors.jsonl"
    progress = stage_dir / "progress.jsonl"
    manifest_path = stage_dir / "manifest.json"

    completed_ids = _completed_ids(output)
    all_trace_ids = {bundle.trace_id for bundle in bundle_iterator()}
    target_ids = all_trace_ids if include_trace_ids is None else all_trace_ids & include_trace_ids
    unknown = (include_trace_ids or set()) - all_trace_ids
    if unknown:
        raise ValueError(f"Unknown trace IDs: {sorted(unknown)[:5]}")
    pending_ids = target_ids - completed_ids
    counts: Counter[str] = Counter()
    if output.is_file():
        with output.open(encoding="utf-8-sig") as existing:
            for line in existing:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    route = (value.get("route") or {}).get("judgment") or "unknown"
                except (json.JSONDecodeError, AttributeError):
                    continue
                counts[str(route)] += 1
    failed = 0
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "stage": "02_user_screen",
        "status": "running",
        "batch_id": batch_id,
        "analysis_run_id": effective_run_id,
        "provider": provider_name,
        "model": client.model,
        "prompt_version": ACTIVE_USER_PROMPT_VERSION,
        "input_product": input_product,
        "input_path": str(input_path),
        "total_trace_count": len(target_ids),
        "completed_trace_count": len(completed_ids & target_ids),
        "pending_trace_count": len(pending_ids),
        "failed_trace_count": 0,
        "route_counts": {},
        "output_path": str(output),
        "errors_path": str(errors),
        "updated_at": _now(),
    }
    _atomic_json(manifest_path, manifest)

    def bundles():
        for bundle in bundle_iterator():
            if bundle.trace_id in pending_ids:
                yield bundle

    iterator = iter(bundles())
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    pending: dict[Future[UserScreenResult], TraceBundle] = {}

    def submit() -> bool:
        try:
            bundle = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(
            analyze_user_trace, bundle, client, effective_run_id,
            request_max_chars,
            bundle.trace_id in (force_hierarchical_trace_ids or set()),
        )] = bundle
        return True

    for _ in range(max(1, workers) * 2):
        if not submit():
            break
    try:
        with output.open("a", encoding="utf-8") as out, \
                errors.open("a", encoding="utf-8") as error_out, \
                progress.open("a", encoding="utf-8") as progress_out:
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    bundle = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed += 1
                        error_out.write(_json_line({
                            "timestamp": _now(), "batch_id": batch_id,
                            "analysis_run_id": effective_run_id,
                            "trace_id": bundle.trace_id,
                            "error_type": type(exc).__name__, "error": str(exc),
                        }))
                        error_out.flush()
                    else:
                        out.write(_json_line(result.record))
                        out.flush()
                        completed_ids.add(bundle.trace_id)
                        route = (result.record.get("route") or {}).get("judgment") or "unknown"
                        counts[str(route)] += 1
                    submit()
                    progress_out.write(_json_line({
                        "timestamp": _now(), "stage": "02_user_screen",
                        "total": len(target_ids),
                        "completed": len(completed_ids & target_ids),
                        "failed": failed,
                        "pending": max(0, len(target_ids) - len(completed_ids & target_ids)),
                    }))
                    progress_out.flush()
                    manifest.update({
                        "completed_trace_count": len(completed_ids & target_ids),
                        "pending_trace_count": max(
                            0, len(target_ids) - len(completed_ids & target_ids)
                        ),
                        "failed_trace_count": failed,
                        "route_counts": dict(counts),
                        "updated_at": _now(),
                    })
                    _atomic_json(manifest_path, manifest)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        manifest.update({"status": "interrupted", "updated_at": _now()})
        _atomic_json(manifest_path, manifest)
        raise
    else:
        executor.shutdown(wait=True)

    status = "completed" if len(completed_ids & target_ids) == len(target_ids) else "partial"
    manifest.update({"status": status, "updated_at": _now()})
    _atomic_json(manifest_path, manifest)
    return {**manifest, "run_dir": str(run_dir), "stage_dir": str(stage_dir)}
