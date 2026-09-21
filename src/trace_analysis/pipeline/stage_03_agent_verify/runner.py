from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .._compat import (
    AGENT_ATTRIBUTION_PROMPT_VERSION,
    INDEX_SELECTION_PROMPT_VERSION,
    JSONClient,
    TraceBundle,
    TraceResult,
    UserScreenResult,
    iter_trace_bundles,
    verify_screened_trace,
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


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not an object: {path}")
            yield value


def _completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row["trace_id"]) for row in _read_jsonl(path) if row.get("trace_id")
    }


def _find_screen(run_dir: Path, batch_id: str, run_id: str) -> Path:
    candidates = [
        run_dir / "02_user_screen" / f"user_screen__{batch_id}__{run_id}.jsonl",
        run_dir / "02_analyze" / f"user_screen__{batch_id}__{run_id}.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(run_dir.glob("02_*/user_screen__*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(candidates[0])


def run_agent_verify(
    batch_dir: Path,
    run_dir: Path,
    client: JSONClient,
    provider_name: str,
    *,
    workers: int = 4,
    direct_max_chars: int = 120_000,
    request_max_chars: int = 800_000,
    include_trace_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Verify only Trace records explicitly routed to ``analyze`` by Stage 02."""
    batch = batch_dir.expanduser().resolve()
    run = run_dir.expanduser().resolve()
    batch_id = batch.name
    run_id = run.name
    turn_path = batch / "unified_turns.jsonl"
    if not turn_path.is_file():
        raise FileNotFoundError(turn_path)
    screen_path = _find_screen(run, batch_id, run_id)
    selected = {
        str(row["trace_id"]): row
        for row in _read_jsonl(screen_path)
        if row.get("trace_id")
        and (row.get("route") or {}).get("judgment") == "analyze"
    }
    if include_trace_ids is not None:
        unknown = include_trace_ids - set(selected)
        if unknown:
            raise ValueError(
                f"Requested Trace IDs are not Stage 02 analyze records: {sorted(unknown)[:5]}"
            )
        selected = {
            trace_id: row for trace_id, row in selected.items()
            if trace_id in include_trace_ids
        }

    stage_dir = run / "03_agent_verify"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # A resume may be started while an older worker is still finishing a slow
    # request.  Serialise writers for one run so the same trace cannot be
    # appended twice by overlapping processes.
    run_lock = (stage_dir / ".run.lock").open("w")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        run_lock.close()
        raise RuntimeError(
            f"Stage 03 is already running for analysis run: {run_id}"
        ) from exc
    output = stage_dir / f"trace_analysis__{batch_id}__{run_id}.jsonl"
    errors = stage_dir / "errors.jsonl"
    progress = stage_dir / "progress.jsonl"
    manifest_path = stage_dir / "manifest.json"
    completed_ids = _completed_ids(output)
    pending_ids = set(selected) - completed_ids
    failed = 0
    strategies: Counter[str] = Counter()
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "stage": "03_agent_verify",
        "status": "running",
        "batch_id": batch_id,
        "analysis_run_id": run_id,
        "provider": provider_name,
        "model": client.model,
        "agent_prompt_version": AGENT_ATTRIBUTION_PROMPT_VERSION,
        "evidence_selection_prompt_version": INDEX_SELECTION_PROMPT_VERSION,
        "input_path": str(screen_path),
        "selected_trace_count": len(selected),
        "completed_trace_count": len(completed_ids & set(selected)),
        "pending_trace_count": len(pending_ids),
        "failed_trace_count": 0,
        "strategy_counts": {},
        "output_path": str(output),
        "errors_path": str(errors),
        "updated_at": _now(),
    }
    _atomic_json(manifest_path, manifest)

    def bundles():
        found: set[str] = set()
        for bundle in iter_trace_bundles(turn_path):
            if bundle.trace_id in pending_ids:
                found.add(bundle.trace_id)
                yield bundle
        missing = pending_ids - found
        if missing:
            raise ValueError(f"Stage 02 Trace IDs missing from batch: {sorted(missing)[:5]}")

    iterator = iter(bundles())
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    pending: dict[Future[TraceResult], TraceBundle] = {}

    def submit() -> bool:
        try:
            bundle = next(iterator)
        except StopIteration:
            return False
        screen = UserScreenResult(selected[bundle.trace_id], "persisted_user_screen", 0, {})
        pending[executor.submit(
            verify_screened_trace,
            bundle,
            client,
            screen,
            direct_max_chars,
            request_max_chars,
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
                            "analysis_run_id": run_id, "trace_id": bundle.trace_id,
                            "error_type": type(exc).__name__, "error": str(exc),
                        }))
                        error_out.flush()
                    else:
                        # This file intentionally has no skip/review rows.
                        out.write(_json_line(result.record))
                        out.flush()
                        completed_ids.add(bundle.trace_id)
                        strategies[result.strategy] += 1
                    submit()
                    progress_out.write(_json_line({
                        "timestamp": _now(), "stage": "03_agent_verify",
                        "selected": len(selected),
                        "completed": len(completed_ids & set(selected)),
                        "failed": failed,
                        "pending": max(0, len(selected) - len(completed_ids & set(selected))),
                    }))
                    progress_out.flush()
                    manifest.update({
                        "completed_trace_count": len(completed_ids & set(selected)),
                        "pending_trace_count": max(
                            0, len(selected) - len(completed_ids & set(selected))
                        ),
                        "failed_trace_count": failed,
                        "strategy_counts": dict(strategies),
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

    status = "completed" if len(completed_ids & set(selected)) == len(selected) else "partial"
    manifest.update({"status": status, "updated_at": _now()})
    _atomic_json(manifest_path, manifest)
    return {**manifest, "run_dir": str(run), "stage_dir": str(stage_dir)}
