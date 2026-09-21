from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from ..model_api import OpenAICompatibleClient, load_provider
from .stage_01_preprocess import resolve_batch
from .stage_02_analyze import preview_trace_analysis
from .stage_02_user_screen import run_user_screen
from .stage_03_agent_verify import run_agent_verify
from .stage_04_build_results import build_pain_cases
from .stage_05_report_publish import publish_pain_cases


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_pipeline(*, batch_dir: Path | None, sources: list[str] | None,
                 preprocessed_root: Path, analysis_root: Path,
                 adapter: str | None, mode: str, provider: str, model: str | None,
                 provider_config: Path, cache_dir: Path, workers: int,
                 limit: int | None, preview_limit: int,
                 direct_max_chars: int, request_max_chars: int,
                 timeout_seconds: int | None, max_retries: int | None,
                 resume_run_id: str | None,
                 include_trace_ids: set[str] | None = None,
                 thinking_type: str | None = None,
                 max_completion_tokens: int | None = None,
                 publication_root: Path | None = None) -> dict[str, Any]:
    """Run the separated user-screen, Agent-verification and publication stages."""
    if mode not in {"preview", "execute"}:
        raise ValueError("mode must be preview or execute")
    if direct_max_chars <= 0 or request_max_chars <= 0:
        raise ValueError("Context limits must be positive")
    if direct_max_chars > request_max_chars:
        raise ValueError("direct_max_chars cannot exceed request_max_chars")
    if mode == "preview" and resume_run_id:
        raise ValueError("--resume is only valid with --mode execute")
    batch, preprocess_result = resolve_batch(
        batch_dir=batch_dir, sources=sources, output_root=preprocessed_root,
        adapter=adapter,
    )
    config = load_provider(provider_config, provider)
    chosen_model = model or config.default_model
    if mode == "preview":
        analyze_result = preview_trace_analysis(
            batch, analysis_root, model=chosen_model, limit=preview_limit,
            direct_max_chars=direct_max_chars, request_max_chars=request_max_chars,
            include_trace_ids=include_trace_ids,
        )
        run_dir = Path(str(analyze_result["run_dir"]))
        result = {
            "pipeline_version": "3.0",
            "status": "preview",
            "batch_id": batch.name,
            "run_id": analyze_result["analysis_run_id"],
            "stages": {
                "01_preprocess": preprocess_result,
                "02_analyze": analyze_result,
                "03_export": {"status": "not_run_in_preview"},
            },
            "updated_at": _now(),
        }
        _atomic_json(run_dir / "run_manifest.json", result)
        return {**result, "run_dir": str(run_dir)}

    client = OpenAICompatibleClient(
        config, model=chosen_model, cache_dir=cache_dir,
        timeout_seconds=timeout_seconds, max_retries=max_retries,
        thinking_type=thinking_type,
        max_completion_tokens=max_completion_tokens,
    )
    if limit is not None:
        raise ValueError(
            "--limit is no longer accepted for execute mode; use explicit Trace IDs "
            "so Stage 02 and Stage 03 have an auditable target set"
        )
    screen_result = run_user_screen(
        batch, analysis_root, client, provider, workers=workers,
        request_max_chars=request_max_chars, run_id=resume_run_id,
        include_trace_ids=include_trace_ids,
    )
    run_dir = Path(str(screen_result["run_dir"]))
    verify_result = run_agent_verify(
        batch, run_dir, client, provider, workers=workers,
        direct_max_chars=direct_max_chars, request_max_chars=request_max_chars,
    )
    results = build_pain_cases(run_dir, batch)
    effective_publication_root = (
        publication_root.expanduser().resolve() if publication_root is not None
        else analysis_root.expanduser().resolve().parent / "publications" / "pain-report"
    )
    publication = publish_pain_cases(run_dir, effective_publication_root)
    status = (
        "completed"
        if screen_result["status"] == "completed"
        and verify_result["status"] == "completed"
        else "partial"
    )
    result = {
        "pipeline_version": "4.0",
        "status": status,
        "batch_id": batch.name,
        "run_id": screen_result["analysis_run_id"],
        "stages": {
            "01_preprocess": preprocess_result,
            "02_user_screen": screen_result,
            "03_agent_verify": verify_result,
            "04_results": results,
            "05_report_publish": publication,
        },
        "updated_at": _now(),
    }
    _atomic_json(run_dir / "run_manifest.json", result)
    return {**result, "run_dir": str(run_dir)}


__all__ = ["run_pipeline"]
