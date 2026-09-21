from __future__ import annotations

from pathlib import Path
from typing import Any

from ...preprocessing.pipeline import run_preprocess
from .user_turns import build_user_turns, user_turn_path


def resolve_batch(*, batch_dir: Path | None = None, sources: list[str] | None = None,
                  output_root: Path = Path("preprocessed"), adapter: str | None = None,
                  limit: int | None = None) -> tuple[Path, dict[str, Any]]:
    """Reuse an existing batch or run the established loss-preserving preprocessor."""
    if (batch_dir is None) == (not sources):
        raise ValueError("Specify exactly one of batch_dir or sources")
    if batch_dir is not None:
        resolved = batch_dir.expanduser().resolve()
        for name in ("unified_turns.jsonl", "unified_traces.jsonl"):
            if not (resolved / name).is_file():
                raise FileNotFoundError(resolved / name)
        user_result = build_user_turns(resolved)
        return resolved, {
            "stage": "01_preprocess",
            "status": "reused",
            "batch_id": resolved.name,
            "output_path": str(resolved),
            "user_turns": user_result,
        }
    result = run_preprocess(sources or [], output_root, adapter, limit)
    resolved = Path(str(result["output_path"])).expanduser().resolve()
    user_result = build_user_turns(resolved)
    return resolved, {"stage": "01_preprocess", **result, "user_turns": user_result}


__all__ = ["build_user_turns", "resolve_batch", "user_turn_path"]
