"""Independent analyses derived from core outputs and features."""

from pathlib import Path
from typing import Any

from .shared import load_context
from .model_comparison import run as run_model_comparison
from .harness_comparison import run as run_harness_comparison
from .training_selection import run as run_training_selection
from .unmet_need_priority import run as run_unmet_need_priority


EXTENSIONS = {
    "model_comparison": (run_model_comparison, "model_capability_comparison.csv"),
    "harness_comparison": (run_harness_comparison, "harness_capability_comparison.csv"),
    "training_selection": (run_training_selection, "training_trace_candidates.jsonl"),
    "unmet_need_priority": (run_unmet_need_priority, "unmet_need_priority.csv"),
}


def run_extension(name: str, core_dir: Path, feature_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    if name not in EXTENSIONS:
        raise ValueError(f"Unknown extension: {name}")
    context = load_context(core_dir, feature_paths)
    runner, filename = EXTENSIONS[name]
    result = runner(context, output_dir / filename)
    return {**result, "extension": name, "taxonomy_version": context.taxonomy_version}


__all__ = ["EXTENSIONS", "run_extension"]
