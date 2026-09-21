"""Pipeline public API with dependency-light lazy imports.

Read-only utilities should not require the model-client dependencies. Execution
entry points are imported only when a caller actually requests them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "run_pipeline": (".runner", "run_pipeline"),
    "resolve_batch": (".stage_01_preprocess", "resolve_batch"),
    "build_pilot_sample": (".stage_02_analyze", "build_pilot_sample"),
    "preview_trace_analysis": (".stage_02_analyze", "preview_trace_analysis"),
    "repair_trace_analysis": (".stage_02_analyze", "repair_trace_analysis"),
    "run_trace_analysis": (".stage_02_analyze", "run_trace_analysis"),
    "export_trace_analysis": (".stage_03_export", "export_trace_analysis"),
    "run_user_screen": (".stage_02_user_screen", "run_user_screen"),
    "run_agent_verify": (".stage_03_agent_verify", "run_agent_verify"),
    "build_pain_cases": (".stage_04_build_results", "build_pain_cases"),
    "publish_pain_cases": (".stage_05_report_publish", "publish_pain_cases"),
    "pipeline_status": (".status", "pipeline_status"),
    "write_pipeline_status": (".status", "write_pipeline_status"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
