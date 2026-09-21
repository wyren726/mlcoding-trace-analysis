"""Narrow compatibility facade for the pre-V4 combined analysis engine.

The V4 pipeline has separate user-screen and Agent-verification stages. Their
stable implementation still reuses a few parsing, prompting, and long-context
primitives from ``stage_02_analyze``. Keeping those imports here makes that
temporary dependency explicit and prevents new production stages from treating
the old combined stage as their public API.

Do not add new analysis behavior here. New shared primitives should live in a
dedicated neutral module and callers can migrate away from this facade
incrementally.
"""

from __future__ import annotations

from .stage_02_analyze.compact import INDEX_SELECTION_PROMPT_VERSION
from .stage_02_analyze.prompt import AGENT_ATTRIBUTION_PROMPT_VERSION
from .stage_02_analyze.runner import (
    ACTIVE_USER_PROMPT_VERSION,
    JSONClient,
    TraceBundle,
    TraceResult,
    UserScreenResult,
    _observable_turn,
    _real_user_event,
    analyze_user_trace,
    iter_trace_bundles,
    verify_screened_trace,
)

__all__ = [
    "ACTIVE_USER_PROMPT_VERSION",
    "AGENT_ATTRIBUTION_PROMPT_VERSION",
    "INDEX_SELECTION_PROMPT_VERSION",
    "JSONClient",
    "TraceBundle",
    "TraceResult",
    "UserScreenResult",
    "_observable_turn",
    "_real_user_event",
    "analyze_user_trace",
    "iter_trace_bundles",
    "verify_screened_trace",
]
