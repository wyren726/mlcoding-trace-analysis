from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from copy import deepcopy
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from ...preprocessing.adapters.common import redact, stable_id
from ...registries import registry_root_for_output, write_immutable_record
from .prompt import (
    AGENT_ATTRIBUTION_JSON_SCHEMA,
    AGENT_ATTRIBUTION_PROMPT_VERSION,
    AGENT_ATTRIBUTION_SYSTEM_PROMPT,
    PROMPT_VERSION,
    USER_ANALYSIS_SYSTEM_PROMPT,
    agent_attribution_prompt,
    user_analysis_prompt,
)
from .compact import (
    BOUNDARY_REPAIR_PROMPT_VERSION,
    BOUNDARY_REPAIR_SYSTEM_PROMPT,
    COMPACT_USER_PROMPT_VERSION,
    COMPACT_USER_SYSTEM_PROMPT,
    LONG_BOUNDARY_SYSTEM_PROMPT,
    LONG_EPISODE_SYSTEM_PROMPT,
    LONG_USER_PROMPT_VERSION,
    INDEX_SELECTION_PROMPT_VERSION,
    INDEX_SELECTION_SYSTEM_PROMPT,
    INDEXED_ATTRIBUTION_PROMPT_VERSION,
    INDEXED_ATTRIBUTION_SYSTEM_PROMPT,
    boundary_repair_prompt,
    compact_user_prompt,
    deterministic_strong_evidence_ids,
    evidence_selection_prompt,
    indexed_agent_attribution_prompt,
    indexed_candidate_payload,
    long_boundary_prompt,
    long_episode_prompt,
    normalise_compact_user_analysis,
    normalise_selected_event_ids,
    partition_indexed_candidate_payload,
    selected_evidence_payload,
)


ANALYSIS_SCHEMA_VERSION = "3.0"
ACTIVE_USER_PROMPT_VERSION = COMPACT_USER_PROMPT_VERSION
ACTIVE_ATTRIBUTION_PROMPT_VERSION = (
    f"{AGENT_ATTRIBUTION_PROMPT_VERSION}|{INDEXED_ATTRIBUTION_PROMPT_VERSION}"
)
OBSERVABLE_EVENT_TYPES = {
    "user_message", "assistant_message", "tool_call", "tool_result", "error",
}
REQUIREMENT_STATUSES = {"met", "partially_met", "unmet", "unknown"}
REQUIREMENT_ORIGINS = {"initial_query", "clarified_later", "new_later"}
OUTCOMES = {"completed", "partially_completed", "failed", "unknown"}
BOUNDARY_VALUES = {"reliable", "uncertain"}
DOMAIN_VALUES = {"in_scope", "adjacent", "out_of_scope", "uncertain"}
ANALYSIS_VALUE_VALUES = {"high", "medium", "low", "uncertain"}
QUERY_VALUES = {"identified", "uncertain", "not_identifiable"}
CLARITY_VALUES = {"clear", "partly_clear", "unclear"}
GOAL_VALUES = {"complete", "partial", "insufficient"}
EVOLUTION_VALUES = {"stable", "evolved", "uncertain"}
CHANGE_TYPES = {"clarification", "correction", "new_requirement"}
SIGNAL_JUDGMENTS = {"present", "absent", "uncertain"}
INTERRUPTION_JUDGMENTS = {
    "dissatisfaction_signal", "task_control", "accidental_or_unknown",
}
SIGNAL_TYPES = {
    "explicit_rejection", "result_defect", "constraint_repeated",
    "redo_or_rollback", "retry_after_failure", "user_takeover",
    "blocked_or_abandoned",
}
ROUTES = {"analyze", "review", "skip"}
GOAL_ASSESSMENTS = {"confirmed", "revised", "insufficient"}
ATTRIBUTIONS = {
    "reasoning_or_response", "execution_or_verification", "tool_use",
    "context_handling", "permission", "external", "user_input", "unknown",
}
ANALYSIS_METADATA_KEYS = {
    "model", "models", "harness", "provider", "api_provider", "service_tier",
    "request_id", "sessionId", "session_id",
}
SYNTHETIC_USER_PREFIXES = (
    "Base directory for this skill:",
    "[System reminder] This is a runtime system reminder, not user input.",
    "<task-notification>",
    "Resume from the paused agent session. The working directory is unchanged.",
    "[Request interrupted by user",
    "The previous request encountered a temporary API error.",
    "This session is being continued from a previous conversation that ran out of context.",
    "[system] Previously read image bytes have been cleared",
    "[User message] The previous PDF parsing attempt already failed",
)


class JSONClient(Protocol):
    model: str
    config: Any

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]: ...


@dataclass(frozen=True)
class TraceBundle:
    batch_id: str
    trace_id: str
    turns: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TraceResult:
    record: dict[str, Any]
    user_screen_record: dict[str, Any]
    strategy: str
    model_call_count: int
    usage: dict[str, int]


@dataclass(frozen=True)
class UserScreenResult:
    record: dict[str, Any]
    strategy: str
    model_call_count: int
    usage: dict[str, int]


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _turn_sort_key(turn: dict[str, Any]) -> tuple[int, str]:
    try:
        index = int(turn.get("turn_index") or 0)
    except (TypeError, ValueError):
        index = 0
    return index, str(turn.get("turn_id") or "")


def iter_trace_bundles(turn_path: Path) -> Iterator[TraceBundle]:
    """Stream v1 Turn JSONL without loading a multi-gigabyte batch into memory.

    The established preprocessor writes each trace contiguously.  We validate that
    invariant so a future source cannot silently split one logical trace.
    """
    current_id: str | None = None
    current_batch = ""
    current_turns: list[dict[str, Any]] = []
    emitted: set[str] = set()
    with turn_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            turn = json.loads(line)
            if not isinstance(turn, dict):
                raise ValueError(f"Turn line {line_number} is not an object: {turn_path}")
            trace_id = str(turn.get("trace_id") or "")
            if not trace_id:
                raise ValueError(f"Turn line {line_number} has no trace_id: {turn_path}")
            if current_id is None:
                current_id = trace_id
                current_batch = str(turn.get("batch_id") or turn_path.parent.name)
            elif trace_id != current_id:
                emitted.add(current_id)
                yield TraceBundle(
                    current_batch, current_id,
                    tuple(sorted(current_turns, key=_turn_sort_key)),
                )
                if trace_id in emitted:
                    raise ValueError(
                        f"Trace {trace_id} is non-contiguous in {turn_path}; reindex before analysis"
                    )
                current_id = trace_id
                current_batch = str(turn.get("batch_id") or turn_path.parent.name)
                current_turns = []
            current_turns.append(turn)
    if current_id is not None:
        yield TraceBundle(
            current_batch, current_id, tuple(sorted(current_turns, key=_turn_sort_key))
        )


def _clean_user_content(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    if content.startswith(SYNTHETIC_USER_PREFIXES):
        return None
    if content.startswith("[用户提问]") and "用户消息：" in content:
        return content.split("用户消息：", 1)[1].strip() or None
    markers = (
        "### User message",
        "Load and follow the `universal-agent-orchestrator` skill to begin.",
    )
    for marker in markers:
        if marker in content:
            return content.rsplit(marker, 1)[1].strip() or None
    if content.startswith("You are Wismodel"):
        return None
    return content.strip() or None


def _analysis_event(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    # session_jsonl stores platform instructions and the actual user message in
    # one content string.  The marker is source-provided and lossless in the
    # preprocessed Event; the analysis view exposes only the actual user text so
    # platform instructions cannot be counted as user demand.
    if event.get("type") == "user_message" and isinstance(data, dict):
        content = _clean_user_content(data.get("content"))
        data = {**data, "content": content}
    if isinstance(data, dict):
        data = {key: value for key, value in data.items() if key not in ANALYSIS_METADATA_KEYS}
    data = redact(data)
    return {
        "event_id": event.get("event_id"),
        "sequence": event.get("sequence"),
        "timestamp": event.get("timestamp"),
        "type": event.get("type"),
        "data": data,
    }


def _real_user_event(turn: dict[str, Any]) -> dict[str, Any] | None:
    """Return the real user input that opened a Turn.

    Claude-style harnesses also encode Skill payloads, runtime reminders, and
    resume notifications as ``user_message`` records.  Known wrappers are
    unwrapped and synthetic-only Turns are omitted while the original
    preprocessed Event remains untouched.
    """
    for event in turn.get("events") or []:
        if event.get("type") != "user_message":
            continue
        data = event.get("data")
        content = data.get("content") if isinstance(data, dict) else None
        if _clean_user_content(content) is not None:
            return _analysis_event(event)
    return None


def _user_interruption_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Expose a source-recorded user action without treating it as a Query."""
    if event.get("type") != "user_message":
        return None
    data = event.get("data")
    content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(content, str) or not content.startswith(
        "[Request interrupted by user"
    ):
        return None
    value = _analysis_event(event)
    value["type"] = "user_interruption"
    value["data"] = {
        "content": content,
        "action": "tool_use_rejected" if "tool use" in content.lower() else "generation_interrupted",
    }
    return value


def user_trace_payload(bundle: TraceBundle) -> dict[str, Any]:
    turns = []
    for turn in bundle.turns:
        event = _real_user_event(turn)
        if event is None:
            continue
        turns.append({
            "turn_id": turn.get("turn_id"),
            "turn_index": turn.get("turn_index"),
            "user_message": event,
        })
    return {
        "trace_id": bundle.trace_id,
        "input_view": "all_real_user_turns",
        "turns": turns,
    }


_CONTEXT_DEPENDENT_EPISODE_START = re.compile(
    r"^(?:"
    r"好(?:的|呢|呀)?|行|对|是的|没问题|"
    r"方案\s*[0-9一二三四五六七八九十]+|"
    r"按(?:照)?(?:这个|上面|前面)|"
    r"继续(?:完成|修改|处理|做)?|这个|这样"
    r")(?:[\s，,。.!！?？]|$)",
    re.IGNORECASE,
)


def _user_turn_content(turn: dict[str, Any]) -> str:
    message = turn.get("user_message")
    data = message.get("data") if isinstance(message, dict) else None
    content = data.get("content") if isinstance(data, dict) else None
    return content.strip() if isinstance(content, str) else ""


def _context_dependent_episode_start(content: str) -> bool:
    """Conservatively flag short replies that cannot open a standalone task."""
    compact = re.sub(r"\s+", " ", content).strip()
    dependent_prefixes = (
        "继续", "这个", "这样", "按这个", "按照这个",
        "按上面", "按照上面", "按前面", "按照前面",
    )
    return bool(
        compact
        and len(compact) <= 80
        and (
            _CONTEXT_DEPENDENT_EPISODE_START.match(compact)
            or compact.startswith(dependent_prefixes)
        )
    )


def episode_partition_issues(
    raw: dict[str, Any], user_payload: dict[str, Any]
) -> list[str]:
    """Return deterministic structural and conservative semantic boundary issues.

    A valid Episode list is an exact ordered partition of the real User Turns
    visible to the screening model.  This check deliberately does not infer a
    better semantic split; it only detects unsafe output and a small class of
    clearly context-dependent starts.
    """
    turns = user_payload.get("turns")
    turns = turns if isinstance(turns, list) else []
    turn_ids = [str(turn.get("turn_id") or "") for turn in turns]
    positions = {turn_id: index for index, turn_id in enumerate(turn_ids) if turn_id}
    episodes = raw.get("task_episodes")
    if not isinstance(episodes, list):
        return ["structural:task_episodes_not_array"]
    if turns and not episodes:
        return ["structural:task_episodes_empty"]

    issues: list[str] = []
    coverage = [0] * len(turns)
    previous_start = -1
    for episode_index, episode in enumerate(episodes):
        prefix = f"episode_{episode_index}"
        if not isinstance(episode, dict):
            issues.append(f"structural:{prefix}:not_object")
            continue
        boundary = episode.get("episode_boundary")
        if not isinstance(boundary, dict):
            issues.append(f"structural:{prefix}:boundary_not_object")
            continue
        start_id = str(boundary.get("start_turn_id") or "")
        end_id = str(boundary.get("end_turn_id") or "")
        start = positions.get(start_id)
        end = positions.get(end_id)
        if start is None:
            issues.append(f"structural:{prefix}:start_unknown")
        if end is None:
            issues.append(f"structural:{prefix}:end_unknown")
        if start is None or end is None:
            continue
        if start > end:
            issues.append(f"structural:{prefix}:range_reversed")
            continue
        if start <= previous_start:
            issues.append(f"structural:{prefix}:not_strictly_ordered")
        previous_start = start
        for position in range(start, end + 1):
            coverage[position] += 1

        initial_query = episode.get("initial_query")
        initial_query = initial_query if isinstance(initial_query, dict) else {}
        query_id = str(initial_query.get("turn_id") or "")
        query_judgment = str(initial_query.get("judgment") or "")
        query = positions.get(query_id) if query_id else None
        if query_judgment == "identified" and query is None:
            issues.append(f"structural:{prefix}:identified_query_unknown")
        elif query is not None and not start <= query <= end:
            issues.append(f"structural:{prefix}:query_outside_boundary")

        if episode_index > 0 and _context_dependent_episode_start(
            _user_turn_content(turns[start])
        ):
            issues.append(f"semantic:{prefix}:context_dependent_start")

    uncovered = [turn_ids[index] for index, count in enumerate(coverage) if count == 0]
    overlap = [turn_ids[index] for index, count in enumerate(coverage) if count > 1]
    if uncovered:
        issues.append(
            f"structural:uncovered_turns:{len(uncovered)}:" + ",".join(uncovered[:12])
        )
    if overlap:
        issues.append(
            f"structural:overlap_turns:{len(overlap)}:" + ",".join(overlap[:12])
        )
    return issues


def _full_trace_boundary_fallback(
    raw: dict[str, Any], user_payload: dict[str, Any]
) -> dict[str, Any]:
    """Preserve all source turns when a model cannot produce a safe partition."""
    turns = user_payload.get("turns") or []
    first_id = str(turns[0].get("turn_id") or "")
    last_id = str(turns[-1].get("turn_id") or "")
    return {
        "trace_id": user_payload.get("trace_id"),
        "task_episodes": [{
            "episode_boundary": {
                "judgment": "uncertain",
                "reason": "模型边界未通过本地分区校验，保守覆盖完整Trace",
                "start_turn_id": first_id,
                "end_turn_id": last_id,
            },
            "domain": {
                "judgment": "uncertain",
                "reason": "边界修复失败，保留待复核",
                "evidence_turn_ids": [],
            },
            "analysis_value": {
                "judgment": "uncertain",
                "reason": "边界修复失败，保留待复核",
                "evidence_turn_ids": [],
            },
            "initial_query": {
                "judgment": "uncertain",
                "turn_id": None,
                "reason": "无法安全定位完整Trace中的首次Query",
            },
            "initial_query_clarity": {
                "judgment": "unclear",
                "reason": "需要复核任务边界后判断",
                "evidence_turn_ids": [],
            },
            "preliminary_goal": {
                "judgment": "insufficient",
                "reason": "需要复核任务边界后还原目标",
                "goal": "",
                "requirements": [],
            },
            "requirement_evolution": {
                "judgment": "uncertain",
                "reason": "需要复核任务边界后判断",
                "changes": [],
            },
            "candidate_signals": {
                "judgment": "uncertain",
                "reason": "需要复核任务边界后判断",
                "signals": [],
            },
        }],
    }


def _observable_turn(turn: dict[str, Any]) -> dict[str, Any]:
    first_user_id = None
    first_user = _real_user_event(turn)
    if first_user is not None:
        first_user_id = first_user.get("event_id")
    events = []
    for event in turn.get("events") or []:
        if event.get("type") not in OBSERVABLE_EVENT_TYPES:
            continue
        if event.get("type") == "user_message":
            interruption = _user_interruption_event(event)
            if interruption is not None:
                events.append(interruption)
                continue
            if event.get("event_id") != first_user_id:
                continue
        events.append(_analysis_event(event))
    return {
        "turn_id": turn.get("turn_id"),
        "turn_index": turn.get("turn_index"),
        "status": turn.get("status"),
        "events": events,
    }


def candidate_agent_payload(bundle: TraceBundle,
                            user_record: dict[str, Any]) -> dict[str, Any]:
    """Build complete Episode spans for candidates routed to Agent verification.

    Stage 02's ``episode_boundary`` is the authoritative semantic range.  The
    former query-to-last-signal range could omit a later execution Turn and
    falsely turn a completed task into an apparent non-response.  One following
    real-user Turn is included only as outcome context (for example, the user
    reviewing a generated artifact); it is not part of the Episode itself.
    Telemetry, platform events, and reasoning remain excluded.
    """
    positions = {
        str(turn.get("turn_id")): index for index, turn in enumerate(bundle.turns)
    }
    selected_positions: set[int] = set()
    candidates = []
    evidence_spans = []
    for episode in user_record.get("task_episodes") or []:
        route = episode.get("route") if isinstance(episode, dict) else None
        if not isinstance(route, dict) or route.get("judgment") != "analyze":
            continue
        boundary = episode.get("episode_boundary")
        boundary = boundary if isinstance(boundary, dict) else {}
        start_turn_id = str(boundary.get("start_turn_id") or "")
        end_turn_id = str(boundary.get("end_turn_id") or "")
        start = positions.get(start_turn_id)
        end = positions.get(end_turn_id)

        # Compatibility for old persisted screens without valid boundaries.
        query_turn_id = str(episode.get("query_turn_id") or "")
        if start is None:
            start = positions.get(query_turn_id)
        candidate_signals = episode.get("candidate_signals")
        signals = (
            candidate_signals.get("signals")
            if isinstance(candidate_signals, dict) else []
        )
        signal_positions = [
            positions.get(str(item.get("turn_id") or ""))
            for item in signals or [] if isinstance(item, dict)
        ]
        signal_end_positions = [value for value in signal_positions if value is not None]
        if end is None and signal_end_positions:
            end = max(signal_end_positions)
        if start is None or end is None:
            continue
        if end < start:
            continue
        selected_positions.update(range(start, end + 1))
        lookahead = None
        for index in range(end + 1, len(bundle.turns)):
            if _real_user_event(bundle.turns[index]) is not None:
                lookahead = index
                selected_positions.add(index)
                break
        candidates.append(episode)
        evidence_spans.append({
            "episode_id": episode.get("episode_id"),
            "start_turn_id": bundle.turns[start].get("turn_id"),
            "end_turn_id": bundle.turns[end].get("turn_id"),
            "post_episode_context_turn_id": (
                bundle.turns[lookahead].get("turn_id") if lookahead is not None else None
            ),
        })
    return {
        "trace_id": bundle.trace_id,
        "input_view": "complete_episode_events_with_one_user_turn_lookahead",
        "candidate_episodes": candidates,
        "episode_evidence_spans": evidence_spans,
        "episode_turns": [
            _observable_turn(turn) for index, turn in enumerate(bundle.turns)
            if index in selected_positions
        ],
    }


# Compatibility alias for callers that previously previewed a "full" request.
# It now deliberately means the complete user-only view.
def full_trace_payload(bundle: TraceBundle) -> dict[str, Any]:
    return user_trace_payload(bundle)


def _payload_chars(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, (dict, list, tuple)):
        yield json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _enum(value: Any, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _usage(metadata: dict[str, Any]) -> dict[str, int]:
    raw = metadata.get("usage") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            result[key] = int(raw[key])
        except (KeyError, TypeError, ValueError):
            pass
    return result


def _add_usage(target: Counter[str], metadata: dict[str, Any]) -> None:
    target.update(_usage(metadata))


def _event_index(bundle: TraceBundle) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    events: dict[str, dict[str, Any]] = {}
    turns: dict[str, str] = {}
    for turn in bundle.turns:
        turn_id = str(turn.get("turn_id") or "")
        for event in turn.get("events") or []:
            event_id = str(event.get("event_id") or "")
            if event_id:
                events[event_id] = event
                turns[event_id] = turn_id
    return events, turns


def _normalise_evidence(value: Any, events: dict[str, dict[str, Any]], event_turns: dict[str, str],
                        warnings: list[str], path: str,
                        required_types: set[str] | None = None,
                        allowed_event_ids: set[str] | None = None,
                        visible_event_data: dict[str, Any] | None = None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(value if isinstance(value, list) else []):
        if not isinstance(item, dict):
            warnings.append(f"{path}[{index}]:not_object")
            continue
        event_id = str(item.get("event_id") or "")
        quote = _text(item.get("quote"))
        event = events.get(event_id)
        if event is None:
            warnings.append(f"{path}[{index}]:unknown_event_id")
            continue
        if allowed_event_ids is not None and event_id not in allowed_event_ids:
            warnings.append(f"{path}[{index}]:event_not_in_model_view")
            continue
        if required_types is not None and event.get("type") not in required_types:
            warnings.append(f"{path}[{index}]:wrong_event_type")
            continue
        if visible_event_data is not None:
            evidence_data = visible_event_data.get(event_id)
        else:
            evidence_data = (
                _analysis_event(event).get("data")
                if allowed_event_ids is not None else event.get("data")
            )
        if not quote or not any(quote in text for text in _string_values(evidence_data)):
            warnings.append(f"{path}[{index}]:quote_not_exact")
            continue
        key = (event_id, quote)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "turn_id": event_turns[event_id],
            "event_id": event_id,
            "quote": quote,
        })
    return result


def _real_user_event_ids(bundle: TraceBundle) -> set[str]:
    result = set()
    for turn in bundle.turns:
        event = _real_user_event(turn)
        if event and event.get("event_id"):
            result.add(str(event["event_id"]))
    return result


def _reasoned_judgment(value: Any, allowed: set[str], default: str,
                       warnings: list[str], path: str) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    if not isinstance(value, dict):
        warnings.append(f"{path}:not_object")
    judgment = _enum(source.get("judgment"), allowed, default)
    reason = _text(source.get("reason"))
    if not reason:
        warnings.append(f"{path}.reason:empty")
    return {"judgment": judgment, "reason": reason}


def _valid_turn_refs(value: Any, valid_turn_ids: set[str], warnings: list[str],
                     path: str) -> list[str]:
    result: list[str] = []
    for index, turn_id in enumerate(value if isinstance(value, list) else []):
        key = str(turn_id or "")
        if key not in valid_turn_ids:
            warnings.append(f"{path}[{index}]:unknown")
            continue
        if key not in result:
            result.append(key)
    return result


def _episode_route(episode: dict[str, Any]) -> dict[str, str]:
    domain = episode["domain"]["judgment"]
    value = episode["analysis_value"]["judgment"]
    boundary = episode["episode_boundary"]["judgment"]
    query = episode["initial_query"]["judgment"]
    clarity = episode["initial_query_clarity"]["judgment"]
    goal = episode["preliminary_goal"]["judgment"]
    signals = episode["candidate_signals"]["judgment"]
    if domain == "out_of_scope":
        return {"judgment": "skip", "reason": "目标领域明确不在当前研究范围内"}
    if value == "low":
        return {"judgment": "skip", "reason": "任务明确缺少进一步分析或构题价值"}
    if query == "not_identifiable" and goal == "insufficient":
        return {"judgment": "skip", "reason": "无法定位初始 Query，也无法还原可分析的目标"}
    if (
        domain in {"in_scope", "adjacent"}
        and value in {"high", "medium"}
        and boundary == "reliable"
        and query == "identified"
        and clarity == "clear"
        and goal in {"complete", "partial"}
        and signals == "present"
    ):
        return {
            "judgment": "analyze",
            "reason": "领域、价值、边界、Query 和初步目标满足条件，且存在用户侧候选信号",
        }
    return {
        "judgment": "review",
        "reason": "不能明确过滤，但至少一项进入完整 Agent 验证的条件尚未满足",
    }


def _trace_route(episodes: list[dict[str, Any]]) -> dict[str, str]:
    routes = [episode["route"]["judgment"] for episode in episodes]
    if "analyze" in routes:
        return {"judgment": "analyze", "reason": "至少一个 Episode 需要完整 Agent 验证"}
    if "review" in routes:
        return {"judgment": "review", "reason": "没有直接验证候选，但至少一个 Episode 需保守复核"}
    return {"judgment": "skip", "reason": "所有 Episode 均满足明确过滤条件"}


def normalise_user_analysis(raw: dict[str, Any], bundle: TraceBundle,
                            run_id: str) -> dict[str, Any]:
    """Validate R1-R8 and compute the R9 route without trusting model routing."""
    warnings: list[str] = []
    if raw.get("trace_id") != bundle.trace_id:
        warnings.append("trace_id:replaced_with_source_id")
    events, event_turns = _event_index(bundle)
    real_user_event_ids = _real_user_event_ids(bundle)
    user_payload = user_trace_payload(bundle)
    partition_issues = episode_partition_issues(raw, user_payload)
    valid_turn_ids = {
        str(turn.get("turn_id")) for turn in user_payload.get("turns") or []
    }
    episodes: list[dict[str, Any]] = []
    source_episodes = raw.get("task_episodes")
    if not isinstance(source_episodes, list):
        source_episodes = []
        warnings.append("task_episodes:not_array")
    for episode_index, item in enumerate(source_episodes):
        path = f"task_episodes[{episode_index}]"
        if not isinstance(item, dict):
            warnings.append(f"{path}:not_object")
            continue

        boundary_source = item.get("episode_boundary")
        boundary = _reasoned_judgment(
            boundary_source, BOUNDARY_VALUES, "uncertain", warnings,
            f"{path}.episode_boundary",
        )
        boundary_source = boundary_source if isinstance(boundary_source, dict) else {}
        start_turn_id = str(boundary_source.get("start_turn_id") or "")
        end_turn_id = str(boundary_source.get("end_turn_id") or "")
        if start_turn_id not in valid_turn_ids:
            warnings.append(f"{path}.episode_boundary.start_turn_id:unknown")
            start_turn_id = ""
        if end_turn_id not in valid_turn_ids:
            warnings.append(f"{path}.episode_boundary.end_turn_id:unknown")
            end_turn_id = ""
        boundary.update({"start_turn_id": start_turn_id, "end_turn_id": end_turn_id})

        domain_source = item.get("domain")
        domain = _reasoned_judgment(
            domain_source, DOMAIN_VALUES, "uncertain", warnings, f"{path}.domain"
        )
        domain["evidence_turn_ids"] = _valid_turn_refs(
            domain_source.get("evidence_turn_ids") if isinstance(domain_source, dict) else [],
            valid_turn_ids, warnings, f"{path}.domain.evidence_turn_ids",
        )

        value_source = item.get("analysis_value")
        analysis_value = _reasoned_judgment(
            value_source, ANALYSIS_VALUE_VALUES, "uncertain", warnings,
            f"{path}.analysis_value",
        )
        analysis_value["evidence_turn_ids"] = _valid_turn_refs(
            value_source.get("evidence_turn_ids") if isinstance(value_source, dict) else [],
            valid_turn_ids, warnings, f"{path}.analysis_value.evidence_turn_ids",
        )

        query_source = item.get("initial_query")
        initial_query = _reasoned_judgment(
            query_source, QUERY_VALUES, "uncertain", warnings, f"{path}.initial_query"
        )
        query_source = query_source if isinstance(query_source, dict) else {}
        query_turn_id = str(query_source.get("turn_id") or "")
        if query_turn_id and query_turn_id not in valid_turn_ids:
            warnings.append(f"{path}.initial_query.turn_id:unknown")
            query_turn_id = ""
        if initial_query["judgment"] == "identified" and not query_turn_id:
            warnings.append(f"{path}.initial_query:identified_without_turn")
            initial_query["judgment"] = "uncertain"
        initial_query["turn_id"] = query_turn_id or None

        clarity_source = item.get("initial_query_clarity")
        clarity = _reasoned_judgment(
            clarity_source, CLARITY_VALUES, "unclear", warnings,
            f"{path}.initial_query_clarity",
        )
        clarity["evidence_turn_ids"] = _valid_turn_refs(
            clarity_source.get("evidence_turn_ids") if isinstance(clarity_source, dict) else [],
            valid_turn_ids, warnings, f"{path}.initial_query_clarity.evidence_turn_ids",
        )

        goal_source = item.get("preliminary_goal")
        preliminary_goal = _reasoned_judgment(
            goal_source, GOAL_VALUES, "insufficient", warnings,
            f"{path}.preliminary_goal",
        )
        goal_source = goal_source if isinstance(goal_source, dict) else {}
        goal = _text(goal_source.get("goal"))
        if preliminary_goal["judgment"] != "insufficient" and not goal:
            warnings.append(f"{path}.preliminary_goal.goal:empty")
            preliminary_goal["judgment"] = "insufficient"
        episode_id = stable_id(
            "episode", bundle.trace_id, query_turn_id, episode_index, goal
        )
        requirements: list[dict[str, Any]] = []
        source_requirements = goal_source.get("requirements")
        if not isinstance(source_requirements, list):
            source_requirements = []
            warnings.append(f"{path}.preliminary_goal.requirements:not_array")
        for requirement_index, requirement in enumerate(source_requirements):
            requirement_path = f"{path}.preliminary_goal.requirements[{requirement_index}]"
            if not isinstance(requirement, dict):
                warnings.append(f"{requirement_path}:not_object")
                continue
            requirement_text = _text(requirement.get("text"))
            if not requirement_text:
                warnings.append(f"{requirement_path}.text:empty")
                continue
            evidence = _normalise_evidence(
                requirement.get("evidence"), events, event_turns, warnings,
                f"{requirement_path}.evidence", {"user_message"}, real_user_event_ids,
            )
            if not evidence:
                warnings.append(f"{requirement_path}.evidence:no_real_user_message")
            requirements.append({
                "requirement_id": stable_id(
                    "requirement", episode_id, requirement_index, requirement_text
                ),
                "text": requirement_text,
                "origin": _enum(
                    requirement.get("origin"), REQUIREMENT_ORIGINS, "clarified_later"
                ),
                "status": "unknown",
                "evidence": evidence,
            })
        preliminary_goal.update({"goal": goal, "requirements": requirements})

        evolution_source = item.get("requirement_evolution")
        evolution = _reasoned_judgment(
            evolution_source, EVOLUTION_VALUES, "uncertain", warnings,
            f"{path}.requirement_evolution",
        )
        evolution_source = evolution_source if isinstance(evolution_source, dict) else {}
        changes: list[dict[str, Any]] = []
        for change_index, change in enumerate(evolution_source.get("changes") or []):
            change_path = f"{path}.requirement_evolution.changes[{change_index}]"
            if not isinstance(change, dict):
                warnings.append(f"{change_path}:not_object")
                continue
            turn_id = str(change.get("turn_id") or "")
            text_value = _text(change.get("text"))
            evidence_source = change.get("evidence")
            evidence = _normalise_evidence(
                [evidence_source] if isinstance(evidence_source, dict) else [],
                events, event_turns, warnings, f"{change_path}.evidence",
                {"user_message"}, real_user_event_ids,
            )
            if turn_id not in valid_turn_ids or not text_value or not evidence:
                warnings.append(f"{change_path}:incomplete")
                continue
            changes.append({
                "turn_id": turn_id,
                "type": _enum(change.get("type"), CHANGE_TYPES, "clarification"),
                "text": text_value,
                "evidence": evidence[0],
            })
        evolution["changes"] = changes

        signals_source = item.get("candidate_signals")
        candidate_signals = _reasoned_judgment(
            signals_source, SIGNAL_JUDGMENTS, "uncertain", warnings,
            f"{path}.candidate_signals",
        )
        signals_source = signals_source if isinstance(signals_source, dict) else {}
        signals: list[dict[str, Any]] = []
        for signal_index, signal in enumerate(signals_source.get("signals") or []):
            signal_path = f"{path}.candidate_signals.signals[{signal_index}]"
            if not isinstance(signal, dict):
                warnings.append(f"{signal_path}:not_object")
                continue
            evidence = _normalise_evidence(
                [signal], events, event_turns, warnings, signal_path,
                {"user_message"}, real_user_event_ids,
            )
            signal_type = _enum(signal.get("type"), SIGNAL_TYPES, "")
            if not evidence or not signal_type:
                warnings.append(f"{signal_path}:incomplete")
                continue
            signals.append({"type": signal_type, **evidence[0]})
        if signals and candidate_signals["judgment"] != "present":
            warnings.append(f"{path}.candidate_signals:recomputed_present")
            candidate_signals["judgment"] = "present"
        if not signals and candidate_signals["judgment"] == "present":
            warnings.append(f"{path}.candidate_signals:present_without_valid_signal")
            candidate_signals["judgment"] = "uncertain"
        candidate_signals["signals"] = signals

        episode = {
            "episode_id": episode_id,
            "query_turn_id": query_turn_id or None,
            "episode_boundary": boundary,
            "domain": domain,
            "analysis_value": analysis_value,
            "initial_query": initial_query,
            "initial_query_clarity": clarity,
            "preliminary_goal": preliminary_goal,
            "requirement_evolution": evolution,
            "candidate_signals": candidate_signals,
            "outcome": "unknown",
            "agent_assessment": None,
        }
        episode["route"] = _episode_route(episode)
        episodes.append(episode)
    if partition_issues:
        warnings.extend(
            f"episode_partition:{issue}" for issue in partition_issues
        )
        # Never allow an unsafe partition to route into automatic Agent
        # verification merely because the model labelled it reliable.
        for episode in episodes:
            episode["episode_boundary"]["judgment"] = "uncertain"
            episode["route"] = _episode_route(episode)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "batch_id": bundle.batch_id,
        "trace_id": bundle.trace_id,
        "analysis_run_id": run_id,
        "analysis_status": "needs_review" if warnings else "complete",
        "route": _trace_route(episodes),
        "task_episodes": episodes,
        **({"validation_warnings": warnings} if warnings else {}),
    }


def merge_agent_attribution(raw: dict[str, Any], bundle: TraceBundle,
                            user_record: dict[str, Any],
                            allowed_event_ids: set[str],
                            visible_event_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Confirm the preliminary goal and attach observable Agent verification."""
    record = deepcopy(user_record)
    warnings = list(record.get("validation_warnings") or [])
    if raw.get("trace_id") != bundle.trace_id:
        warnings.append("agent_attribution.trace_id:replaced_with_source_id")
    events, event_turns = _event_index(bundle)
    source = raw.get("episode_attributions")
    if not isinstance(source, list):
        source = []
        warnings.append("episode_attributions:not_array")
    by_query = {
        str(item.get("query_turn_id")): item for item in source if isinstance(item, dict)
    }
    for episode_index, episode in enumerate(record["task_episodes"]):
        route = episode.get("route")
        if not isinstance(route, dict) or route.get("judgment") != "analyze":
            continue
        path = f"task_episodes[{episode_index}].agent_assessment"
        item = by_query.get(str(episode.get("query_turn_id")))
        if not isinstance(item, dict):
            warnings.append(f"{path}:missing")
            continue
        preliminary_goal = episode.get("preliminary_goal")
        requirements = (
            preliminary_goal.get("requirements")
            if isinstance(preliminary_goal, dict) else []
        )
        goal_source = item.get("goal_assessment")
        goal_assessment = _reasoned_judgment(
            goal_source, GOAL_ASSESSMENTS, "insufficient", warnings,
            f"{path}.goal_assessment",
        )
        goal_source = goal_source if isinstance(goal_source, dict) else {}
        confirmed_goal = _text(goal_source.get("goal"))
        goal_evidence = _normalise_evidence(
            goal_source.get("evidence"), events, event_turns, warnings,
            f"{path}.goal_assessment.evidence",
            allowed_event_ids=allowed_event_ids,
            visible_event_data=visible_event_data,
        )
        if goal_assessment["judgment"] != "insufficient" and not confirmed_goal:
            warnings.append(f"{path}.goal_assessment.goal:empty")
            goal_assessment["judgment"] = "insufficient"
        if goal_assessment["judgment"] != "insufficient" and not goal_evidence:
            warnings.append(f"{path}.goal_assessment.evidence:empty")
            goal_assessment["judgment"] = "insufficient"
        goal_assessment.update({"goal": confirmed_goal, "evidence": goal_evidence})

        result_by_index: dict[int, dict[str, Any]] = {}
        for result_index, result in enumerate(item.get("requirement_results") or []):
            if not isinstance(result, dict):
                warnings.append(f"{path}.requirement_results[{result_index}]:not_object")
                continue
            index = result.get("requirement_index")
            if not isinstance(index, int) or not 0 <= index < len(requirements):
                warnings.append(f"{path}.requirement_results[{result_index}]:bad_index")
                continue
            evidence = _normalise_evidence(
                result.get("evidence"), events, event_turns, warnings,
                f"{path}.requirement_results[{result_index}].evidence",
                allowed_event_ids=allowed_event_ids,
                visible_event_data=visible_event_data,
            )
            status = _enum(result.get("status"), REQUIREMENT_STATUSES, "unknown")
            if status != "unknown" and not evidence:
                warnings.append(f"{path}.requirement_results[{result_index}]:status_without_evidence")
                status = "unknown"
            result_by_index[index] = {"status": status, "evidence": evidence}
        for requirement_index, requirement in enumerate(requirements):
            result = result_by_index.get(requirement_index)
            if result is None:
                warnings.append(f"{path}.requirement_results[{requirement_index}]:missing")
                continue
            requirement["status"] = result["status"]
            requirement["evidence"] = list({
                (row["event_id"], row["quote"]): row
                for row in requirement["evidence"] + result["evidence"]
            }.values())

        confirmed_requirements = []
        for requirement_index, requirement in enumerate(
            item.get("confirmed_requirements") or []
        ):
            if not isinstance(requirement, dict) or requirement_index >= 8:
                continue
            text = _text(requirement.get("text"))
            origin = _enum(
                requirement.get("origin"),
                {"initial_query", "clarified_later", "new_later"},
                "clarified_later",
            )
            origin_evidence = _normalise_evidence(
                requirement.get("origin_evidence"), events, event_turns, warnings,
                f"{path}.confirmed_requirements[{requirement_index}].origin_evidence",
                allowed_event_ids=allowed_event_ids,
                visible_event_data=visible_event_data,
            )
            status = _enum(
                requirement.get("status"), REQUIREMENT_STATUSES, "unknown"
            )
            status_evidence = _normalise_evidence(
                requirement.get("status_evidence"), events, event_turns, warnings,
                f"{path}.confirmed_requirements[{requirement_index}].status_evidence",
                allowed_event_ids=allowed_event_ids,
                visible_event_data=visible_event_data,
            )
            if not text or not origin_evidence:
                warnings.append(
                    f"{path}.confirmed_requirements[{requirement_index}]:incomplete"
                )
                continue
            if status != "unknown" and not status_evidence:
                warnings.append(
                    f"{path}.confirmed_requirements[{requirement_index}]:status_without_evidence"
                )
                status = "unknown"
            confirmed_requirements.append({
                "requirement_id": stable_id(
                    "requirement", episode["episode_id"], requirement_index, text
                ),
                "text": text,
                "origin": origin,
                "status": status,
                "evidence": list({
                    (row["event_id"], row["quote"]): row
                    for row in [*origin_evidence, *status_evidence]
                }.values()),
            })
        if confirmed_requirements:
            preliminary_goal["requirements"] = confirmed_requirements

        user_interruptions = []
        interruption_values = item.get("user_interruptions")
        if not isinstance(interruption_values, list):
            interruption_values = []
            if item.get("user_interruptions") is not None:
                warnings.append(f"{path}.user_interruptions:not_array")
        for interruption_index, interruption in enumerate(interruption_values):
            interruption_path = (
                f"{path}.user_interruptions[{interruption_index}]"
            )
            if not isinstance(interruption, dict):
                warnings.append(f"{interruption_path}:not_object")
                continue
            direct = _normalise_evidence(
                [interruption.get("interruption_event")],
                events, event_turns, warnings,
                f"{interruption_path}.interruption_event",
                allowed_event_ids=allowed_event_ids,
                visible_event_data=visible_event_data,
            )
            if direct:
                source_event = events.get(direct[0]["event_id"]) or {}
                source_data = source_event.get("data")
                source_content = (
                    source_data.get("content")
                    if isinstance(source_data, dict) else None
                )
                if not isinstance(source_content, str) or not source_content.startswith(
                    "[Request interrupted by user"
                ):
                    warnings.append(
                        f"{interruption_path}.interruption_event:not_user_interruption"
                    )
                    direct = []
            followup = _normalise_evidence(
                interruption.get("followup_evidence"),
                events, event_turns, warnings,
                f"{interruption_path}.followup_evidence",
                allowed_event_ids=allowed_event_ids,
                visible_event_data=visible_event_data,
            )
            judgment = _enum(
                interruption.get("judgment"), INTERRUPTION_JUDGMENTS,
                "accidental_or_unknown",
            )
            reason = _text(interruption.get("reason"))
            if judgment == "dissatisfaction_signal" and not followup:
                warnings.append(
                    f"{interruption_path}:dissatisfaction_without_followup_evidence"
                )
                judgment = "accidental_or_unknown"
            if not direct or not reason:
                warnings.append(f"{interruption_path}:incomplete")
                continue
            user_interruptions.append({
                "interruption_event": direct[0],
                "judgment": judgment,
                "reason": reason,
                "followup_evidence": followup,
            })
        turn_positions = {
            str(turn.get("turn_id") or ""): index
            for index, turn in enumerate(bundle.turns)
        }
        boundary = episode.get("episode_boundary") or {}
        boundary_start = turn_positions.get(str(boundary.get("start_turn_id") or ""))
        boundary_end = turn_positions.get(str(boundary.get("end_turn_id") or ""))
        expected_interruption_ids = set()
        for event_id in allowed_event_ids:
            source_event = events.get(event_id) or {}
            source_data = source_event.get("data")
            source_content = (
                source_data.get("content")
                if isinstance(source_data, dict) else None
            )
            event_position = turn_positions.get(event_turns.get(event_id, ""))
            in_episode = (
                boundary_start is not None and boundary_end is not None
                and event_position is not None
                and boundary_start <= event_position <= boundary_end
            )
            if in_episode and isinstance(source_content, str) and source_content.startswith(
                "[Request interrupted by user"
            ):
                expected_interruption_ids.add(event_id)
        reported_interruption_ids = {
            value["interruption_event"]["event_id"] for value in user_interruptions
        }
        if expected_interruption_ids - reported_interruption_ids:
            warnings.append(f"{path}.user_interruptions:missing_visible_events")

        agent_failure = _text(item.get("agent_failure")) or None
        related_value = item.get("agent_related")
        agent_related = related_value if isinstance(related_value, bool) else None
        related_reason = _text(item.get("agent_related_reason"))
        gaps = []
        for gap_index, gap in enumerate(item.get("capability_gaps") or []):
            gap_path = f"{path}.capability_gaps[{gap_index}]"
            if not isinstance(gap, dict):
                warnings.append(f"{gap_path}:not_object")
                continue
            capability_key = _text(gap.get("capability_key"))
            capability_name = _text(
                gap.get("capability_name") or gap.get("capability")
            )
            reason = _text(gap.get("reason"))
            evidence = _normalise_evidence(
                gap.get("evidence"), events, event_turns, warnings,
                f"{gap_path}.evidence", allowed_event_ids=allowed_event_ids,
                visible_event_data=visible_event_data,
            )
            if not capability_name or not reason or not evidence:
                warnings.append(f"{gap_path}:incomplete")
                continue
            gaps.append({
                "gap_id": stable_id(
                    "gap", episode["episode_id"], gap_index, capability_name, reason
                ),
                "capability": capability_name,
                "capability_key": capability_key or None,
                "capability_name": capability_name,
                "reason": reason,
                "evidence": evidence,
            })
        if agent_related is True and (not agent_failure or not related_reason or not gaps):
            warnings.append(f"{path}:unsupported_agent_attribution")
            agent_related = None
        if agent_related is not True and gaps:
            warnings.append(f"{path}:gaps_without_agent_attribution")
            gaps = []
        episode["outcome"] = _enum(item.get("outcome"), OUTCOMES, "unknown")
        episode["agent_assessment"] = {
            "goal_assessment": goal_assessment,
            "user_interruptions": user_interruptions,
            "agent_failure": agent_failure,
            "agent_related": agent_related,
            "agent_related_reason": related_reason,
            "attribution": _enum(item.get("attribution"), ATTRIBUTIONS, "unknown"),
            "capability_gaps": gaps,
        }
        episode["pain_confirmed"] = agent_related is True
    record["analysis_status"] = "needs_review" if warnings else "complete"
    if warnings:
        record["validation_warnings"] = warnings
    else:
        record.pop("validation_warnings", None)
    return record


def normalise_analysis(raw: dict[str, Any], bundle: TraceBundle,
                       run_id: str) -> dict[str, Any]:
    """Backward-compatible name for the first, user-only normalization step."""
    return normalise_user_analysis(raw, bundle, run_id)


def analyze_user_trace(bundle: TraceBundle, client: JSONClient, run_id: str,
                       request_max_chars: int,
                       force_hierarchical: bool = False) -> UserScreenResult:
    """Run only the compact user-side screen so it can be persisted immediately."""
    usage: Counter[str] = Counter()
    user_payload = user_trace_payload(bundle)
    if not user_payload["turns"]:
        record = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "batch_id": bundle.batch_id,
            "trace_id": bundle.trace_id,
            "analysis_run_id": run_id,
            "analysis_status": "complete",
            "route": {"judgment": "skip", "reason": "没有真实用户 Turn"},
            "task_episodes": [],
            "analysis_note": "no_real_user_turns",
        }
        return UserScreenResult(record, "no_real_user_turns", 0, {})
    first_prompt = compact_user_prompt(user_payload)
    # Turn count alone is not a context-size signal.  Some traces contain
    # hundreds of very short user messages; forcing those through hierarchical
    # screening can create dozens of serial episode calls while the complete
    # prompt still fits comfortably in one request.
    hierarchical_threshold = min(request_max_chars, 120_000)
    if force_hierarchical or (
        len(user_payload["turns"]) >= 300
        and len(first_prompt) > hierarchical_threshold
    ):
        boundary_prompt = long_boundary_prompt(user_payload)
        if len(boundary_prompt) > request_max_chars:
            raise ValueError(
                "Complete user-turn view exceeds request_max_chars; refusing to truncate it"
            )
        boundary_raw, boundary_metadata = client.complete_json(
            LONG_BOUNDARY_SYSTEM_PROMPT, boundary_prompt
        )
        _add_usage(usage, boundary_metadata)
        positions = {
            str(turn.get("turn_id") or ""): index
            for index, turn in enumerate(user_payload["turns"])
        }
        # Models occasionally echo the human-readable turn_index as
        # ``turn_<index>`` even though the payload also carries an opaque
        # turn_id.  The alias is deterministic and unambiguous within a Trace.
        turn_aliases = {
            f"turn_{turn.get('turn_index')}": str(turn.get("turn_id") or "")
            for turn in user_payload["turns"]
            if turn.get("turn_index") is not None
        }
        boundaries = boundary_raw.get("episode_boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("Long Trace boundary pass returned no episode_boundaries")
        resolved_by_position: dict[int, dict[str, Any]] = {}
        boundary_corrections: list[str] = []
        for boundary_index, boundary in enumerate(boundaries):
            if not isinstance(boundary, dict):
                boundary_corrections.append(f"boundary_{boundary_index}:not_object_dropped")
                continue
            model_start_id = str(boundary.get("start_turn_id") or "")
            start_id = turn_aliases.get(model_start_id, model_start_id)
            start = positions.get(start_id)
            if start is None:
                # Dropping an unresolvable split merges neighbouring candidate
                # episodes.  It never drops a User Turn or invents a boundary.
                boundary_corrections.append(
                    f"boundary_{boundary_index}:unknown_start_dropped"
                )
                continue
            if start in resolved_by_position:
                boundary_corrections.append(
                    f"boundary_{boundary_index}:duplicate_start_dropped"
                )
                continue
            resolved_by_position[start] = boundary
        if 0 not in resolved_by_position:
            resolved_by_position[0] = {
                "start_turn_id": user_payload["turns"][0].get("turn_id"),
                "reason": "本地补全首个连续边界",
            }
            boundary_corrections.append("first_boundary:locally_inserted")
        resolved_boundaries = sorted(resolved_by_position.items())
        if list(resolved_by_position) != [item[0] for item in resolved_boundaries]:
            boundary_corrections.append("boundaries:locally_reordered")
        raw_episodes: list[dict[str, Any]] = []
        for boundary_index, (start, boundary) in enumerate(resolved_boundaries):
            start_id = str(boundary.get("start_turn_id") or "")
            start_id = turn_aliases.get(start_id, start_id)
            end = (
                resolved_boundaries[boundary_index + 1][0] - 1
                if boundary_index + 1 < len(resolved_boundaries)
                else len(user_payload["turns"]) - 1
            )
            end_id = str(user_payload["turns"][end].get("turn_id") or "")
            focused_payload = {
                "trace_id": bundle.trace_id,
                "episode_boundary": {
                    "start_turn_id": start_id,
                    "end_turn_id": end_id,
                    "boundary_reason": str(boundary.get("reason") or ""),
                },
                "turns": user_payload["turns"][start:end + 1],
            }
            episode_prompt = long_episode_prompt(focused_payload)
            if len(episode_prompt) > request_max_chars:
                raise ValueError(
                    f"Long Trace episode {boundary_index} exceeds request_max_chars"
                )
            episode_raw, episode_metadata = client.complete_json(
                LONG_EPISODE_SYSTEM_PROMPT, episode_prompt
            )
            _add_usage(usage, episode_metadata)
            episode_values = episode_raw.get("task_episodes")
            if isinstance(episode_values, list) and len(episode_values) > 1:
                first_value = episode_values[0]
                if all(value == first_value for value in episode_values[1:]):
                    episode_values = [first_value]
            if not isinstance(episode_values, list) or len(episode_values) != 1:
                # Change the prompt so a cached schema-valid but semantically
                # invalid response cannot poison every resume attempt.
                correction_prompt = episode_prompt + (
                    "\n\n纠正：上一次返回的task_episodes数量不是1。"
                    "本次只能返回一个元素，且必须概括整个指定边界，不得再次拆分。"
                )
                episode_raw, episode_metadata = client.complete_json(
                    LONG_EPISODE_SYSTEM_PROMPT, correction_prompt
                )
                _add_usage(usage, episode_metadata)
                episode_values = episode_raw.get("task_episodes")
                boundary_corrections.append(
                    f"episode_{boundary_index}:schema_correction_retry"
                )
            if not isinstance(episode_values, list) or len(episode_values) != 1:
                raise ValueError(
                    f"Long Trace episode {boundary_index} did not return exactly one result"
                )
            episode_value = episode_values[0]
            if not isinstance(episode_value, dict):
                raise ValueError(f"Long Trace episode {boundary_index} is not an object")
            if not isinstance(episode_value.get("episode_boundary"), dict):
                episode_value["episode_boundary"] = {}
            episode_value["episode_boundary"].update({
                "start_turn_id": start_id,
                "end_turn_id": end_id,
            })
            raw_episodes.append(episode_value)
        record = normalise_compact_user_analysis(
            {"trace_id": bundle.trace_id, "task_episodes": raw_episodes}, bundle, run_id
        )
        record["analysis_provenance"] = {
            "user_prompt_version": LONG_USER_PROMPT_VERSION,
            "user_input_view": "all_real_user_turns_hierarchical",
            "boundary_call_count": 1,
            "episode_call_count": len(raw_episodes),
            "boundary_corrections": boundary_corrections,
        }
        return UserScreenResult(
            record, "hierarchical_user_screen", 1 + len(raw_episodes), dict(usage)
        )
    if len(first_prompt) > request_max_chars:
        raise ValueError(
            "Complete user-turn view exceeds request_max_chars; refusing to truncate it"
        )
    raw, metadata = client.complete_json(COMPACT_USER_SYSTEM_PROMPT, first_prompt)
    _add_usage(usage, metadata)
    model_call_count = 1
    initial_partition_issues = episode_partition_issues(raw, user_payload)
    final_partition_issues = list(initial_partition_issues)
    repair_attempted = False
    fallback_used = False
    strategy = "compact_user_screen"
    if initial_partition_issues:
        repair_payload = {
            "trace_id": bundle.trace_id,
            "input_view": "all_real_user_turns_with_previous_boundary_output",
            "turns": user_payload["turns"],
            "previous_output": raw,
            "validation_issues": initial_partition_issues,
        }
        repair_prompt = boundary_repair_prompt(repair_payload)
        if len(repair_prompt) <= request_max_chars:
            repair_attempted = True
            repaired_raw, repair_metadata = client.complete_json(
                BOUNDARY_REPAIR_SYSTEM_PROMPT, repair_prompt
            )
            _add_usage(usage, repair_metadata)
            model_call_count += 1
            raw = repaired_raw
            final_partition_issues = episode_partition_issues(raw, user_payload)
            strategy = "compact_user_screen_boundary_repair"
        structural_issues = [
            issue for issue in final_partition_issues
            if issue.startswith("structural:")
        ]
        if structural_issues:
            raw = _full_trace_boundary_fallback(raw, user_payload)
            fallback_used = True
            strategy = "compact_user_screen_boundary_fallback"
    record = normalise_compact_user_analysis(raw, bundle, run_id)
    provenance = record.setdefault("analysis_provenance", {})
    provenance.update({
        "boundary_validation_version": "episode-partition-v1",
        "boundary_repair_prompt_version": BOUNDARY_REPAIR_PROMPT_VERSION,
        "boundary_repair_attempted": repair_attempted,
        "boundary_fallback_used": fallback_used,
        "initial_boundary_issues": initial_partition_issues,
        "final_boundary_issues": final_partition_issues,
    })
    if fallback_used:
        warnings = list(record.get("validation_warnings") or [])
        warnings.append("episode_partition:full_trace_uncertain_fallback")
        record["validation_warnings"] = warnings
        record["analysis_status"] = "needs_review"
    return UserScreenResult(
        record, strategy, model_call_count, dict(usage)
    )


def _evidence_review(raw: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("evidence_review")
    if not isinstance(source, dict):
        return {
            "judgment": "unknown",
            "reason": "模型未返回evidence_review",
            "missing_evidence_queries": [],
        }
    judgment = source.get("judgment")
    if judgment not in {"sufficient", "insufficient"}:
        judgment = "unknown"
    queries = []
    for value in source.get("missing_evidence_queries") or []:
        text = str(value or "").strip()
        if text and text not in queries:
            queries.append(text)
        if len(queries) >= 8:
            break
    return {
        "judgment": judgment,
        "reason": str(source.get("reason") or "").strip(),
        "missing_evidence_queries": queries,
    }


def _mark_evidence_review(record: dict[str, Any], review: dict[str, Any]) -> None:
    record.setdefault("analysis_provenance", {})["evidence_review"] = review
    if review.get("judgment") == "sufficient":
        return
    for episode in record.get("task_episodes") or []:
        route = episode.get("route") if isinstance(episode, dict) else None
        if not isinstance(route, dict) or route.get("judgment") != "analyze":
            continue
        assessment = episode.get("agent_assessment")
        if isinstance(assessment, dict):
            assessment["agent_related"] = None
            assessment["agent_related_reason"] = (
                "当前取回证据不足，不能确认或排除Agent能力问题"
            )
        episode["pain_confirmed"] = None
    warning = "agent_verification:evidence_not_confirmed_sufficient"
    warnings = record.setdefault("validation_warnings", [])
    if warning not in warnings:
        warnings.append(warning)
    record["analysis_status"] = "needs_review"


def complete_index_partition_with_fallback(
    client: JSONClient,
    partition: dict[str, Any],
    *,
    retrieval_context: dict[str, Any] | None = None,
    retry_depth: int = 0,
    max_retry_depth: int = 3,
    min_partition_chars: int = 2_000,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Complete one index partition, subdividing only a failing large request.

    Provider-side timeouts and malformed responses are often confined to one
    unusually difficult partition.  Successful siblings remain cacheable; a
    failed partition is split into smaller complete indexes instead of
    truncating or restarting the whole Trace.
    """
    prompt = evidence_selection_prompt(
        partition, retrieval_context=retrieval_context
    )
    try:
        raw, metadata = client.complete_json(
            INDEX_SELECTION_SYSTEM_PROMPT, prompt
        )
        return [(partition, raw, metadata)]
    except Exception:
        if retry_depth >= max_retry_depth or len(prompt) <= min_partition_chars:
            raise
        target_chars = max(min_partition_chars, len(prompt) // 2)
        if target_chars >= len(prompt):
            raise
        parent = partition.get("partition") or {}
        subpartitions = partition_indexed_candidate_payload(
            partition,
            max_prompt_chars=target_chars,
            total_selection_budget=int(parent.get("selection_limit") or 1),
        )
        if len(subpartitions) <= 1:
            raise
        results = []
        for subpartition in subpartitions:
            subpartition["partition"].update({
                "parent_index": parent.get("parent_index") or parent.get("index"),
                "fallback_depth": retry_depth + 1,
            })
            results.extend(complete_index_partition_with_fallback(
                client,
                subpartition,
                retrieval_context=retrieval_context,
                retry_depth=retry_depth + 1,
                max_retry_depth=max_retry_depth,
                min_partition_chars=min_partition_chars,
            ))
        return results


def _complete_agent_attribution(
    client: JSONClient, system_prompt: str, user_prompt: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require the structural payload needed for an auditable attribution.

    Syntactically valid JSON can still omit ``episode_attributions``.  Retry
    once with an explicit repair instruction rather than silently publishing a
    record whose evidence review and episode judgments contradict each other.
    """
    strict_complete = getattr(client, "complete_json_schema", None)
    if callable(strict_complete):
        value, metadata = strict_complete(
            system_prompt, user_prompt,
            name="agent_episode_attribution",
            schema=AGENT_ATTRIBUTION_JSON_SCHEMA,
        )
    else:
        value, metadata = client.complete_json(system_prompt, user_prompt)
    metadata_values = [metadata]
    def repair_known_key_typo(result: dict[str, Any]) -> None:
        if isinstance(result.get("episode_attributions"), list):
            return
        # GLM occasionally preserves the full valid payload under a
        # misspelled container key.  Repair only the observed, unambiguous
        # aliases; the nested attribution remains subject to normal validation.
        aliases = (
            "episode_attributations",
            "episode_attriburations",
        )
        candidates = [result.get(key) for key in aliases if isinstance(result.get(key), list)]
        if len(candidates) == 1:
            result["episode_attributions"] = candidates[0]

    repair_known_key_typo(value)
    episodes = value.get("episode_attributions")
    if not isinstance(episodes, list) or not episodes:
        repair_system = system_prompt + """

结构修复要求：返回对象必须包含非空的 episode_attributions 数组，并对输入中的每个候选 Episode
给出一项归因。即使判断不是 Agent 问题，也必须返回对应项并使用 agent_related=false/null，不能只返回
evidence_review。
"""
        if callable(strict_complete):
            value, repair_metadata = strict_complete(
                repair_system, user_prompt,
                name="agent_episode_attribution_repair",
                schema=AGENT_ATTRIBUTION_JSON_SCHEMA,
            )
        else:
            value, repair_metadata = client.complete_json(repair_system, user_prompt)
        metadata_values.append(repair_metadata)
        repair_known_key_typo(value)
        episodes = value.get("episode_attributions")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(
                "Agent verification omitted required non-empty episode_attributions after repair"
            )
    return value, metadata_values


def verify_screened_trace(bundle: TraceBundle, client: JSONClient,
                          user_screen: UserScreenResult,
                          direct_max_chars: int,
                          request_max_chars: int) -> TraceResult:
    """Verify an already-persistable user screen with direct or indexed evidence."""
    record = deepcopy(user_screen.record)
    usage: Counter[str] = Counter(user_screen.usage)
    if not any(
        isinstance(episode.get("route"), dict)
        and episode["route"].get("judgment") == "analyze"
        for episode in record.get("task_episodes") or []
    ):
        strategy = (
            user_screen.strategy
            if user_screen.strategy == "no_real_user_turns"
            else "compact_user_screen_only"
        )
        return TraceResult(
            record, deepcopy(user_screen.record), strategy,
            user_screen.model_call_count, dict(usage),
        )

    agent_payload = candidate_agent_payload(bundle, record)
    second_prompt = agent_attribution_prompt(agent_payload)
    model_call_count = user_screen.model_call_count
    analyze_episodes = [
        episode for episode in record.get("task_episodes") or []
        if isinstance(episode, dict)
        and isinstance(episode.get("route"), dict)
        and episode["route"].get("judgment") == "analyze"
    ]
    # A long Trace can contain several independent candidate Episodes whose
    # union exceeds the provider context even though each Episode fits. Verify
    # them independently and merge by stable episode_id. This preserves every
    # Episode's exact evidence and avoids arbitrary character truncation.
    if len(analyze_episodes) > 1 and len(second_prompt) > direct_max_chars:
        merged_record = deepcopy(record)
        merged_by_id = {
            str(episode.get("episode_id")): episode
            for episode in merged_record.get("task_episodes") or []
            if isinstance(episode, dict) and episode.get("episode_id")
        }
        child_strategies = []
        child_provenance = []
        total_calls = model_call_count
        for episode in analyze_episodes:
            episode_record = deepcopy(record)
            episode_record["task_episodes"] = [deepcopy(episode)]
            child = verify_screened_trace(
                bundle,
                client,
                UserScreenResult(
                    episode_record, "persisted_user_screen_episode", 0, {}
                ),
                direct_max_chars,
                request_max_chars,
            )
            child_episode = child.record["task_episodes"][0]
            merged_by_id[str(child_episode.get("episode_id"))] = child_episode
            total_calls += child.model_call_count
            usage.update(child.usage)
            child_strategies.append(child.strategy)
            child_provenance.append({
                "episode_id": child_episode.get("episode_id"),
                "strategy": child.strategy,
                **(child.record.get("analysis_provenance") or {}),
            })
        merged_record["task_episodes"] = [
            merged_by_id.get(str(episode.get("episode_id")), episode)
            if isinstance(episode, dict) else episode
            for episode in merged_record.get("task_episodes") or []
        ]
        if any(
            isinstance(episode, dict)
            and episode.get("route", {}).get("judgment") == "analyze"
            and episode.get("pain_confirmed") is None
            for episode in merged_record.get("task_episodes") or []
        ):
            merged_record["analysis_status"] = "needs_review"
        merged_record.setdefault("analysis_provenance", {}).update({
            "strategy": "episode_partitioned_agent_verification",
            "episode_verifications": child_provenance,
        })
        return TraceResult(
            merged_record,
            deepcopy(user_screen.record),
            "episode_partitioned_agent_verification",
            total_calls,
            dict(usage),
        )
    if len(second_prompt) <= direct_max_chars:
        attribution, metadata_values = _complete_agent_attribution(
            client,
            AGENT_ATTRIBUTION_SYSTEM_PROMPT, second_prompt
        )
        for metadata in metadata_values:
            _add_usage(usage, metadata)
        model_call_count += len(metadata_values)
        evidence_payload = agent_payload
        strategy = "compact_user_screen_then_direct_agent_verification"
        attribution_version = AGENT_ATTRIBUTION_PROMPT_VERSION
    else:
        index_payload = indexed_candidate_payload(agent_payload)
        partitions = partition_indexed_candidate_payload(
            index_payload, max_prompt_chars=min(request_max_chars, 50_000)
        )
        # The deterministic pass is a safety net beside semantic index
        # retrieval, not permission to attach every error-looking chunk. Keep
        # it within the same auditable retrieval budget; all remaining chunks
        # stay addressable through the complete index and the second round.
        deterministic_ids = deterministic_strong_evidence_ids(agent_payload)[:24]
        selected_ids = list(deterministic_ids)
        retrieval_rounds = []
        for partition in partitions:
            for effective, selection, metadata in complete_index_partition_with_fallback(
                client, partition
            ):
                _add_usage(usage, metadata)
                model_call_count += 1
                limit = int(effective.get("partition", {}).get("selection_limit") or 1)
                additional = normalise_selected_event_ids(
                    selection, agent_payload, limit=limit, excluded=set(selected_ids)
                )
                selected_ids.extend(additional)
                retrieval_rounds.append({
                    "round": 1,
                    "partition": effective.get("partition"),
                    "selected_tool_event_ids": additional,
                    "reason": str(selection.get("reason") or "").strip(),
                })
        evidence_payload = selected_evidence_payload(agent_payload, selected_ids)
        second_prompt = indexed_agent_attribution_prompt(evidence_payload)
        if len(second_prompt) > request_max_chars:
            raise ValueError(
                "Selected exact evidence exceeds request_max_chars; refusing arbitrary truncation"
            )
        attribution, metadata_values = _complete_agent_attribution(
            client,
            INDEXED_ATTRIBUTION_SYSTEM_PROMPT, second_prompt
        )
        for metadata in metadata_values:
            _add_usage(usage, metadata)
        model_call_count += len(metadata_values)
        review = _evidence_review(attribution)
        if review["judgment"] == "insufficient":
            additional_ids = []
            for partition in partitions:
                retrieval_context = {
                    "round": 2,
                    "previously_selected_event_ids": selected_ids,
                    "missing_evidence_queries": review["missing_evidence_queries"],
                    "evidence_review_reason": review["reason"],
                }
                for effective, followup, metadata in complete_index_partition_with_fallback(
                    client, partition, retrieval_context=retrieval_context
                ):
                    _add_usage(usage, metadata)
                    model_call_count += 1
                    limit = int(effective.get("partition", {}).get("selection_limit") or 1)
                    new_ids = normalise_selected_event_ids(
                        followup, agent_payload, limit=limit,
                        excluded=set(selected_ids) | set(additional_ids),
                    )
                    additional_ids.extend(new_ids)
                    retrieval_rounds.append({
                        "round": 2,
                        "partition": effective.get("partition"),
                        "selected_tool_event_ids": new_ids,
                        "reason": str(followup.get("reason") or "").strip(),
                        "requested_by": review,
                    })
            if additional_ids:
                selected_ids = [*selected_ids, *additional_ids]
                evidence_payload = selected_evidence_payload(agent_payload, selected_ids)
                second_prompt = indexed_agent_attribution_prompt(evidence_payload)
                if len(second_prompt) > request_max_chars:
                    raise ValueError(
                        "Expanded exact evidence exceeds request_max_chars; refusing arbitrary truncation"
                    )
                attribution, metadata_values = _complete_agent_attribution(
                    client,
                    INDEXED_ATTRIBUTION_SYSTEM_PROMPT, second_prompt
                )
                for metadata in metadata_values:
                    _add_usage(usage, metadata)
                model_call_count += len(metadata_values)
                review = _evidence_review(attribution)
        strategy = "compact_user_screen_then_indexed_agent_verification"
        attribution_version = INDEXED_ATTRIBUTION_PROMPT_VERSION
        record.setdefault("analysis_provenance", {}).update({
            "deterministic_strong_evidence_ids": deterministic_ids,
            "evidence_index_partition_count": len(partitions),
            "evidence_selection_prompt_version": INDEX_SELECTION_PROMPT_VERSION,
            "selected_tool_event_ids": selected_ids,
            "evidence_selection_reason": str(selection.get("reason") or "").strip(),
            "evidence_retrieval_rounds": retrieval_rounds,
        })
    allowed_event_ids = {
        str(event.get("event_id"))
        for turn in evidence_payload.get("episode_turns") or []
        for event in turn.get("events") or []
        if event.get("event_id")
    }
    visible_event_data = {
        str(event.get("event_id")): event.get("data")
        for turn in evidence_payload.get("episode_turns") or []
        for event in turn.get("events") or []
        if event.get("event_id")
    }
    merged = merge_agent_attribution(
        attribution, bundle, record, allowed_event_ids, visible_event_data
    )
    merged.setdefault("analysis_provenance", {}).update({
        "agent_prompt_version": attribution_version,
        "agent_input_view": evidence_payload.get("input_view"),
        "strategy": strategy,
    })
    if strategy == "compact_user_screen_then_indexed_agent_verification":
        _mark_evidence_review(merged, _evidence_review(attribution))
    return TraceResult(
        merged, deepcopy(user_screen.record), strategy, model_call_count, dict(usage),
    )


def analyze_trace(bundle: TraceBundle, client: JSONClient, run_id: str,
                  direct_max_chars: int, request_max_chars: int) -> TraceResult:
    user_screen = analyze_user_trace(bundle, client, run_id, request_max_chars)
    return verify_screened_trace(
        bundle, client, user_screen, direct_max_chars, request_max_chars
    )


def _batch_and_turn_path(batch_dir: Path) -> tuple[Path, Path, str]:
    batch = batch_dir.expanduser().resolve()
    turn_path = batch / "unified_turns.jsonl"
    if not turn_path.is_file():
        raise FileNotFoundError(turn_path)
    return batch, turn_path, batch.name


def _new_run_id(batch_id: str, model: str) -> str:
    now = dt.datetime.now().astimezone()
    suffix = hashlib.sha256(
        f"{batch_id}|{ACTIVE_USER_PROMPT_VERSION}|{model}|{now.isoformat()}".encode()
    ).hexdigest()[:8]
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"


def _completed_trace_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("trace_id"):
                result.add(str(value["trace_id"]))
    return result


def _progress_row(batch_id: str, run_id: str, total: int, completed: int, failed: int,
                  status: str) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "batch_id": batch_id,
        "analysis_run_id": run_id,
        "stage": "02_analyze",
        "status": status,
        "total_trace_count": total,
        "completed_trace_count": completed,
        "failed_trace_count": failed,
        "pending_trace_count": max(0, total - completed),
    }


def preview_trace_analysis(batch_dir: Path, analysis_root: Path, *, model: str,
                           limit: int = 5, direct_max_chars: int = 120_000,
                           request_max_chars: int = 800_000,
                           include_trace_ids: set[str] | None = None) -> dict[str, Any]:
    batch, turn_path, batch_id = _batch_and_turn_path(batch_dir)
    run_id = _new_run_id(batch_id, model).replace("run_", "preview_", 1)
    run_dir = analysis_root.expanduser().resolve() / batch_id / run_id
    stage_dir = run_dir / "02_analyze"
    stage_dir.mkdir(parents=True, exist_ok=False)
    output = stage_dir / "requests.jsonl"
    count = 0
    found_trace_ids: set[str] = set()
    strategies: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as target:
        for bundle in iter_trace_bundles(turn_path):
            if include_trace_ids is not None and bundle.trace_id not in include_trace_ids:
                continue
            if count >= limit:
                break
            found_trace_ids.add(bundle.trace_id)
            payload = user_trace_payload(bundle)
            prompt = compact_user_prompt(payload)
            if len(prompt) <= request_max_chars:
                row = {
                    "trace_id": bundle.trace_id,
                    "strategy": "compact_user_only_first_call",
                    "system": COMPACT_USER_SYSTEM_PROMPT,
                    "user": prompt,
                    "agent_evidence_strategy": (
                        "direct" if len(prompt) <= direct_max_chars else "decided_after_user_screen"
                    ),
                }
            else:
                row = {
                    "trace_id": bundle.trace_id,
                    "strategy": "rejected_oversize_user_view",
                    "request_chars": len(prompt),
                    "request_max_chars": request_max_chars,
                }
            strategies[str(row["strategy"])] += 1
            target.write(_json_line(row))
            count += 1
    if include_trace_ids is not None and limit >= len(include_trace_ids):
        missing_ids = include_trace_ids - found_trace_ids
        if missing_ids:
            raise ValueError(
                f"Unknown trace IDs for batch {batch_id}: {sorted(missing_ids)[:5]}"
            )
    manifest = {
        "manifest_version": "1.0",
        "analysis_run_id": run_id,
        "batch_id": batch_id,
        "batch_path": str(batch),
        "stage": "02_analyze",
        "status": "preview",
        "model": model,
        "prompt_version": ACTIVE_USER_PROMPT_VERSION,
        "attribution_prompt_version": ACTIVE_ATTRIBUTION_PROMPT_VERSION,
        "evidence_selection_prompt_version": INDEX_SELECTION_PROMPT_VERSION,
        "record_count": count,
        "selection": (
            "explicit_trace_ids" if include_trace_ids is not None else "first_traces"
        ),
        "strategy_counts": dict(strategies),
        "output_path": str(output),
        "updated_at": _now(),
    }
    _atomic_json(stage_dir / "manifest.json", manifest)
    return {**manifest, "run_dir": str(run_dir), "stage_dir": str(stage_dir)}


def run_trace_analysis(batch_dir: Path, analysis_root: Path, client: JSONClient,
                       provider_name: str, *, workers: int = 4, limit: int | None = None,
                       direct_max_chars: int = 120_000,
                       request_max_chars: int = 800_000,
                       resume_run_id: str | None = None,
                       include_trace_ids: set[str] | None = None) -> dict[str, Any]:
    batch, turn_path, batch_id = _batch_and_turn_path(batch_dir)
    root = analysis_root.expanduser().resolve()
    run_id = resume_run_id or _new_run_id(batch_id, client.model)
    run_dir = root / batch_id / run_id
    stage_dir = run_dir / "02_analyze"
    manifest_path = stage_dir / "manifest.json"
    screen_output = stage_dir / f"user_screen__{batch_id}__{run_id}.jsonl"
    output = stage_dir / f"trace_analysis__{batch_id}__{run_id}.jsonl"
    errors = stage_dir / "errors.jsonl"
    progress = stage_dir / "progress.jsonl"
    thinking_type = getattr(client, "thinking_type", None)
    model_parameters = {
        "thinking_type": thinking_type,
        "reasoning_effort": (
            "none" if thinking_type == "disabled"
            else getattr(client.config, "reasoning_effort", None)
        ),
        "max_completion_tokens": getattr(
            client, "max_completion_tokens",
            getattr(client.config, "max_completion_tokens", None),
        ),
    }
    previous: dict[str, Any] | None = None
    if resume_run_id:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("batch_id") != batch_id:
            raise ValueError("Resume run belongs to a different batch")
        if (previous.get("model") != client.model
                or previous.get("prompt_version") != ACTIVE_USER_PROMPT_VERSION):
            raise ValueError("Resume run model or prompt version does not match")
        previous_parameters = previous.get("model_parameters") or {}
        for key in ("thinking_type", "reasoning_effort"):
            if previous_parameters.get(key) != model_parameters.get(key):
                raise ValueError("Resume run model parameters do not match")
        previous_limit = previous_parameters.get("max_completion_tokens")
        current_limit = model_parameters.get("max_completion_tokens")
        if previous_limit != current_limit and not (
            isinstance(previous_limit, int)
            and isinstance(current_limit, int)
            and current_limit > previous_limit
        ):
            raise ValueError(
                "Resume may only increase max_completion_tokens; other model "
                "parameter changes require a new run"
            )
    else:
        stage_dir.mkdir(parents=True, exist_ok=False)

    completed_ids = _completed_trace_ids(output)
    screened_ids = _completed_trace_ids(screen_output)
    # Count in one streaming pass, then stream again into the bounded executor.
    # Holding all Turn payloads would multiply the memory cost of a large batch.
    batch_trace_ids = {bundle.trace_id for bundle in iter_trace_bundles(turn_path)}
    total = len(batch_trace_ids)
    if include_trace_ids is not None:
        missing_ids = include_trace_ids - batch_trace_ids
        if missing_ids:
            raise ValueError(
                f"Unknown trace IDs for batch {batch_id}: {sorted(missing_ids)[:5]}"
            )
        eligible_trace_ids = include_trace_ids - completed_ids
    else:
        eligible_trace_ids = batch_trace_ids - completed_ids
    # A resumed run used to encounter historical failures in source order.  A
    # few slow, repeatedly timing-out traces could therefore occupy every
    # worker before untouched traces received a chance to run.  Keep those
    # auditable failures eligible, but defer them until fresh pending work has
    # been attempted.
    historical_failure_ids = _completed_trace_ids(errors) - completed_ids
    deferred_retry_ids = eligible_trace_ids & historical_failure_ids
    fresh_trace_ids = eligible_trace_ids - deferred_retry_ids
    pending_count = len(eligible_trace_ids)
    selected_count = min(pending_count, limit) if limit is not None else pending_count
    manifest = {
        "manifest_version": "1.0",
        "analysis_run_id": run_id,
        "batch_id": batch_id,
        "batch_path": str(batch),
        "stage": "02_analyze",
        "status": "running",
        "provider": provider_name,
        "model": client.model,
        "prompt_version": ACTIVE_USER_PROMPT_VERSION,
        "attribution_prompt_version": ACTIVE_ATTRIBUTION_PROMPT_VERSION,
        "evidence_selection_prompt_version": INDEX_SELECTION_PROMPT_VERSION,
        "model_parameters": model_parameters,
        "total_trace_count": total,
        "target_trace_count": total,
        "selected_trace_count": selected_count,
        "fresh_trace_count": len(fresh_trace_ids),
        "deferred_retry_trace_count": len(deferred_retry_ids),
        "scheduling_policy": "fresh_before_historical_failures",
        "selection": (
            "explicit_trace_ids" if include_trace_ids is not None else "all_pending"
        ),
        "completed_trace_count": len(completed_ids),
        "failed_trace_count": 0,
        "model_call_count": 0,
        "strategy_counts": {},
        "usage": {},
        "user_screen_output_path": str(screen_output),
        "output_path": str(output),
        "errors_path": str(errors),
        "progress_path": str(progress),
        "updated_at": _now(),
    }
    if previous and previous.get("execution_segments"):
        manifest["execution_segments"] = previous["execution_segments"]
    parameter_history = list((previous or {}).get("model_parameter_history") or [])
    if previous and not parameter_history and previous.get("model_parameters"):
        parameter_history.append(previous["model_parameters"])
    if model_parameters not in parameter_history:
        parameter_history.append(model_parameters)
    manifest["model_parameter_history"] = parameter_history
    _atomic_json(manifest_path, manifest)
    completed_count = len(completed_ids)
    failed_count = 0
    model_call_count = 0
    strategies: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as progress_out:
        progress_out.write(_json_line(_progress_row(
            batch_id, run_id, total, completed_count, failed_count, "running"
        )))

    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    pending_futures: dict[Future[TraceResult], TraceBundle] = {}

    def selected_bundles() -> Iterator[TraceBundle]:
        emitted = 0
        for scheduled_ids in (fresh_trace_ids, deferred_retry_ids):
            for bundle in iter_trace_bundles(turn_path):
                if bundle.trace_id not in scheduled_ids:
                    continue
                if limit is not None and emitted >= limit:
                    return
                emitted += 1
                yield bundle

    bundles = iter(selected_bundles())

    def submit_one() -> bool:
        try:
            bundle = next(bundles)
        except StopIteration:
            return False
        future = executor.submit(
            analyze_trace, bundle, client, run_id, direct_max_chars, request_max_chars
        )
        pending_futures[future] = bundle
        return True

    try:
        for _ in range(max(1, workers) * 2):
            if not submit_one():
                break
        with screen_output.open("a", encoding="utf-8") as screen_out, \
                output.open("a", encoding="utf-8") as output_out, \
                errors.open("a", encoding="utf-8") as error_out, \
                progress.open("a", encoding="utf-8") as progress_out:
            while pending_futures:
                finished, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    bundle = pending_futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # one trace must not abort the batch
                        failed_count += 1
                        error_out.write(_json_line({
                            "timestamp": _now(),
                            "batch_id": batch_id,
                            "analysis_run_id": run_id,
                            "trace_id": bundle.trace_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }))
                        error_out.flush()
                    else:
                        if bundle.trace_id not in screened_ids:
                            screen_out.write(_json_line(result.user_screen_record))
                            screen_out.flush()
                            screened_ids.add(bundle.trace_id)
                        output_out.write(_json_line(result.record))
                        output_out.flush()
                        completed_count += 1
                        model_call_count += result.model_call_count
                        strategies[result.strategy] += 1
                        usage.update(result.usage)
                    progress_out.write(_json_line(_progress_row(
                        batch_id, run_id, total, completed_count,
                        failed_count, "running",
                    )))
                    progress_out.flush()
                    submit_one()
                    manifest.update({
                        "completed_trace_count": completed_count,
                        "failed_trace_count": failed_count,
                        "model_call_count": model_call_count,
                        "strategy_counts": dict(strategies),
                        "usage": dict(usage),
                        "updated_at": _now(),
                    })
                    _atomic_json(manifest_path, manifest)
    except BaseException:
        manifest.update({"status": "interrupted", "updated_at": _now()})
        _atomic_json(manifest_path, manifest)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    status = (
        "completed" if failed_count == 0 and completed_count >= total else "partial"
    )
    manifest.update({
        "status": status,
        "completed_trace_count": completed_count,
        "failed_trace_count": failed_count,
        "model_call_count": model_call_count,
        "strategy_counts": dict(strategies),
        "usage": dict(usage),
        "updated_at": _now(),
    })
    _atomic_json(manifest_path, manifest)
    with progress.open("a", encoding="utf-8") as progress_out:
        progress_out.write(_json_line(_progress_row(
            batch_id, run_id, total, completed_count, failed_count, status
        )))
    registry_record = None
    if status == "completed":
        registry_record = write_immutable_record(
            "analysis-runs", run_id,
            {
                "analysis_run_id": run_id,
                "product_id": "trace_analysis",
                "product_version": ANALYSIS_SCHEMA_VERSION,
                "input_batches": [batch_id],
                "output_path": str(stage_dir),
                "status": status,
                "record_count": completed_count,
                "error_count": failed_count,
                "prompt_version": ACTIVE_USER_PROMPT_VERSION,
            },
            artifacts=(screen_output, output, errors, progress, manifest_path),
            root=registry_root_for_output(root),
        )
    return {
        **manifest,
        "run_dir": str(run_dir),
        "stage_dir": str(stage_dir),
        "registry_record_path": str(registry_record.resolve()) if registry_record else None,
    }


__all__ = [
    "ACTIVE_ATTRIBUTION_PROMPT_VERSION", "ACTIVE_USER_PROMPT_VERSION",
    "ANALYSIS_SCHEMA_VERSION", "JSONClient", "TraceBundle", "TraceResult",
    "UserScreenResult", "analyze_trace", "analyze_user_trace",
    "candidate_agent_payload", "episode_partition_issues", "full_trace_payload", "iter_trace_bundles",
    "merge_agent_attribution", "normalise_analysis", "normalise_user_analysis",
    "preview_trace_analysis", "run_trace_analysis", "user_trace_payload",
    "verify_screened_trace",
]
