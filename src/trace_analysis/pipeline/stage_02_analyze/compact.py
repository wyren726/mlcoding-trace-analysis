"""Compact user screening and indexed evidence retrieval for oversized Traces."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .prompt import AGENT_ATTRIBUTION_OUTPUT_SHAPE

if TYPE_CHECKING:
    from .runner import TraceBundle


COMPACT_USER_PROMPT_VERSION = "trace-user-screen-compact-v5"
LONG_USER_PROMPT_VERSION = "trace-user-screen-hierarchical-v2"
BOUNDARY_REPAIR_PROMPT_VERSION = "trace-user-screen-boundary-repair-v1"
INDEX_SELECTION_PROMPT_VERSION = "trace-agent-evidence-index-v3"
INDEXED_ATTRIBUTION_PROMPT_VERSION = "trace-agent-verification-indexed-v6"


COMPACT_USER_OUTPUT_SHAPE: dict[str, Any] = {
    "trace_id": "string",
    "task_episodes": [
        {
            "episode_boundary": {
                "judgment": "reliable|uncertain",
                "reason": "short string",
                "start_turn_id": "string",
                "end_turn_id": "string",
            },
            "domain": {
                "judgment": "in_scope|adjacent|out_of_scope|uncertain",
                "reason": "short string",
                "evidence_turn_ids": ["at most one string"],
            },
            "analysis_value": {
                "judgment": "high|medium|low|uncertain",
                "reason": "short string",
                "evidence_turn_ids": ["at most one string"],
            },
            "initial_query": {
                "judgment": "identified|uncertain|not_identifiable",
                "turn_id": "string|null",
                "reason": "short string",
            },
            "initial_query_clarity": {
                "judgment": "clear|partly_clear|unclear",
                "reason": "short string",
                "evidence_turn_ids": ["at most one string"],
            },
            "preliminary_goal": {
                "judgment": "complete|partial|insufficient",
                "reason": "short string",
                "goal": "one concise sentence",
            },
            "requirement_evolution": {
                "judgment": "stable|evolved|uncertain",
                "reason": "short string",
            },
            "candidate_signals": {
                "judgment": "present|absent|uncertain",
                "reason": "short string",
                "signals": [
                    {
                        "type": (
                            "explicit_rejection|result_defect|constraint_repeated|"
                            "redo_or_rollback|retry_after_failure|user_takeover|"
                            "blocked_or_abandoned"
                        ),
                        "turn_id": "string",
                        "event_id": "string",
                        "quote": "short exact user quote",
                    }
                ],
            },
        }
    ],
}


COMPACT_USER_SYSTEM_PROMPT = f"""
你是 Coding/ML Agent 真实用户轨迹的轻量用户侧筛选员。输入只包含按时间排列的真实用户消息，不包含
Agent 回复、工具结果、内部推理或模型身份。返回一个 JSON 对象，不输出 Markdown 或额外文字。

逐个连续用户目标完成 R1-R8：R1领域、R2分析价值、R3 Episode边界、R4初始Query、R5初始Query
清晰度、R6初步目标、R7需求是否变化、R8候选不满意信号。判断口径与以下要求同时成立：

- 拆分Episode前先做全Trace领域门控。目标范围是跨学科科研全流程，包括问题形成、文献研究、方案设计、
  数据收集与定量/质性分析、Coding、ML/LLM建模、实验与验证、结果解释、论文/PPT等科研表达，以及代码、
  数据、模型和其他Artifact交付。医学、生物、化学、材料、物理、社会科学等领域均可in_scope；不要求必须写代码。
- 简单润色、机械格式修改等科研场景任务仍可in_scope，其构题价值由analysis_value判断，不能仅因不含Coding
  判out_of_scope。只有明显不是科研活动的闲聊、生活服务、营销文案和一般平台内容才out_of_scope。
- 如果整条Trace的主要目标都明确不是科研活动，只输出一个覆盖首尾真实User Turn的汇总Episode，不再按每个
  请求拆分；该Episode的domain必须为out_of_scope，后续本地路由会直接skip。
- 同一目标的背景、澄清、纠正、直接追问、参数确认、重试和返工属于同一Episode；只有独立交付物切换才拆分。
- task_episodes必须是全部输入User Turn的有序、连续、无重叠分区：第一个Episode从首个Turn开始，
  最后一个Episode在末个Turn结束；相邻Episode首尾衔接，每个Turn恰好出现一次，不得遗漏、重叠或倒序。
- Episode必须从理解该目标所需的最早用户上下文开始。“好呢”“方案2”“按这个”“继续”“这个可以吗”
  等依赖前文的短回应不能单独作为新Episode的起点；应与其所回应的目标合并。紧邻边界、仍在询问同一
  对象的参数、材料、验证或实现细节，也应保留在原Episode中。
- initial_query.turn_id必须位于本Episode起止范围内；只能使用输入提供的opaque turn_id，不得自行构造ID。
- in_scope/adjacent限于Coding、ML/LLM、Agent工作流、实验/算法研究或紧密相关技术任务。
- high/medium要求任务真实、非平凡且原则上可验证；文本短或回合少不能单独判low。
- 清晰度只使用初始Query当时已有的用户上下文；Workspace本应由Agent读取的信息不算用户缺失。
- preliminary_goal只是用户侧的一句话初步假设，后续仍须结合Agent交互确认。
- 候选信号仅限明确拒绝、结果缺陷、重复约束、重做/回滚、失败后重试、用户接管、阻塞/放弃。
  普通补充、正常追问、回答澄清、新增要求和单纯没听懂不自动构成候选信号。

轻量输出约束：
- 每个reason最多40个中文字符或25个英文词；goal最多100个中文字符或60个英文词。
- domain、analysis_value、initial_query_clarity各最多一个evidence_turn_id。
- 只对真实候选信号输出逐条证据；每个Episode最多保留3个最强信号。
- 本阶段不展开requirements列表和requirement changes；它们只对进入Agent验证的Episode后续提取。
- 每项判断仍必须有具体reason；不确定时使用uncertain/insufficient，不得猜测。
- 只能引用输入中存在的turn_id/event_id，quote必须是对应用户消息中的短小连续原文。

输出结构：
{json.dumps(COMPACT_USER_OUTPUT_SHAPE, ensure_ascii=False)}
""".strip()


LONG_BOUNDARY_SYSTEM_PROMPT = """
你是科研 Agent 用户轨迹的 Episode 边界识别员。输入包含一条超长 Trace 的全部真实 User Turn。
只识别用户目标和独立交付物的边界，不分析 Agent，不判断痛点，不展开 Rubrics。

边界规则：同一目标的补充、澄清、纠正、直接追问、参数确认、重试、返工和连续迭代必须合并；只有切换到可独立验收的
新目标或新交付物才拆分。边界必须连续、按输入顺序排列且覆盖全部 User Turn，不得遗漏或重叠。
“好呢”“方案2”“按这个”“继续”“这个可以吗”等依赖前文的回应不能作为新Episode起点；紧邻边界且
仍在询问同一对象参数、材料、验证或实现细节的消息属于原Episode。只能复制输入中的opaque turn_id。
返回 JSON，不输出 Markdown。结构为：
{"trace_id":"string","episode_boundaries":[{"start_turn_id":"string","end_turn_id":"string","reason":"简短理由"}]}
""".strip()


LONG_EPISODE_SYSTEM_PROMPT = COMPACT_USER_SYSTEM_PROMPT + """

这是超长 Trace 的第二层分析。输入给出已经由完整 Trace 确定的一个 Episode 边界，以及该 Episode
范围内的完整真实 User Turn。只能输出一个 task_episodes 元素，不得再次拆分边界；应完成该 Episode
的紧凑 R1-R8 判断。episode_boundary.start_turn_id/end_turn_id 必须原样使用输入指定的边界。
"""


BOUNDARY_REPAIR_SYSTEM_PROMPT = f"""
你是科研 Agent 用户轨迹的 Episode 边界修复员。输入包含完整真实User Turn、上一次R1-R8输出及本地
确定性校验发现的问题。请返回修复后的完整JSON对象，不输出Markdown或额外文字。

硬约束：
1. task_episodes必须按输入顺序排列，并对全部输入Turn形成精确分区；第一个Episode从首个Turn开始，
   最后一个Episode在末个Turn结束，每个Turn恰好属于一个Episode，不得遗漏、重叠、倒序或使用未知ID。
2. initial_query.turn_id为null或位于对应Episode内部；judgment=identified时必须提供范围内的真实turn_id。
3. 同一目标的背景、补充、澄清、直接追问、参数确认、纠正、重试和返工必须合并。只有可独立验收的
   新目标或新交付物才拆分。
4. “好呢”“方案2”“按这个”“继续”“这个可以吗”等依赖前文的短回应不能作为新Episode起点；应将其
   与所依赖的前一目标合并。边界后的消息若仍询问同一对象的参数、材料、验证或实现细节，也属于原Episode。
5. 尽量保留上一次输出中仍然有效的R1-R8判断，只修复边界及受边界变化影响的字段；不得编造证据。

输出结构：
{json.dumps(COMPACT_USER_OUTPUT_SHAPE, ensure_ascii=False)}
""".strip()


INDEX_SELECTION_OUTPUT_SHAPE = {
    "trace_id": "string",
    "selected_event_ids": [
        "string, at most 24 exact tool event IDs or event_id#chunk_NNNN IDs"
    ],
    "reason": "short string",
}


INDEX_SELECTION_SYSTEM_PROMPT = f"""
你是候选Agent失败的证据检索员。输入包含用户侧候选Episode、完整User/Assistant/UserInterruption/Error事件，以及Tool
事件的可审计索引。Tool大字段保留event_id、结构摘要、检索概述、首尾预览、字符数和SHA256，原文仍在源数据中。

请选择最多24个必须读取精确原文的tool_call/tool_result证据ID，以验证候选失败、任务结果及归因。
较小事件使用event_id；较大事件会列出event_id#chunk_NNNN及其字符范围，必须选择具体chunk ID。
如果输入含retrieval_context，这是证据不足后的补充检索：根据missing_evidence_queries选择尚未读取的证据，
不要重复previously_selected_event_ids。
不要选择User/Assistant/UserInterruption/Error事件，因为它们已自动完整进入最终验证。不要作最终痛点判断。
只能返回输入索引中存在的Tool证据ID。返回JSON，不输出其他文字。
如果输入包含partition.selection_limit，则本分区最多选择该数量的证据ID。

输出结构：
{json.dumps(INDEX_SELECTION_OUTPUT_SHAPE, ensure_ascii=False)}
""".strip()


INDEXED_ATTRIBUTION_SYSTEM_PROMPT = f"""
你是 Coding/ML Agent 候选Episode验证员。输入包含用户侧Rubrics、preliminary_goal、完整的
User/Assistant/UserInterruption/Error事件，以及证据检索步骤选出的精确Tool事件或字符块。未选Tool证据仍可凭event_id回到源数据，
但不得作为本次结论证据。返回JSON，不输出其他文字。

任务：
1. 结合可观察交互确认、修订或保留insufficient的用户目标。
2. 判断候选信号是否对应真实Agent失败，而不是正常澄清、新增要求、权限、外部故障或用户输入不足。
3. agent_related=true必须给出可验证的agent_failure、agent_related_reason和至少一个capability_gap。
4. 所有证据只能引用本次完整输入中的turn_id/event_id，quote必须是连续原文。
5. 当前轻量用户筛选可能没有requirements；此时requirement_results返回空数组，并根据完整Episode在
   confirmed_requirements中提取最多8条真实用户要求、来源、完成状态及各自证据。
6. capability_gap必须同时提供英文snake_case capability_key和简洁中文capability_name。
7. evidence_review必须判断当前精确证据是否足够。证据不足时返回insufficient及具体missing_evidence_queries，
   不得仅因缺少Tool原文而判定agent_related=false。
8. user_interruption是用户主动打断行为，不等于Agent自行截断。逐个输出user_interruptions，并结合打断后
   首个真实用户Turn区分dissatisfaction_signal、task_control与accidental_or_unknown；没有后续语义证据时
   不得猜测具体不满意原因。仅因用户打断造成的回复不完整不能归因为Agent能力缺陷。

输出结构：
{json.dumps(AGENT_ATTRIBUTION_OUTPUT_SHAPE, ensure_ascii=False)}
""".strip()


def compact_user_prompt(payload: dict[str, Any]) -> str:
    return (
        "请根据以下完整真实User Turn输出紧凑的R1-R8筛选结果。\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def boundary_repair_prompt(payload: dict[str, Any]) -> str:
    return (
        "请根据本地校验问题修复Episode分区，并返回完整R1-R8结果。\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def long_boundary_prompt(payload: dict[str, Any]) -> str:
    return "请为以下完整超长User Trace识别连续Episode边界。\n\n" + _serialized(payload)


def long_episode_prompt(payload: dict[str, Any]) -> str:
    return "请只分析以下指定Episode并输出一个task_episodes元素。\n\n" + _serialized(payload)


def normalise_compact_user_analysis(
    raw: dict[str, Any], bundle: TraceBundle, run_id: str
) -> dict[str, Any]:
    """Adapt compact output to the established reviewable v3 record schema."""
    # Local import prevents a module cycle: the normal runner imports the compact
    # strategy, while the schema normalizer remains owned by the normal runner.
    from .runner import normalise_user_analysis

    adapted = deepcopy(raw)
    episodes = adapted.get("task_episodes")
    if isinstance(episodes, list):
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            goal = episode.get("preliminary_goal")
            if isinstance(goal, dict):
                goal.setdefault("requirements", [])
            evolution = episode.get("requirement_evolution")
            if isinstance(evolution, dict):
                evolution.setdefault("changes", [])
    record = normalise_user_analysis(adapted, bundle, run_id)
    record["analysis_provenance"] = {
        "user_prompt_version": COMPACT_USER_PROMPT_VERSION,
        "user_input_view": "all_real_user_turns",
    }
    return record


def _serialized(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _preview(text: str, width: int = 180) -> dict[str, str]:
    if len(text) <= width * 2:
        return {"text": text}
    return {"head": text[:width], "tail": text[-width:]}


def _searchable_text(value: Any) -> str:
    """Flatten textual values without searchable JSON wrapper keys/noise."""
    values: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return "\n".join(values) if values else _serialized(value)


_RETRIEVAL_TERMS = (
    "error", "exception", "failed", "failure", "missing", "empty", "not found",
    "timeout", "warning", "invalid", "denied", "abort", "rollback", "result",
    "artifact", "archive", "persist", "download", "upload", "write", "save",
    "file", "path", "zip", "test", "metric", "错误", "异常", "失败", "缺失",
    "为空", "未找到", "超时", "警告", "无效", "拒绝", "中止", "回滚",
    "结果", "产物", "归档", "持久化", "下载", "上传", "写入", "保存",
    "文件", "路径", "测试", "指标",
)


def _retrieval_summary(text: str, *, max_excerpts: int = 6,
                       excerpt_width: int = 220) -> dict[str, Any]:
    """Create a deterministic search aid from the full chunk.

    This is deliberately not evidence and not an abstractive model summary.  It
    scans the complete chunk for failure/result/artifact cues so relevant text in
    the middle is searchable even though the ordinary preview shows only the ends.
    """
    # Chunks are slices of serialized JSON and therefore contain escaped line
    # breaks.  Projecting them back makes individual error/config lines visible
    # to the index without changing the exact source span or its hash.
    projected = text.replace("\\n", "\n").replace("\\t", "\t")
    lowered = projected.lower()
    hits = [term for term in _RETRIEVAL_TERMS if term.lower() in lowered]
    excerpts: list[str] = []
    occupied: list[tuple[int, int]] = []
    for term in hits:
        start = 0
        needle = term.lower()
        while len(excerpts) < max_excerpts:
            position = lowered.find(needle, start)
            if position < 0:
                break
            left = max(0, position - excerpt_width // 2)
            right = min(len(projected), left + excerpt_width)
            if not any(left < old_right and right > old_left for old_left, old_right in occupied):
                excerpt = re.sub(r"\s+", " ", projected[left:right]).strip()
                if excerpt:
                    excerpts.append(excerpt)
                    occupied.append((left, right))
            start = position + max(1, len(term))
        if len(excerpts) >= max_excerpts:
            break
    structured_facts = []
    fact_patterns = (
        re.compile(r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s\\]+"),
        re.compile(r"\b(?:exit_status|exit_code|status)\s*[:=]\s*[^\s,;]+", re.I),
    )
    for pattern in fact_patterns:
        for match in pattern.finditer(projected):
            fact = match.group(0)[:320]
            if fact not in structured_facts:
                structured_facts.append(fact)
            if len(structured_facts) >= 12:
                break
        if len(structured_facts) >= 12:
            break
    return {
        "line_count": projected.count("\n") + 1,
        "keyword_hits": hits[:20],
        "salient_excerpts": excerpts,
        "structured_facts": structured_facts,
    }


def _structural_fields(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    allowed = {
        "call_id", "tool", "tool_name", "name", "command", "path", "file_path",
        "status", "error", "is_error", "exit_code", "url", "artifact_path",
    }
    result = {}
    for key, value in data.items():
        if key not in allowed:
            continue
        text = _serialized(value) if isinstance(value, (dict, list)) else str(value)
        result[key] = text[:500]
    return result


def indexed_candidate_payload(
    candidate_payload: dict[str, Any], *, externalize_chars: int = 2_000,
    evidence_chunk_chars: int = 12_000,
) -> dict[str, Any]:
    """Externalize large event bodies without deleting their source identity."""
    indexed_turns = []
    for turn in candidate_payload.get("episode_turns") or []:
        events = []
        for event in turn.get("events") or []:
            data_text = _serialized(event.get("data"))
            event_type = str(event.get("type") or "")
            should_externalize = (
                event_type in {"tool_call", "tool_result"}
                or len(data_text) > externalize_chars
            )
            if should_externalize:
                data = {
                    "externalized": True,
                    "char_count": len(data_text),
                    "sha256": hashlib.sha256(data_text.encode()).hexdigest(),
                    "structural_fields": _structural_fields(event.get("data")),
                    "preview": _preview(data_text),
                    "retrieval_summary": _retrieval_summary(
                        _searchable_text(event.get("data"))
                    ),
                }
                if len(data_text) > evidence_chunk_chars:
                    data["chunks"] = [
                        {
                            "chunk_id": (
                                f"{event.get('event_id')}#chunk_{start // evidence_chunk_chars:04d}"
                            ),
                            "start_char": start,
                            "end_char": min(start + evidence_chunk_chars, len(data_text)),
                            "char_count": min(evidence_chunk_chars, len(data_text) - start),
                            "sha256": hashlib.sha256(
                                data_text[start:start + evidence_chunk_chars].encode()
                            ).hexdigest(),
                            "preview": _preview(
                                data_text[start:start + evidence_chunk_chars]
                            ),
                            "retrieval_summary": _retrieval_summary(
                                data_text[start:start + evidence_chunk_chars]
                            ),
                        }
                        for start in range(0, len(data_text), evidence_chunk_chars)
                    ]
            else:
                data = event.get("data")
            events.append({**event, "data": data})
        indexed_turns.append({**turn, "events": events})
    return {
        "trace_id": candidate_payload.get("trace_id"),
        "input_view": "candidate_episode_event_index",
        "candidate_episodes": candidate_payload.get("candidate_episodes") or [],
        "episode_turns": indexed_turns,
    }


def evidence_selection_prompt(payload: dict[str, Any], *,
                              retrieval_context: dict[str, Any] | None = None) -> str:
    request = deepcopy(payload)
    if retrieval_context:
        request["retrieval_context"] = retrieval_context
    return (
        "请选择需要取回完整原文的Tool事件。\n\n"
        + _serialized(request)
    )


_STRONG_EVIDENCE_RE = re.compile(
    r"traceback|(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))|"
    r"\b(?:failed|failure|timeout|timed out|denied|not found)\b|"
    r"\b(?:exit_status|exit_code)\s*[=:]\s*[1-9][0-9]*|"
    r"错误|异常|失败|超时|拒绝|未找到",
    re.I,
)


def deterministic_strong_evidence_ids(
    candidate_payload: dict[str, Any], *, evidence_chunk_chars: int = 12_000,
    per_event_chunk_limit: int = 3,
) -> list[str]:
    """Select observable errors and explicit user/config conflicts locally."""
    selected: list[str] = []
    user_numeric_constraints: set[str] = set()
    for turn in candidate_payload.get("episode_turns") or []:
        for event in turn.get("events") or []:
            if event.get("type") == "user_message":
                user_numeric_constraints.update(re.findall(
                    r"(?<![A-Za-z0-9_])\d{3,}(?![A-Za-z0-9_])",
                    _searchable_text(event.get("data")),
                ))
    config_assignment = re.compile(
        r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*[\"']?(\d{3,})(?![A-Za-z0-9_])"
    )
    for turn in candidate_payload.get("episode_turns") or []:
        for event in turn.get("events") or []:
            if event.get("type") != "tool_result" or not event.get("event_id"):
                continue
            data = event.get("data")
            data_text = _serialized(data)
            searchable = _searchable_text(data)
            explicit_error = bool(isinstance(data, dict) and data.get("is_error") is True)
            matches = list(_STRONG_EVIDENCE_RE.finditer(searchable))
            relevant_config = [
                match for match in config_assignment.finditer(searchable)
                if match.group(1) in user_numeric_constraints
            ]
            if not explicit_error and not matches and not relevant_config:
                continue
            event_id = str(event["event_id"])
            if len(data_text) <= evidence_chunk_chars:
                selected.append(event_id)
                continue
            # Search exact serialized text for the same cues so selected chunk
            # offsets remain tied to the hashed source representation.
            positions = [match.start() for match in _STRONG_EVIDENCE_RE.finditer(data_text)]
            positions.extend(
                match.start() for match in config_assignment.finditer(data_text)
                if match.group(1) in user_numeric_constraints
            )
            chunk_indexes = sorted({position // evidence_chunk_chars for position in positions})
            if explicit_error and not chunk_indexes:
                chunk_indexes = [0]
            selected.extend(
                f"{event_id}#chunk_{index:04d}"
                for index in chunk_indexes[:per_event_chunk_limit]
            )
    return list(dict.fromkeys(selected))


def partition_indexed_candidate_payload(
    indexed_payload: dict[str, Any], *, max_prompt_chars: int = 100_000,
    total_selection_budget: int = 24,
) -> list[dict[str, Any]]:
    """Partition an evidence index without dropping any event or chunk index."""
    fragments: list[dict[str, Any]] = []
    for turn in indexed_payload.get("episode_turns") or []:
        for event in turn.get("events") or []:
            data = event.get("data")
            chunks = data.get("chunks") if isinstance(data, dict) else None
            if chunks:
                common = {key: value for key, value in data.items() if key != "chunks"}
                for chunk in chunks:
                    fragments.append({**turn, "events": [{
                        **event, "data": {**common, "chunks": [chunk]},
                    }]})
            else:
                fragments.append({**turn, "events": [event]})
    base = {
        "trace_id": indexed_payload.get("trace_id"),
        "input_view": "candidate_episode_event_index_partition",
        "candidate_episodes": indexed_payload.get("candidate_episodes") or [],
        "episode_turns": [],
    }
    if len(evidence_selection_prompt(base)) > max_prompt_chars:
        raise ValueError("Candidate episode metadata alone exceeds partition limit")
    partitions: list[dict[str, Any]] = []
    current = deepcopy(base)
    for fragment in fragments:
        trial = deepcopy(current)
        trial["episode_turns"].append(fragment)
        if len(evidence_selection_prompt(trial)) > max_prompt_chars:
            if not current["episode_turns"]:
                raise ValueError("Single evidence-index fragment exceeds partition limit")
            partitions.append(current)
            current = deepcopy(base)
            current["episode_turns"].append(fragment)
            if len(evidence_selection_prompt(current)) > max_prompt_chars:
                raise ValueError("Single evidence-index fragment exceeds partition limit")
        else:
            current = trial
    if current["episode_turns"] or not partitions:
        partitions.append(current)
    count = len(partitions)
    per_partition = max(1, (total_selection_budget + count - 1) // count)
    for index, partition in enumerate(partitions, 1):
        partition["partition"] = {
            "index": index, "count": count, "selection_limit": per_partition,
        }
    return partitions


def normalise_selected_event_ids(
    raw: dict[str, Any], candidate_payload: dict[str, Any], *, limit: int = 24,
    evidence_chunk_chars: int = 12_000, excluded: set[str] | None = None,
) -> list[str]:
    tool_ids: set[str] = set()
    for turn in candidate_payload.get("episode_turns") or []:
        for event in turn.get("events") or []:
            if event.get("type") not in {"tool_call", "tool_result"}:
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            data_text = _serialized(event.get("data"))
            if len(data_text) <= evidence_chunk_chars:
                tool_ids.add(event_id)
            else:
                tool_ids.update(
                    f"{event_id}#chunk_{start // evidence_chunk_chars:04d}"
                    for start in range(0, len(data_text), evidence_chunk_chars)
                )
    excluded = excluded or set()
    selected = []
    for value in raw.get("selected_event_ids") or []:
        event_id = str(value or "")
        if event_id in tool_ids and event_id not in selected and event_id not in excluded:
            selected.append(event_id)
        if len(selected) >= limit:
            break
    return selected


def selected_evidence_payload(
    candidate_payload: dict[str, Any], selected_event_ids: list[str], *,
    evidence_chunk_chars: int = 12_000,
) -> dict[str, Any]:
    """Restore selected exact Tool events/spans and pair calls/results by call_id."""
    selected = set(selected_event_ids)
    call_ids = set()
    for turn in candidate_payload.get("episode_turns") or []:
        for event in turn.get("events") or []:
            event_id = str(event.get("event_id") or "")
            if event_id not in selected and not any(
                value.startswith(f"{event_id}#chunk_") for value in selected
            ):
                continue
            data = event.get("data")
            if isinstance(data, dict) and data.get("call_id"):
                call_ids.add(str(data["call_id"]))
    final_turns = []
    for turn in candidate_payload.get("episode_turns") or []:
        events = []
        for event in turn.get("events") or []:
            event_type = str(event.get("type") or "")
            data = event.get("data")
            call_id = str(data.get("call_id") or "") if isinstance(data, dict) else ""
            event_id = str(event.get("event_id") or "")
            selected_chunks = sorted(
                value for value in selected
                if value.startswith(f"{event_id}#chunk_")
            )
            include = event_type in {"user_message", "assistant_message", "error"}
            include = include or event_id in selected or bool(selected_chunks)
            include = include or bool(call_id and call_id in call_ids)
            if include:
                if selected_chunks and event_type in {"tool_call", "tool_result"}:
                    data_text = _serialized(data)
                    chunks = []
                    for chunk_id in selected_chunks:
                        try:
                            chunk_index = int(chunk_id.rsplit("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        start = chunk_index * evidence_chunk_chars
                        if start >= len(data_text):
                            continue
                        text = data_text[start:start + evidence_chunk_chars]
                        chunks.append({
                            "chunk_id": chunk_id,
                            "start_char": start,
                            "end_char": start + len(text),
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "text": text,
                        })
                    selected_data = {
                        "source_event_id": event_id,
                        "source_data_char_count": len(data_text),
                        "source_data_sha256": hashlib.sha256(data_text.encode()).hexdigest(),
                        "structural_fields": _structural_fields(data),
                        "selected_exact_chunks": chunks,
                    }
                    events.append({**event, "data": selected_data})
                else:
                    events.append(event)
        if events:
            final_turns.append({**turn, "events": events})
    return {
        "trace_id": candidate_payload.get("trace_id"),
        "input_view": "candidate_episode_with_selected_exact_tool_evidence_spans",
        "candidate_episodes": candidate_payload.get("candidate_episodes") or [],
        "selected_tool_event_ids": sorted(selected),
        "episode_turns": final_turns,
    }


def indexed_agent_attribution_prompt(payload: dict[str, Any]) -> str:
    return (
        "请根据以下候选Episode与按需取回的完整证据完成Agent归因。\n\n"
        + _serialized(payload)
    )


def payload_chars(value: dict[str, Any]) -> int:
    return len(_serialized(value))


__all__ = [
    "BOUNDARY_REPAIR_PROMPT_VERSION", "BOUNDARY_REPAIR_SYSTEM_PROMPT",
    "COMPACT_USER_PROMPT_VERSION", "COMPACT_USER_SYSTEM_PROMPT",
    "INDEX_SELECTION_PROMPT_VERSION", "INDEX_SELECTION_SYSTEM_PROMPT",
    "INDEXED_ATTRIBUTION_PROMPT_VERSION", "INDEXED_ATTRIBUTION_SYSTEM_PROMPT",
    "compact_user_prompt", "boundary_repair_prompt", "normalise_compact_user_analysis",
    "indexed_candidate_payload", "evidence_selection_prompt",
    "normalise_selected_event_ids", "selected_evidence_payload",
    "deterministic_strong_evidence_ids", "partition_indexed_candidate_payload",
    "indexed_agent_attribution_prompt", "payload_chars",
]
