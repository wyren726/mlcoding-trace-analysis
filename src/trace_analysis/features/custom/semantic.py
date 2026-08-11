from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from ..model_api import OpenAICompatibleClient, load_provider
from ...preprocessing.adapters.common import redact_text, stable_id
from ..incremental import compatible_feature_files, completed_turn_ids
from ...registries import registry_root_for_output, write_immutable_record


class JSONClient(Protocol):
    model: str
    config: Any

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]: ...


TASKS = {
    "user_feedback": {
        "version": "v4",
        "instruction": ("评价当前 Turn 中 Agent 结果收到的用户反馈。NEXT_TURN_USER_FOLLOW_UP 仅作为当前 Turn 的后续反馈；"
                        "不要把它归到下一 Turn。只能使用 NEXT_TURN_USER_FOLLOW_UP 中明确评价当前结果的原文作为证据，"
                        "证据 quote 必须逐字摘录。新需求、进一步请求、追问、上传说明、简单的‘继续/继续修改’以及用户沉默，"
                        "都不能单独证明满意或不满意，此时 sentiment=none、explicit=false、evidence=[]。"
                        "只有明确赞扬、抱怨、否定、纠错或陈述未完成时，才标 positive/negative/mixed 且 explicit=true。"
                        "positive 必须含‘很好、满意、正是我要的、做得不错’等对结果的直接肯定。"
                        "‘还要、也需要改、请继续、请核查、请提供、可以再做’属于后续工作要求，不是 positive；"
                        "例如‘DOI联网逐条核查真实性，文内引用也需要改’应标 none，而不是 positive。"),
        "schema": {"sentiment": "positive|negative|mixed|none|unclear", "explicit": "boolean",
                   "user_correction": "boolean", "complaints": ["string"],
                   "evidence": [{"event_id": "string", "quote": "string"}]},
    },
    "task_outcome": {
        "version": "v3",
        "instruction": ("判断当前 Turn 中用户目标的实际完成结果。Agent 自称完成不等于完成，优先使用工具结果、"
                        "产物验证和明确的后续用户反馈。NEXT_TURN_USER_FOLLOW_UP 只用于判断当前 Turn，"
                        "但后续新需求、追问、‘继续’或工作范围扩展不能自动证明当前 Turn 未完成。"
                        "completed 表示当前 Turn 的明确要求均有结果支持；partially_completed 表示只完成部分明确要求；"
                        "failed 表示核心目标明确未达成；证据不足时必须为 unclear。"
                        "verified=true 仅限工具验证、产物检查或用户明确确认，不能依据 Agent 自述。"
                        "每条 evidence.quote 必须是对应事件中的短小逐字原文，不得概括、改写、拼接或用省略号代替内容。"),
        "schema": {"outcome": "completed|partially_completed|failed|unclear", "verified": "boolean|unclear",
                   "unmet_parts": ["string"], "evidence": [{"event_id": "string", "quote": "string"}]},
    },
    "difficulty": {
        "version": "v2",
        "instruction": ("分别判断两个互不替代的维度。intrinsic_difficulty 是用户任务本身对一个能力正常的 Coding Agent 的"
                        "客观要求：easy=单步、范围明确且几乎不需外部信息；medium=多步骤或需要工具、领域知识、若干约束；"
                        "hard=跨文件/多来源综合、复杂产物、强约束验证或长期研究。observed_difficulty 是本次执行实际遇到的阻力："
                        "low=过程顺畅且无明显错误重试；medium=存在有限重试、澄清或局部受阻；high=反复失败、严重阻塞、"
                        "大量返工或核心执行中断。Agent 表现差、轨迹很长、工具很多都不能单独提高固有难度；"
                        "任务本身很难也不能自动说明本次执行困难。证据不足时标 unclear。factors 分别写清任务因素和执行因素。"
                        "每条 evidence.quote 必须是对应事件中的短小逐字原文，不得概括、改写或使用省略号。"),
        "schema": {"intrinsic_difficulty": "easy|medium|hard|unclear",
                   "observed_difficulty": "low|medium|high|unclear", "factors": ["string"],
                   "evidence": [{"event_id": "string", "quote": "string"}]},
    },
    "behavior_quality": {
        "version": "v2",
        "instruction": ("只评价当前 Turn 中 Agent 的执行行为是否值得模仿，分别检查：计划是否与任务规模相称、工具选择是否合适、"
                        "关键结果是否验证、遇错是否诊断纠正、步骤是否避免明显浪费、操作是否安全。"
                        "exemplary=各关键方面均有明确优秀证据且无实质缺陷；good=整体可靠但有轻微遗漏；"
                        "mixed=同时存在有意义的优点和缺点；poor=存在严重且可避免的错误、无验证、危险操作或明显无效循环；"
                        "证据不足时为 unclear。任务完成不自动代表行为优秀，任务未完成不自动代表行为差；"
                        "用户满意度、任务难度、回复长度、工具数量均不能替代行为判断。合理澄清或等待用户材料不算缺点。"
                        "strengths 和 weaknesses 只能写轨迹中可观察的行为。每条 evidence.quote 必须是对应事件中的短小逐字原文，"
                        "不得概括、改写或使用省略号。"),
        "schema": {"quality": "exemplary|good|mixed|poor|unclear", "strengths": ["string"],
                   "weaknesses": ["string"], "evidence": [{"event_id": "string", "quote": "string"}]},
    },
    "impact": {
        "version": "v1",
        "instruction": "判断需求未满足时对用户目标的影响严重性；只依据轨迹证据，不推测业务损失。",
        "schema": {"severity": "low|medium|high|critical|unclear", "affected_goal": "string",
                   "evidence": [{"event_id": "string", "quote": "string"}]},
    },
    "demand_capability_instances": {
        "version": "v4",
        "instruction": ("开放式抽取用户需求和原子能力实例，不得从预设类别中强制选择。每个实例只描述一个可独立验证的需求；"
                        "只抽取用户明确提出的需求，或完成用户目标不可缺少的隐含需求；不得把 Agent 主动增加的解释、风险提醒、"
                        "后续建议或可选操作当成用户需求。区分用户需要什么、当前 Agent 是否满足、若未满足具体缺少什么能力。"
                        "Agent 的 reasoning 可用于理解上下文，但不能单独证明用户提出了该需求；Agent 自称完成也不能单独证明已满足，"
                        "满足判断优先依据工具结果、产物验证或用户反馈。只使用给定事件中的证据。"
                        "每个实例的requirement_evidence必须至少包含一条user_message事件中的用户原文，禁止用assistant_message、"
                        "reasoning、tool_call或tool_result证明用户提出了需求。即使是implicit_necessary，也必须由用户原始目标直接支持。"),
        "schema": {"user_goal": "string", "instances": [{
            "need": "string", "object": "string", "expected_outcome": "string", "constraints": ["string"],
            "requirement_source": "explicit_user|implicit_necessary",
            "fulfillment": "met|partially_met|unmet|unclear", "agent_gap": "string",
            "candidate_capability": "string",
            "requirement_evidence": [{"event_id": "string", "quote": "string"}],
            "fulfillment_evidence": [{"event_id": "string", "quote": "string"}],
        }]},
    },
    "requirement_negotiation": {
        "version": "v2",
        "instruction": (
            "识别当前Turn中的用户初始目标、Agent明确提出的工作计划项，以及NEXT_TURN_USER_FOLLOW_UP对这些计划项的回应。"
            "不得把Agent计划自动视为用户需求。plan_items只记录Agent明确提议要执行的具体步骤，每项使用本次输出内唯一的"
            "local_id。responses只能依据下一Turn用户原文，response_type只能是explicit_acceptance、behavioral_acceptance、"
            "partial_acceptance、modified、rejected、not_confirmed；accepted_item_ids和rejected_item_ids只能引用plan_items.local_id。"
            "明确的‘可以、按这个做、开始吧、继续’可接受全部或语境明确的计划项；用户实际提供计划所请求的材料或执行下一步"
            "可构成behavioral_acceptance，但只能接受其实际推进的部分。沉默、转移话题、Agent自行执行均为not_confirmed。"
            "plan_items必须是Agent面向未来明确提议执行的工作；已经在当前Turn完成的内容、结果标题和纯总结不是计划项。"
            "可选提议只有在用户明确选择时才被接受；用户提出另一项新需求不等于拒绝未选择的计划项，应标not_confirmed。"
            "goal_evidence必须来自当前Turn的user_message；proposal_evidence必须来自当前Turn的assistant_message；"
            "response_evidence必须来自NEXT_TURN_USER_FOLLOW_UP的user_message。所有quote必须逐字摘录。"),
        "schema": {
            "user_goals": [{"description": "string", "evidence": [{"event_id": "string", "quote": "string"}]}],
            "plan_items": [{"local_id": "string", "description": "string",
                            "proposal_evidence": [{"event_id": "string", "quote": "string"}]}],
            "responses": [{"response_type": "explicit_acceptance|behavioral_acceptance|partial_acceptance|modified|rejected|not_confirmed",
                           "accepted_item_ids": ["string"], "rejected_item_ids": ["string"],
                           "response_evidence": [{"event_id": "string", "quote": "string"}],
                           "reason": "string"}]
        },
    },
    "ml_llm_coding_scope": {
        "version": "v1",
        "instruction": ("判断当前Turn的用户需求是否实质属于科研ML/LLM Coding。included仅指需要编写、修改、调试、"
                        "运行或验证机器学习/深度学习/大模型相关代码、数据管线、训练、微调、推理、评测、部署、"
                        "实验复现或计算分析。纯写作排版、普通文献综述、领域问答、研究叙事、通用软件开发和仅有概念"
                        "讨论而无计算实现需求均为excluded；证据不足为uncertain。不要因出现AI、模型或数据等词就纳入。"),
        "schema": {"scope": "included|excluded|uncertain", "reason": "string",
                   "confidence": "number", "evidence": [{"event_id": "string", "quote": "string"}]},
    },
}

REGISTRY_FIELDS = ["feature_run_id", "feature_set", "feature_version", "input_batch", "processed_at", "record_count",
                   "error_count", "provider", "model", "prompt_version", "output_path", "status"]


def redact_for_api(value: str) -> str:
    return redact_text(value)


def _event_rows(events: list[dict[str, Any]]) -> list[str]:
    rows = []
    for event in events:
        kind = event.get("type")
        if kind not in {"user_message", "assistant_message", "reasoning", "tool_call", "tool_result", "error"}:
            continue
        data = event.get("data")
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        rows.append(f'[{event.get("event_id")}] {kind}: {text}')
    return rows


def next_turn_user_events(turns: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        by_session.setdefault(str(turn.get("session_id")), []).append(turn)
    result: dict[str, list[dict[str, Any]]] = {}
    for session_turns in by_session.values():
        session_turns.sort(key=lambda item: int(item.get("turn_index") or 0))
        for index, turn in enumerate(session_turns):
            follow_up = [] if index + 1 >= len(session_turns) else [
                event for event in session_turns[index + 1].get("events") or []
                if event.get("type") == "user_message"
            ]
            result[str(turn.get("turn_id"))] = follow_up
    return result


def is_negotiation_candidate(turn: dict[str, Any], follow_up_events: list[dict[str, Any]]) -> bool:
    """Cheap recall-oriented filter; it never decides whether the user accepted a plan."""
    if not follow_up_events:
        return False
    assistant_text = "\n".join(
        searchable_event_text(event.get("data")) for event in turn.get("events") or []
        if event.get("type") == "assistant_message"
    )
    if not assistant_text.strip():
        return False
    cues = re.compile(
        r"(?:计划|步骤|阶段|接下来|下一步|首先|然后|我会|我们会|将会|按以下|工作流|"
        r"plan|steps?|phases?|next|first|then|I\s+will|we\s+will|workflow|"
        r"(?:^|\n)\s*(?:[-*]|\d+[.)、]))",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return bool(cues.search(assistant_text))


def compact_turn(turn: dict[str, Any], max_chars: int = 24000,
                 follow_up_events: list[dict[str, Any]] | None = None) -> str:
    current = redact_for_api("\n".join(_event_rows(turn.get("events") or [])))
    if follow_up_events is None:
        value = current
    else:
        follow_up = redact_for_api("\n".join(_event_rows(follow_up_events)))
        suffix = ("\n\n[NEXT_TURN_USER_FOLLOW_UP]\n" + follow_up if follow_up
                  else "\n\n[NO_NEXT_TURN_USER_FOLLOW_UP]")
        prefix = "[CURRENT_TURN]\n"
        available = max_chars - len(prefix) - len(suffix)
        if len(current) > available:
            marker = "\n[...当前Turn中间内容因长度限制省略，保留开头、结尾和后续反馈...]\n"
            half = max(0, (available - len(marker)) // 2)
            current = current[:half] + marker + current[-half:]
        value = prefix + current + suffix
    if len(value) <= max_chars:
        return value
    half = (max_chars - 80) // 2
    return value[:half] + "\n[...中间内容因长度限制省略，保留开头和结尾...]\n" + value[-half:]


ENUMS = {
    "user_feedback": {"sentiment": {"positive", "negative", "mixed", "none", "unclear"}},
    "task_outcome": {"outcome": {"completed", "partially_completed", "failed", "unclear"}},
    "difficulty": {"intrinsic_difficulty": {"easy", "medium", "hard", "unclear"},
                   "observed_difficulty": {"low", "medium", "high", "unclear"}},
    "behavior_quality": {"quality": {"exemplary", "good", "mixed", "poor", "unclear"}},
    "impact": {"severity": {"low", "medium", "high", "critical", "unclear"}},
    "demand_capability_instances": {},
    "requirement_negotiation": {},
    "ml_llm_coding_scope": {"scope": {"included", "excluded", "uncertain"}},
}


def validate(task_name: str, value: dict[str, Any], event_ids: set[str],
             event_texts: dict[str, str] | None = None) -> None:
    required = set(TASKS[task_name]["schema"])
    missing = required - set(value)
    if missing:
        raise ValueError(f"Missing model output fields: {sorted(missing)}")
    if task_name == "demand_capability_instances":
        if not isinstance(value.get("instances"), list):
            raise ValueError("instances must be a list")
        required_instance = {"need", "object", "expected_outcome", "constraints", "requirement_source",
                             "fulfillment", "agent_gap", "candidate_capability", "requirement_evidence",
                             "fulfillment_evidence"}
        invalid_evidence: list[str | None] = []
        for instance in value["instances"]:
            if not isinstance(instance, dict) or not required_instance.issubset(instance):
                raise ValueError("Invalid demand capability instance")
            if instance["fulfillment"] not in {"met", "partially_met", "unmet", "unclear"}:
                raise ValueError(f"Invalid fulfillment: {instance['fulfillment']!r}")
            if instance["requirement_source"] not in {"explicit_user", "implicit_necessary"}:
                raise ValueError(f"Invalid requirement_source: {instance['requirement_source']!r}")
            for evidence_field in ("requirement_evidence", "fulfillment_evidence"):
                if not isinstance(instance[evidence_field], list):
                    raise ValueError(f"{evidence_field} must be a list")
                invalid_evidence.extend(item.get("event_id") for item in instance[evidence_field]
                                        if not isinstance(item, dict) or item.get("event_id") not in event_ids)
        if invalid_evidence:
            raise ValueError(f"Evidence references events outside this Turn: {invalid_evidence}")
        return
    if task_name == "requirement_negotiation":
        for field in ("user_goals", "plan_items", "responses"):
            if not isinstance(value.get(field), list):
                raise ValueError(f"{field} must be a list")
        local_ids = [str(row.get("local_id")) for row in value["plan_items"] if isinstance(row, dict)]
        if len(local_ids) != len(set(local_ids)) or any(not item for item in local_ids):
            raise ValueError("plan_items.local_id must be non-empty and unique")
        allowed_ids = set(local_ids)
        valid_response_types = {"explicit_acceptance", "behavioral_acceptance", "partial_acceptance",
                                "modified", "rejected", "not_confirmed"}
        evidence_rows = []
        for goal in value["user_goals"]:
            if not goal.get("evidence"):
                raise ValueError("Every negotiation user goal requires evidence")
            evidence_rows.extend(goal.get("evidence") or [])
        for item in value["plan_items"]:
            if not item.get("proposal_evidence"):
                raise ValueError("Every negotiation plan item requires proposal evidence")
            evidence_rows.extend(item.get("proposal_evidence") or [])
        for response in value["responses"]:
            if response.get("response_type") not in valid_response_types:
                raise ValueError("Invalid negotiation response_type")
            references = set(map(str, response.get("accepted_item_ids") or [])) | set(
                map(str, response.get("rejected_item_ids") or []))
            if not references.issubset(allowed_ids):
                raise ValueError("Negotiation response references unknown plan item")
            if response.get("response_type") != "not_confirmed" and not response.get("response_evidence"):
                raise ValueError("Confirmed negotiation response requires user evidence")
            if response.get("response_type") == "not_confirmed" and references:
                raise ValueError("not_confirmed response cannot accept or reject plan items")
            if response.get("response_type") in {"explicit_acceptance", "behavioral_acceptance",
                                                 "partial_acceptance", "modified"} and not response.get("accepted_item_ids"):
                raise ValueError("Accepted or modified response requires accepted_item_ids")
            if response.get("response_type") == "rejected" and not response.get("rejected_item_ids"):
                raise ValueError("Rejected response requires rejected_item_ids")
            evidence_rows.extend(response.get("response_evidence") or [])
        invalid = [row.get("event_id") for row in evidence_rows
                   if not isinstance(row, dict) or row.get("event_id") not in event_ids]
        if invalid:
            raise ValueError(f"Negotiation evidence references unknown events: {invalid}")
        if event_texts is not None:
            inexact = [row.get("event_id") for row in evidence_rows
                       if not isinstance(row.get("quote"), str) or not row["quote"].strip()
                       or row["quote"].strip() not in event_texts.get(str(row.get("event_id")), "")]
            if inexact:
                raise ValueError(f"Negotiation evidence quotes are not exact excerpts: {inexact}")
        return
    if not isinstance(value.get("evidence"), list):
        raise ValueError("evidence must be a list")
    for field, allowed in ENUMS.get(task_name, {}).items():
        if value.get(field) not in allowed:
            raise ValueError(f"Invalid {field}: {value.get(field)!r}")
    invalid_evidence = [item.get("event_id") for item in value["evidence"]
                        if not isinstance(item, dict) or item.get("event_id") not in event_ids]
    if invalid_evidence:
        raise ValueError(f"Evidence references events outside this Turn: {invalid_evidence}")
    if event_texts is not None:
        inexact = [item.get("event_id") for item in value["evidence"]
                   if not isinstance(item.get("quote"), str) or not item["quote"].strip()
                   or item["quote"].strip() not in event_texts.get(str(item.get("event_id")), "")]
        if inexact:
            raise ValueError(f"Evidence quotes are not exact event excerpts: {inexact}")
    if task_name == "user_feedback":
        explicit_sentiments = {"positive", "negative", "mixed"}
        if value.get("sentiment") in explicit_sentiments:
            if value.get("explicit") is not True or not value["evidence"]:
                raise ValueError("Explicit positive/negative/mixed feedback requires explicit=true and evidence")
        elif value.get("explicit") is True:
            raise ValueError("none/unclear feedback cannot have explicit=true")
        if value.get("sentiment") == "none" and value["evidence"]:
            raise ValueError("none feedback must not contain evidence")


def normalize_model_value(task_name: str, value: dict[str, Any]) -> dict[str, Any]:
    required = set(TASKS[task_name]["schema"])
    wrapped = value.get(task_name)
    if not required.issubset(value) and isinstance(wrapped, dict):
        return wrapped
    if task_name == "task_outcome" and not required.issubset(value) and isinstance(value.get("outcome"), dict):
        return value["outcome"]
    return value


def searchable_event_text(value: Any) -> str:
    """Return redacted serialized data plus nested strings for exact-quote checks."""
    strings: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    serialized = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    )
    return "\n".join(redact_for_api(text) for text in [serialized, *strings])


def add_instance_ids(turn_id: str, values: dict[str, Any]) -> dict[str, Any]:
    for instance in values.get("instances") or []:
        instance["instance_id"] = stable_id(
            "capinst", turn_id, instance.get("need"), instance.get("object"),
            instance.get("expected_outcome"), instance.get("requirement_source")
        )
    return values


def add_negotiation_ids(turn_id: str, values: dict[str, Any]) -> dict[str, Any]:
    local_to_stable = {}
    for item in values.get("plan_items") or []:
        local_id = str(item["local_id"])
        item_id = stable_id("planitem", turn_id, item.get("description"), local_id)
        local_to_stable[local_id] = item_id
        item["item_id"] = item_id
    for response in values.get("responses") or []:
        response["accepted_item_ids"] = [local_to_stable[str(value)]
                                         for value in response.get("accepted_item_ids") or []]
        response["rejected_item_ids"] = [local_to_stable[str(value)]
                                         for value in response.get("rejected_item_ids") or []]
        response["response_id"] = stable_id(
            "reqresponse", turn_id, response.get("response_type"),
            response.get("accepted_item_ids"), response.get("rejected_item_ids"))
    return values


def run_semantic_features(batch_dir: Path, output_root: Path, task_name: str, client: JSONClient,
                          provider_name: str, limit: int | None = None,
                          max_input_chars: int = 24000, workers: int = 8,
                          evidence_repair: bool = False) -> dict[str, Any]:
    task = TASKS[task_name]
    turn_path = batch_dir / "unified_turns.jsonl"
    now = dt.datetime.now().astimezone()
    suffix = hashlib.sha256(f"{turn_path.resolve()}|{task_name}|{task['version']}|{client.model}".encode()).hexdigest()[:8]
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}_{suffix}"
    run_dir = output_root / task_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "features.jsonl"
    errors = run_dir / "errors.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    count = error_count = 0
    reused_files = compatible_feature_files(
        output_root, task_name, task["version"], provider_name, client.model,
        input_batch=batch_dir.name,
    )
    completed = completed_turn_ids(reused_files)
    skipped = 0
    prompt_version = task["version"] + ("-evidence-repair-v1" if evidence_repair else "")
    system = ("你是 Coding Agent Trace 数据标注器。必须返回一个 JSON 对象，不要输出 Markdown。"
              f"任务：{task['instruction']} 输出结构：{json.dumps(task['schema'], ensure_ascii=False)}")
    if evidence_repair and task_name == "demand_capability_instances":
        system += (
            " 本次是失败记录的证据修复重试。requirement_evidence.event_id 必须引用当前 Turn 中真实存在的"
            " user_message 事件；quote 必须直接从该事件 data.content 中连续逐字复制，禁止概括、改写、"
            "修正标点或使用省略号。fulfillment_evidence 也必须逐字复制对应事件原文。"
            "如果当前 Turn 只有系统提醒、工具结果或中断标记，没有可证实的用户需求，返回 user_goal 为空字符串、"
            "instances 为空数组；不要强行生成能力实例。"
        )
    with turn_path.open(encoding="utf-8") as source:
        turns = [json.loads(line) for line in source if line.strip()]
    contextual_tasks = {"user_feedback", "task_outcome", "requirement_negotiation"}
    follow_ups = next_turn_user_events(turns) if task_name in contextual_tasks else {}
    manifest = {
        "feature_run_id": run_id, "feature_set": task_name, "feature_version": task["version"],
        "input_batch": batch_dir.name, "provider": provider_name, "model": client.model,
        "prompt_version": prompt_version,
        "output_path": str(output.resolve()), "errors_path": str(errors.resolve()),
        "status": "running", "record_count": 0, "error_count": 0,
        "skipped_existing_count": skipped, "updated_at": now.isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    started_at = time.monotonic()
    last_progress_at = started_at
    pending_turns = []
    for turn in turns:
        turn_id = str(turn.get("turn_id"))
        if task_name == "requirement_negotiation" and not is_negotiation_candidate(
                turn, follow_ups.get(turn_id) or []):
            skipped += 1
        elif turn_id in completed:
            skipped += 1
        elif limit is None or len(pending_turns) < limit:
            pending_turns.append(turn)
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = []
    for turn in pending_turns:
        contextual_events = (follow_ups.get(str(turn.get("turn_id")))
                             if task_name in contextual_tasks else None)
        futures.append(executor.submit(
            client.complete_json, system, compact_turn(turn, max_input_chars, contextual_events)
        ))
    try:
      with output.open("w", encoding="utf-8") as target, errors.open("w", encoding="utf-8") as error_out:
        for turn, future in zip(pending_turns, futures):
            try:
                contextual_events = (follow_ups.get(str(turn.get("turn_id")))
                                     if task_name in contextual_tasks else None)
                values, api_meta = future.result()
                values = normalize_model_value(task_name, values)
                valid_events = (list(contextual_events or []) if task_name == "user_feedback" else
                                list(turn.get("events") or []) + list(contextual_events or []))
                event_texts = {}
                event_types = {}
                for event in valid_events:
                    event_texts[str(event.get("event_id"))] = searchable_event_text(event.get("data"))
                    event_types[str(event.get("event_id"))] = str(event.get("type"))
                if task_name == "demand_capability_instances":
                    dropped = 0
                    dropped_instances = 0
                    downgraded_fulfillment = 0
                    original_instances = values.get("instances") or []
                    valid_instances = []
                    for instance in original_instances:
                        for field in ("requirement_evidence", "fulfillment_evidence"):
                            evidence = instance.get(field) if isinstance(instance.get(field), list) else []
                            exact = [item for item in evidence if isinstance(item, dict)
                                     and item.get("event_id") in event_texts
                                     and isinstance(item.get("quote"), str) and item["quote"].strip()
                                     and item["quote"].strip() in event_texts[item["event_id"]]
                                     and (field != "requirement_evidence"
                                          or event_types.get(str(item.get("event_id"))) == "user_message")]
                            dropped += len(evidence) - len(exact)
                            instance[field] = exact
                        if not instance["requirement_evidence"]:
                            dropped_instances += 1
                            continue
                        if (instance.get("fulfillment") != "unclear"
                                and not instance["fulfillment_evidence"]):
                            instance["fulfillment"] = "unclear"
                            instance["agent_gap"] = "精确证据不足，无法判断本次是否满足"
                            downgraded_fulfillment += 1
                        valid_instances.append(instance)
                    if original_instances and not valid_instances:
                        raise ValueError("All capability instances lack exact requirement evidence")
                    values["instances"] = valid_instances
                    if dropped or dropped_instances or downgraded_fulfillment:
                        api_meta = {
                            **api_meta,
                            "dropped_inexact_evidence_count": dropped,
                            "dropped_ungrounded_instance_count": dropped_instances,
                            "downgraded_fulfillment_count": downgraded_fulfillment,
                        }
                if task_name == "requirement_negotiation":
                    follow_up_ids = {str(row.get("event_id")) for row in contextual_events or []}
                    evidence_fields = (("user_goals", "evidence", "user_message", None),
                                       ("plan_items", "proposal_evidence", "assistant_message", None),
                                       ("responses", "response_evidence", "user_message", follow_up_ids))
                    for collection, evidence_field, expected_type, allowed_event_ids in evidence_fields:
                        for item in values.get(collection) or []:
                            evidence = item.get(evidence_field) if isinstance(item.get(evidence_field), list) else []
                            item[evidence_field] = [row for row in evidence if isinstance(row, dict)
                                                    and row.get("event_id") in event_texts
                                                    and event_types.get(str(row.get("event_id"))) == expected_type
                                                    and (allowed_event_ids is None
                                                         or str(row.get("event_id")) in allowed_event_ids)
                                                    and isinstance(row.get("quote"), str) and row["quote"].strip()
                                                    and row["quote"].strip() in event_texts[row["event_id"]]]
                    values["user_goals"] = [row for row in values.get("user_goals") or []
                                             if row.get("evidence")]
                    original_plan_count = len(values.get("plan_items") or [])
                    values["plan_items"] = [row for row in values.get("plan_items") or []
                                            if row.get("proposal_evidence")]
                    valid_local_ids = {str(row.get("local_id")) for row in values["plan_items"]}
                    downgraded_responses = 0
                    for response in values.get("responses") or []:
                        response["accepted_item_ids"] = [str(value) for value in
                            response.get("accepted_item_ids") or [] if str(value) in valid_local_ids]
                        response["rejected_item_ids"] = [str(value) for value in
                            response.get("rejected_item_ids") or [] if str(value) in valid_local_ids]
                        response_type = response.get("response_type")
                        if response_type == "not_confirmed":
                            response["accepted_item_ids"] = []
                            response["rejected_item_ids"] = []
                        elif response_type in {"explicit_acceptance", "behavioral_acceptance",
                                               "partial_acceptance", "modified"} and not response["accepted_item_ids"]:
                            response["response_type"] = "not_confirmed"
                            response["rejected_item_ids"] = []
                            downgraded_responses += 1
                        elif response_type == "rejected" and not response["rejected_item_ids"]:
                            response["response_type"] = "not_confirmed"
                            response["accepted_item_ids"] = []
                            downgraded_responses += 1
                    if original_plan_count != len(values["plan_items"]) or downgraded_responses:
                        api_meta = {**api_meta,
                                    "dropped_plan_items_without_exact_assistant_evidence":
                                        original_plan_count - len(values["plan_items"]),
                                    "downgraded_unlinked_responses": downgraded_responses}
                if task_name in {"task_outcome", "difficulty", "behavior_quality", "impact",
                                 "ml_llm_coding_scope"}:
                    evidence = values.get("evidence") if isinstance(values.get("evidence"), list) else []
                    exact_evidence = [item for item in evidence if isinstance(item, dict)
                                      and item.get("event_id") in event_texts
                                      and isinstance(item.get("quote"), str) and item["quote"].strip()
                                      and item["quote"].strip() in event_texts[item["event_id"]]]
                    dropped = len(evidence) - len(exact_evidence)
                    values["evidence"] = exact_evidence
                    if dropped:
                        api_meta = {**api_meta, "dropped_inexact_evidence_count": dropped}
                    needs_evidence = (
                        task_name == "task_outcome" and values.get("outcome") != "unclear"
                    ) or (
                        task_name == "difficulty" and
                        (values.get("intrinsic_difficulty") != "unclear"
                         or values.get("observed_difficulty") != "unclear")
                    ) or (
                        task_name == "behavior_quality" and values.get("quality") != "unclear"
                    ) or (
                        task_name == "impact" and values.get("severity") != "unclear"
                    ) or (
                        task_name == "ml_llm_coding_scope" and values.get("scope") != "uncertain"
                    )
                    if task_name == "impact" and needs_evidence and not exact_evidence:
                        values["severity"] = "unclear"
                        values["affected_goal"] = str(values.get("affected_goal") or "")
                        api_meta = {**api_meta, "downgraded_missing_exact_evidence": True}
                    elif task_name == "ml_llm_coding_scope" and needs_evidence and not exact_evidence:
                        values["scope"] = "uncertain"
                        values["reason"] = str(values.get("reason") or "缺少可核验的逐字证据")
                        api_meta = {**api_meta, "downgraded_missing_exact_evidence": True}
                    elif needs_evidence and not exact_evidence:
                        raise ValueError(f"{task_name} has no exact evidence after filtering")
                validate(task_name, values, set(event_texts), event_texts)
                if task_name == "demand_capability_instances":
                    values = add_instance_ids(str(turn.get("turn_id")), values)
                elif task_name == "requirement_negotiation":
                    values = add_negotiation_ids(str(turn.get("turn_id")), values)
                record = {"trace_id": turn.get("trace_id"), "session_id": turn.get("session_id"), "turn_id": turn.get("turn_id"),
                          "feature_set": task_name, "feature_version": task["version"], "feature_run_id": run_id,
                          "generated_by": {"method": "llm", "provider": provider_name, "model": client.model,
                                           "prompt_version": prompt_version}, "values": values, "api_meta": api_meta}
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                target.flush()
                count += 1
            except Exception as exc:
                error_out.write(json.dumps({"turn_id": turn.get("turn_id"), "error": str(exc)}, ensure_ascii=False) + "\n")
                error_out.flush()
                error_count += 1
            processed = count + error_count
            if processed % 25 == 0:
                os.fsync(target.fileno())
                os.fsync(error_out.fileno())
                manifest.update({"record_count": count, "error_count": error_count,
                                 "updated_at": dt.datetime.now().astimezone().isoformat()})
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            now_monotonic = time.monotonic()
            if processed and (processed % 25 == 0 or now_monotonic - last_progress_at >= 30):
                elapsed = max(now_monotonic - started_at, 0.001)
                rate = processed / elapsed
                remaining = max(0, len(pending_turns) - processed)
                eta_seconds = round(remaining / rate) if rate else None
                print(json.dumps({"feature_run_id": run_id, "input_batch": batch_dir.name,
                                  "completed": count, "errors": error_count, "skipped": skipped,
                                  "rate_per_minute": round(rate * 60, 2), "eta_seconds": eta_seconds},
                                 ensure_ascii=False), file=sys.stderr, flush=True)
                last_progress_at = now_monotonic
        target.flush()
        error_out.flush()
        os.fsync(target.fileno())
        os.fsync(error_out.fileno())
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    status = "success" if not error_count else ("partial" if count else "failed")
    manifest.update({"status": status, "record_count": count, "error_count": error_count,
                     "skipped_existing_count": skipped,
                     "updated_at": dt.datetime.now().astimezone().isoformat()})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    row = {"feature_run_id": run_id, "feature_set": task_name, "feature_version": task["version"],
           "input_batch": batch_dir.name, "processed_at": now.isoformat(), "record_count": count,
           "error_count": error_count, "provider": provider_name, "model": client.model,
           "prompt_version": prompt_version, "output_path": str(output.resolve()), "status": status}
    registry = output_root / "feature_registry.csv"
    existing_header = registry.exists()
    if existing_header:
        with registry.open(encoding="utf-8-sig") as handle:
            existing_header = next(csv.reader(handle), []) == REGISTRY_FIELDS
    registry_path = registry if existing_header or not registry.exists() else output_root / "semantic_feature_registry.csv"
    write_header = not registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    registry_record = write_immutable_record(
        "feature-runs", run_id, row, artifacts=(output, errors, manifest_path),
        root=registry_root_for_output(output_root),
    )
    return {**row, "skipped_existing_count": skipped,
            "reused_feature_files": [str(path.resolve()) for path in reused_files],
            "registry_record_path": str(registry_record.resolve())}


def configured_client(config_path: Path, provider: str, model: str | None, cache_dir: Path,
                      timeout_seconds: int | None = None,
                      max_retries: int | None = None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(load_provider(config_path, provider), model=model, cache_dir=cache_dir,
                                  timeout_seconds=timeout_seconds, max_retries=max_retries)


def write_dry_run(batch_dir: Path, output_root: Path, task_name: str, limit: int | None = None,
                  max_input_chars: int = 24000) -> dict[str, Any]:
    task = TASKS[task_name]
    turn_path = batch_dir / "unified_turns.jsonl"
    now = dt.datetime.now().astimezone()
    run_id = f"dry_run_{now.strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = output_root / task_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "requests.jsonl"
    system = ("你是 Coding Agent Trace 数据标注器。必须返回一个 JSON 对象，不要输出 Markdown。"
              f"任务：{task['instruction']} 输出结构：{json.dumps(task['schema'], ensure_ascii=False)}")
    count = 0
    with turn_path.open(encoding="utf-8") as source:
        turns = [json.loads(line) for line in source if line.strip()]
    contextual_tasks = {"user_feedback", "task_outcome", "requirement_negotiation"}
    follow_ups = next_turn_user_events(turns) if task_name in contextual_tasks else {}
    with output.open("w", encoding="utf-8") as target:
        for turn in turns:
            if task_name == "requirement_negotiation" and not is_negotiation_candidate(
                    turn, follow_ups.get(str(turn.get("turn_id"))) or []):
                continue
            if limit is not None and count >= limit:
                continue
            target.write(json.dumps({"turn_id": turn.get("turn_id"), "system": system,
                                     "user": compact_turn(
                                         turn, max_input_chars,
                                         follow_ups.get(str(turn.get("turn_id")))
                                         if task_name in contextual_tasks else None,
                                     )},
                                    ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return {"dry_run_id": run_id, "feature_set": task_name, "request_count": count,
            "output_path": str(output.resolve()), "status": "dry_run"}
