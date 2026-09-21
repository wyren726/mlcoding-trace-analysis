from __future__ import annotations

import json
from typing import Any


PROMPT_VERSION = "trace-user-screen-rubrics-v1"
AGENT_ATTRIBUTION_PROMPT_VERSION = "trace-agent-verification-v4"


# The first call sees real user messages only. It reports each rubric explicitly;
# routing is omitted because local code derives it deterministically.
USER_ANALYSIS_OUTPUT_SHAPE: dict[str, Any] = {
    "trace_id": "string",
    "task_episodes": [
        {
            "episode_boundary": {
                "judgment": "reliable|uncertain",
                "reason": "string",
                "start_turn_id": "string",
                "end_turn_id": "string",
            },
            "domain": {
                "judgment": "in_scope|adjacent|out_of_scope|uncertain",
                "reason": "string",
                "evidence_turn_ids": ["string"],
            },
            "analysis_value": {
                "judgment": "high|medium|low|uncertain",
                "reason": "string",
                "evidence_turn_ids": ["string"],
            },
            "initial_query": {
                "judgment": "identified|uncertain|not_identifiable",
                "turn_id": "string|null",
                "reason": "string",
            },
            "initial_query_clarity": {
                "judgment": "clear|partly_clear|unclear",
                "reason": "string",
                "evidence_turn_ids": ["string"],
            },
            "preliminary_goal": {
                "judgment": "complete|partial|insufficient",
                "reason": "string",
                "goal": "string",
                "requirements": [
                    {
                        "text": "string",
                        "origin": "initial_query|clarified_later|new_later",
                        "evidence": [
                            {"turn_id": "string", "event_id": "string", "quote": "string"}
                        ],
                    }
                ],
            },
            "requirement_evolution": {
                "judgment": "stable|evolved|uncertain",
                "reason": "string",
                "changes": [
                    {
                        "turn_id": "string",
                        "type": "clarification|correction|new_requirement",
                        "text": "string",
                        "evidence": {"event_id": "string", "quote": "string"},
                    }
                ],
            },
            "candidate_signals": {
                "judgment": "present|absent|uncertain",
                "reason": "string",
                "signals": [
                    {
                        "type": (
                            "explicit_rejection|result_defect|constraint_repeated|"
                            "redo_or_rollback|retry_after_failure|user_takeover|"
                            "blocked_or_abandoned"
                        ),
                        "turn_id": "string",
                        "event_id": "string",
                        "quote": "string",
                    }
                ],
            },
        }
    ],
}


USER_ANALYSIS_SYSTEM_PROMPT = f"""
你是 Coding/ML Agent 真实用户轨迹的用户侧筛选员。输入只包含一条 Trace 中按时间排列的真实用户消息，
故意不包含 Agent 回复、工具结果、系统指令、内部推理和模型身份。请返回一个 JSON 对象，不要输出
Markdown 或 JSON 之外的文字。

本阶段的边界：
- 只生成用户侧筛选判断和 preliminary_goal（初步需求假设）。
- 不能判断 Agent 是否失败，不能确认用户痛点，不能输出能力缺口。
- 用户完整真实需求可能受到 Agent 提问和回复影响，必须在后续读取完整 Episode 后确认。

先识别连续用户目标，再按以下 Rubrics 顺序分析每个候选 Episode：

R1 目标领域：
- in_scope：代码理解/实现/调试/测试，ML/LLM 数据、训练、评估、推理、系统、Agent 工作流，
  实验设计、算法研究或复现。
- adjacent：与上述工作密切相关的技术方案、结果分析或架构讨论。
- out_of_scope：闲聊、普通写作、无关事实问答、平台内容。
- uncertain：仅凭用户消息不能安全判断。

R2 分析与构题价值：
- high：非平凡、真实、原则上可验证，可能构成能区分 Agent 能力的任务。
- medium：真实且有一定难度，但验收条件或 Workspace 信息暂时不完整。
- low：机械、极简单、没有可执行目标或即使获得 Workspace 也难以评价。
- uncertain：信息不足，但不能安全判 low。交互短或文本短本身不能作为 low 的理由。

R3 Episode 边界：同一目标的背景补充、澄清、纠正、重试和返工属于同一 Episode；只有切换到具有
独立交付物的新目标时才拆分。不要按所需能力拆分。输出边界判断、理由和首尾用户 Turn。

R4 初始 Query：定位该 Episode 第一次提出可执行核心任务的用户 Turn。背景消息不一定是 Query；后来
最完整的说法也不应替代最初 Query。

R5 初始 Query 清晰度：只使用 Query 当时已有的用户上下文。
- clear：核心目标、对象和主要约束足以合理开始任务。
- partly_clear：核心目标明确，但缺少会显著影响执行且应向用户确认的信息。
- unclear：核心目标无法识别，或必须依赖后来真正新增的要求才能理解。
Workspace 中本应由 Agent 自己读取的信息缺失，不自动意味着用户 Query 不清晰。

R6 初步目标还原：综合当前可见的用户消息，写 preliminary_goal。它只是等待完整 Episode 验证的需求
假设，不得表述为最终确认事实。requirements 描述用户想得到的结果或交付约束，不写 Agent 的工具、
Skill、计划或自由选择的步骤。origin 含义：initial_query=当时已经明确；clarified_later=后来解释当时
已隐含的要求；new_later=后来真正新增，不能反推成 Agent 一开始就知道。

R7 需求变化：
- clarification：解释原来已经隐含的目标；
- correction：原要求已经存在，用户因为当前方向可能偏离而重新强调；
- new_requirement：后来真正增加或改变范围。
整体无变化为 stable，有有效变化为 evolved，无法判断为 uncertain。

R8 用户侧候选信号：只识别值得后续读取 Agent 行为验证的信号：explicit_rejection、result_defect、
constraint_repeated、redo_or_rollback、retry_after_failure、user_takeover、blocked_or_abandoned。
普通补充、回答澄清问题、新增要求、中性追问、正常推进、“继续/下一步/再看看”以及单纯没听懂，均不
自动构成候选信号。只能说可能存在问题，不能归因给 Agent。

证据规则：
- 只能引用输入中存在的 turn_id 和 event_id。
- quote 必须是对应用户消息中的短小连续原文，不得拼接、改写或概括。
- 每个 requirement 至少有一条用户证据；每个 change 和 signal 必须有直接用户原文。
- 每项 Rubric 必须给出简短、具体的 reason；信息不足时使用相应 uncertain/insufficient，不要猜测。

输出结构：
{json.dumps(USER_ANALYSIS_OUTPUT_SHAPE, ensure_ascii=False)}
""".strip()


AGENT_ATTRIBUTION_OUTPUT_SHAPE: dict[str, Any] = {
    "trace_id": "string",
    "evidence_review": {
        "judgment": "sufficient|insufficient",
        "reason": "string",
        "missing_evidence_queries": ["string"],
    },
    "episode_attributions": [
        {
            "query_turn_id": "string",
            "goal_assessment": {
                "judgment": "confirmed|revised|insufficient",
                "goal": "string",
                "reason": "string",
                "evidence": [
                    {"turn_id": "string", "event_id": "string", "quote": "string"}
                ],
            },
            "requirement_results": [
                {
                    "requirement_index": "integer",
                    "status": "met|partially_met|unmet|unknown",
                    "evidence": [
                        {"turn_id": "string", "event_id": "string", "quote": "string"}
                    ],
                }
            ],
            "confirmed_requirements": [
                {
                    "text": "string",
                    "origin": "initial_query|clarified_later|new_later",
                    "origin_evidence": [
                        {"turn_id": "string", "event_id": "string", "quote": "string"}
                    ],
                    "status": "met|partially_met|unmet|unknown",
                    "status_evidence": [
                        {"turn_id": "string", "event_id": "string", "quote": "string"}
                    ],
                }
            ],
            "outcome": "completed|partially_completed|failed|unknown",
            "user_interruptions": [
                {
                    "interruption_event": {
                        "turn_id": "string",
                        "event_id": "string",
                        "quote": "string",
                    },
                    "judgment": (
                        "dissatisfaction_signal|task_control|accidental_or_unknown"
                    ),
                    "reason": "string",
                    "followup_evidence": [
                        {"turn_id": "string", "event_id": "string", "quote": "string"}
                    ],
                }
            ],
            "agent_failure": "string|null",
            "agent_related": "boolean|null",
            "agent_related_reason": "string",
            "attribution": (
                "reasoning_or_response|execution_or_verification|tool_use|"
                "context_handling|permission|external|user_input|unknown"
            ),
            "capability_gaps": [
                {
                    "capability_key": "lower_snake_case string",
                    "capability_name": "concise Chinese string",
                    "reason": "string",
                    "evidence": [
                        {"turn_id": "string", "event_id": "string", "quote": "string"}
                    ],
                }
            ],
        }
    ],
}


_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "turn_id": {"type": "string"},
        "event_id": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["turn_id", "event_id", "quote"],
    "additionalProperties": False,
}

AGENT_ATTRIBUTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trace_id": {"type": "string"},
        "evidence_review": {
            "type": "object",
            "properties": {
                "judgment": {"type": "string", "enum": ["sufficient", "insufficient"]},
                "reason": {"type": "string"},
                "missing_evidence_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["judgment", "reason", "missing_evidence_queries"],
            "additionalProperties": False,
        },
        "episode_attributions": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "query_turn_id": {"type": "string"},
                    "goal_assessment": {
                        "type": "object",
                        "properties": {
                            "judgment": {"type": "string", "enum": ["confirmed", "revised", "insufficient"]},
                            "goal": {"type": "string"}, "reason": {"type": "string"},
                            "evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
                        },
                        "required": ["judgment", "goal", "reason", "evidence"],
                        "additionalProperties": False,
                    },
                    "requirement_results": {
                        "type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "requirement_index": {"type": "integer"},
                                "status": {"type": "string", "enum": ["met", "partially_met", "unmet", "unknown"]},
                                "evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
                            },
                            "required": ["requirement_index", "status", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                    "confirmed_requirements": {
                        "type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "origin": {"type": "string", "enum": ["initial_query", "clarified_later", "new_later"]},
                                "origin_evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
                                "status": {"type": "string", "enum": ["met", "partially_met", "unmet", "unknown"]},
                                "status_evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
                            },
                            "required": ["text", "origin", "origin_evidence", "status", "status_evidence"],
                            "additionalProperties": False,
                        },
                    },
                    "outcome": {"type": "string", "enum": ["completed", "partially_completed", "failed", "unknown"]},
                    "user_interruptions": {
                        "type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "interruption_event": _EVIDENCE_SCHEMA,
                                "judgment": {"type": "string", "enum": ["dissatisfaction_signal", "task_control", "accidental_or_unknown"]},
                                "reason": {"type": "string"},
                                "followup_evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
                            },
                            "required": ["interruption_event", "judgment", "reason", "followup_evidence"],
                            "additionalProperties": False,
                        },
                    },
                    "agent_failure": {"type": ["string", "null"]},
                    "agent_related": {"type": ["boolean", "null"]},
                    "agent_related_reason": {"type": "string"},
                    "attribution": {"type": "string", "enum": ["reasoning_or_response", "execution_or_verification", "tool_use", "context_handling", "permission", "external", "user_input", "unknown"]},
                    "capability_gaps": {
                        "type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "capability_key": {"type": "string"},
                                "capability_name": {"type": "string"},
                                "reason": {"type": "string"},
                                "evidence": {"type": "array", "items": _EVIDENCE_SCHEMA},
                            },
                            "required": ["capability_key", "capability_name", "reason", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["query_turn_id", "goal_assessment", "requirement_results", "confirmed_requirements", "outcome", "user_interruptions", "agent_failure", "agent_related", "agent_related_reason", "attribution", "capability_gaps"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["trace_id", "evidence_review", "episode_attributions"],
    "additionalProperties": False,
}


AGENT_ATTRIBUTION_SYSTEM_PROMPT = f"""
你是科研全流程 Agent 候选 Episode 验证员。输入包含用户侧 Rubrics、preliminary_goal、02 阶段确定的
完整 Episode 边界内的可观察 Agent 回复、工具调用、工具结果、错误和用户消息。输入还可能包含 Episode
结束后的一个真实用户 Turn，字段 post_episode_context_turn_id 会明确标记；它只能用于验证用户是否已经
查看、评价或继续修改交付物，不能反推成当前 Episode 的新增要求。输入不包含内部推理、Token 遥测或
模型/Harness 身份。请返回 JSON，不要输出其他文字。

	任务：
1. 结合 Agent 的澄清问题、回复和用户回应，确认或修订 preliminary_goal。goal_assessment=confirmed
   表示用户侧初步目标成立；revised 表示完整交互提供了必须修正的证据；insufficient 表示仍无法确认。
2. 按 preliminary_goal.requirements 数组从 0 开始的 requirement_index 判断要求是否满足。
   如果该数组为空，必须结合完整Episode提取confirmed_requirements；每条要求区分initial_query、
   clarified_later和new_later，并分别给出用户来源证据及完成状态证据。最多8条，不得把Agent自行选择的
   工具、步骤或计划写成用户要求。
3. 判断用户侧候选信号是否对应真实的 Agent 失败，而不是正常澄清、正常迭代、新增要求、权限限制、
   外部故障或用户输入不足。
   输入中的user_interruption表示可验证的用户主动打断行为，不是普通用户Query，也不能直接等同于Agent
   自行截断。必须逐个输出user_interruptions：结合打断前Agent正在做什么、打断后的首个真实用户Turn判断
   它更符合dissatisfaction_signal、task_control还是accidental_or_unknown，并给出打断事件及后续用户原文证据。
   仅有打断事件、没有后续语义证据时，不得凭打断本身推断具体不满意原因。
4. agent_related=true 必须有可验证证据表明 Agent 未满足当时清晰的要求、违反约束，或者在回答、执行、
   工具使用、上下文处理、结果验证上出错。
	在判定“未响应、未执行、未生成或中断”前，必须检查 Episode 末端用户指令之后的全部可观察事件，
	包括 Assistant 回复、工具调用、工具结果和 Artifact；并检查 post_episode_context_turn_id 是否表明
	用户已经查看、评价或继续修改交付物。只要存在成功执行或交付证据，就不得作出上述结论。
	若存在user_interruption，必须区分“Agent在被用户打断前已经发生可验证失败”和“回复仅因用户主动
	打断而不完整”；后一种情况不能把输出不完整本身归因成Agent能力缺陷，但可以依据打断后的用户原文
	识别另一个有证据支持的具体痛点。
	5. agent_related=true 时必须填写 agent_failure、agent_related_reason 和至少一个 capability_gap；能力
	   缺口同时输出稳定英文snake_case capability_key和简洁中文capability_name；否则
	   使用 false 或 null，并返回空 capability_gaps。
	6. evidence_review判断本次可观察证据是否足以支持痛点或非痛点结论。若insufficient，必须用
	   missing_evidence_queries具体说明还需要查找什么；不能因为证据不足直接判定agent_related=false。

goal_assessment=confirmed/revised 必须引用支持目标确认或修订的可观察证据。所有证据只能引用输入中
存在的 turn_id/event_id，quote 必须是对应事件中的连续原文。信息不足时使用 unknown、null 或
insufficient，不要猜测。

输出结构：
{json.dumps(AGENT_ATTRIBUTION_OUTPUT_SHAPE, ensure_ascii=False)}
""".strip()


def user_analysis_prompt(payload: dict[str, Any]) -> str:
    return (
        "请仅根据以下真实用户消息逐项完成 R1-R8 Rubrics。\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def agent_attribution_prompt(payload: dict[str, Any]) -> str:
    return (
        "请根据以下用户侧候选和完整可观察 Episode 进行需求确认与 Agent 归因。\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


__all__ = [
    "AGENT_ATTRIBUTION_OUTPUT_SHAPE", "AGENT_ATTRIBUTION_JSON_SCHEMA", "AGENT_ATTRIBUTION_PROMPT_VERSION",
    "AGENT_ATTRIBUTION_SYSTEM_PROMPT", "PROMPT_VERSION",
    "USER_ANALYSIS_OUTPUT_SHAPE", "USER_ANALYSIS_SYSTEM_PROMPT",
    "agent_attribution_prompt", "user_analysis_prompt",
]
