from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from trace_analysis.pipeline.stage_02_analyze.runner import (
    TraceBundle, analyze_trace, analyze_user_trace, candidate_agent_payload,
    episode_partition_issues, full_trace_payload,
    complete_index_partition_with_fallback, merge_agent_attribution,
    normalise_user_analysis, run_trace_analysis,
    user_trace_payload,
)
from trace_analysis.pipeline.stage_02_analyze.compact import (
    COMPACT_USER_OUTPUT_SHAPE, COMPACT_USER_SYSTEM_PROMPT, indexed_candidate_payload,
    deterministic_strong_evidence_ids, evidence_selection_prompt,
    normalise_compact_user_analysis, normalise_selected_event_ids,
    partition_indexed_candidate_payload,
    selected_evidence_payload,
)
from trace_analysis.pipeline.stage_02_analyze.repair import repair_trace_analysis
from trace_analysis.pipeline.stage_03_export import export_trace_analysis
from trace_analysis.pipeline.status import pipeline_status


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _turn(turn_id: str, turn_index: int, events: list[dict]) -> dict:
    return {
        "schema_version": "v1.0",
        "batch_id": "batch_test",
        "trace_id": "trace_1",
        "session_id": "session_1",
        "turn_id": turn_id,
        "turn_index": turn_index,
        "status": "completed",
        "harness": "test_harness",
        "agent_models": ["test_model"],
        "events": events,
    }


def _event(event_id: str, kind: str, content: str, sequence: int) -> dict:
    return {
        "event_id": event_id,
        "sequence": sequence,
        "timestamp": f"2026-01-01T00:00:0{sequence}Z",
        "type": kind,
        "data": {"content": content},
    }


def _bundle() -> TraceBundle:
    return TraceBundle("batch_test", "trace_1", (
        _turn("turn_1", 1, [
            _event("event_query", "user_message", "请把结果保存成 JSON。", 1),
            _event("event_answer", "assistant_message", "结果已经保存成 CSV。", 2),
        ]),
        _turn("turn_2", 2, [
            _event("event_feedback", "user_message", "不对，我明确要求的是 JSON。", 3),
        ]),
    ))


def _multi_turn_bundle(contents: list[str]) -> TraceBundle:
    return TraceBundle("batch_test", "trace_1", tuple(
        _turn(f"turn_{index}", index, [
            _event(f"event_{index}", "user_message", content, index),
        ])
        for index, content in enumerate(contents, 1)
    ))


def _user_analysis() -> dict:
    return {
        "trace_id": "trace_1",
        "task_episodes": [{
            "episode_boundary": {
                "judgment": "reliable", "reason": "同一输出任务",
                "start_turn_id": "turn_1", "end_turn_id": "turn_2",
            },
            "domain": {
                "judgment": "in_scope", "reason": "结构化文件输出任务",
                "evidence_turn_ids": ["turn_1"],
            },
            "analysis_value": {
                "judgment": "medium", "reason": "格式约束可以验证",
                "evidence_turn_ids": ["turn_1", "turn_2"],
            },
            "initial_query": {
                "judgment": "identified", "turn_id": "turn_1",
                "reason": "首次提出输出任务",
            },
            "initial_query_clarity": {
                "judgment": "clear", "reason": "用户明确指定 JSON",
                "evidence_turn_ids": ["turn_1"],
            },
            "preliminary_goal": {
                "judgment": "complete", "reason": "目标和格式都明确",
                "goal": "把结果保存成 JSON",
                "requirements": [{
                    "text": "输出 JSON 文件",
                    "origin": "initial_query",
                    "evidence": [
                        {"turn_id": "turn_1", "event_id": "event_query", "quote": "保存成 JSON"},
                    ],
                }],
            },
            "requirement_evolution": {
                "judgment": "evolved", "reason": "用户重申原格式约束",
                "changes": [{
                    "turn_id": "turn_2", "type": "correction",
                    "text": "重申必须输出 JSON",
                    "evidence": {"event_id": "event_feedback", "quote": "明确要求的是 JSON"},
                }],
            },
            "candidate_signals": {
                "judgment": "present", "reason": "存在明确纠错",
                "signals": [{
                    "type": "explicit_rejection", "turn_id": "turn_2",
                    "event_id": "event_feedback", "quote": "不对，我明确要求的是 JSON",
                }],
            },
        }],
    }


def _attribution() -> dict:
    return {
        "trace_id": "trace_1",
        "evidence_review": {
            "judgment": "sufficient", "reason": "证据足够",
            "missing_evidence_queries": [],
        },
        "episode_attributions": [{
            "query_turn_id": "turn_1",
            "goal_assessment": {
                "judgment": "confirmed", "goal": "把结果保存成 JSON",
                "reason": "完整交互确认格式要求没有变化",
                "evidence": [{
                    "turn_id": "turn_1", "event_id": "event_query",
                    "quote": "保存成 JSON",
                }],
            },
            "requirement_results": [{
                "requirement_index": 0,
                "status": "unmet",
                "evidence": [{
                    "turn_id": "turn_1", "event_id": "event_answer",
                    "quote": "保存成 CSV",
                }],
            }],
            "outcome": "failed",
            "agent_failure": "输出成 CSV",
            "agent_related": True,
            "agent_related_reason": "Agent 违反明确格式约束",
            "attribution": "execution_or_verification",
            "capability_gaps": [{
                "capability": "遵循输出格式约束",
                "reason": "没有按用户明确指定的 JSON 输出",
                "evidence": [{
                    "turn_id": "turn_1", "event_id": "event_answer",
                    "quote": "保存成 CSV",
                }],
            }],
        }],
    }


def _no_pain_analysis() -> dict:
    value = _user_analysis()
    episode = value["task_episodes"][0]
    episode["candidate_signals"] = {
        "judgment": "absent", "reason": "没有候选信号", "signals": [],
    }
    return value


def _screen_episode(start: str, end: str, query: str, goal: str) -> dict:
    episode = deepcopy(_no_pain_analysis()["task_episodes"][0])
    episode["episode_boundary"].update({
        "start_turn_id": start,
        "end_turn_id": end,
        "reason": "回归测试边界",
    })
    episode["initial_query"].update({
        "judgment": "identified",
        "turn_id": query,
    })
    episode["domain"]["evidence_turn_ids"] = [query]
    episode["analysis_value"]["evidence_turn_ids"] = [query]
    episode["initial_query_clarity"]["evidence_turn_ids"] = [query]
    episode["preliminary_goal"].update({
        "goal": goal,
        "requirements": [],
    })
    episode["requirement_evolution"] = {
        "judgment": "stable", "reason": "未变化", "changes": [],
    }
    episode["candidate_signals"] = {
        "judgment": "absent", "reason": "没有候选信号", "signals": [],
    }
    return episode


class FakeClient:
    model = "fake-model"
    config = SimpleNamespace(name="fake")

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> tuple[dict, dict]:
        self.calls.append((system, user))
        return self.responses.pop(0), {"usage": {"total_tokens": 10}}


def test_user_screen_candidate_is_verified_in_a_second_call() -> None:
    client = FakeClient([_user_analysis(), _attribution()])
    result = analyze_trace(_bundle(), client, "run_test", 1_000_000, 1_000_000)

    assert result.strategy == "compact_user_screen_then_direct_agent_verification"
    assert result.model_call_count == 2
    assert result.record["analysis_status"] == "complete"
    episode = result.record["task_episodes"][0]
    assert episode["route"]["judgment"] == "analyze"
    assert episode["pain_confirmed"] is True
    assert episode["agent_assessment"]["agent_related"] is True
    assert "结果已经保存成 CSV" not in client.calls[0][1]
    assert "结果已经保存成 CSV" in client.calls[1][1]
    assert len(client.calls) == 2


def test_analysis_view_separates_platform_wrapper_from_actual_user_message() -> None:
    wrapped = _turn("turn_1", 1, [
        _event(
            "event_query", "user_message",
            "Platform instruction that is not user demand\n\n### User message\n\n真正的用户问题",
            1,
        )
    ])
    payload = full_trace_payload(TraceBundle("batch_test", "trace_1", (wrapped,)))

    assert payload["turns"][0]["user_message"]["data"]["content"] == "真正的用户问题"


def test_user_view_excludes_skill_payload_encoded_as_later_user_message() -> None:
    turn = _turn("turn_1", 1, [
        _event("event_query", "user_message", "真正的用户问题", 1),
        _event(
            "event_skill", "user_message",
            "Base directory for this skill: <LOCAL_PATH>\n内部 Skill 指令",
            2,
        ),
    ])
    payload = full_trace_payload(TraceBundle("batch_test", "trace_1", (turn,)))

    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["user_message"]["event_id"] == "event_query"
    assert "内部 Skill 指令" not in json.dumps(payload, ensure_ascii=False)


def test_user_view_unwraps_legacy_platform_prompt_and_omits_synthetic_turns() -> None:
    wrapped = _turn("turn_1", 1, [_event(
        "event_query", "user_message",
        "You are Wismodel, a model developed by AtomInfinite.ai.\n\n"
        "Load and follow the `universal-agent-orchestrator` skill to begin.\n\n"
        "真正问题",
        1,
    )])
    skill = _turn("turn_2", 2, [_event(
        "event_skill", "user_message",
        "Base directory for this skill: /internal\n内部 Skill 指令",
        2,
    )])
    reminder = _turn("turn_3", 3, [_event(
        "event_reminder", "user_message",
        "[System reminder] This is a runtime system reminder, not user input. Do not answer it.",
        3,
    )])
    relayed = _turn("turn_4", 4, [_event(
        "event_relay", "user_message",
        "[用户提问] 用户发来了以下消息。请先回复。\n\n用户消息：继续完成报告",
        4,
    )])

    payload = full_trace_payload(TraceBundle(
        "batch_test", "trace_1", (wrapped, skill, reminder, relayed)
    ))

    assert [turn["turn_id"] for turn in payload["turns"]] == ["turn_1", "turn_4"]
    assert payload["turns"][0]["user_message"]["data"]["content"] == "真正问题"
    assert payload["turns"][1]["user_message"]["data"]["content"] == "继续完成报告"


def test_analysis_view_does_not_expose_model_or_harness_metadata() -> None:
    event = _event("event_query", "user_message", "用户问题", 1)
    event["data"].update({"harness": "hidden-harness", "model": "hidden-model"})
    payload = full_trace_payload(TraceBundle(
        "batch_test", "trace_1", (_turn("turn_1", 1, [event]),)
    ))

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "hidden-harness" not in serialized
    assert "hidden-model" not in serialized


def test_trace_without_candidate_signal_stops_after_user_only_call() -> None:
    client = FakeClient([_no_pain_analysis()])
    result = analyze_trace(_bundle(), client, "run_test", 1, 1_000_000)

    assert result.strategy == "compact_user_screen_only"
    assert result.model_call_count == 1
    assert result.record["task_episodes"][0]["agent_assessment"] is None
    assert result.record["task_episodes"][0]["route"]["judgment"] == "review"
    assert len(client.calls) == 1


def test_episode_partition_validator_detects_gap_overlap_and_context_start() -> None:
    bundle = _multi_turn_bundle([
        "完成第一项分析",
        "补充第一项参数",
        "好呢 然后给我这部分内容",
        "继续完善这部分",
    ])
    payload = user_trace_payload(bundle)
    raw = {
        "trace_id": "trace_1",
        "task_episodes": [
            _screen_episode("turn_1", "turn_2", "turn_1", "完成第一项分析"),
            _screen_episode("turn_2", "turn_4", "turn_3", "完成第二项分析"),
        ],
    }

    issues = episode_partition_issues(raw, payload)

    assert any(issue.startswith("structural:overlap_turns:1") for issue in issues)
    assert "semantic:episode_1:context_dependent_start" not in issues
    raw["task_episodes"][1]["episode_boundary"]["start_turn_id"] = "turn_3"
    issues = episode_partition_issues(raw, payload)
    assert "semantic:episode_1:context_dependent_start" in issues


def test_invalid_compact_partition_is_repaired_once_before_normalization() -> None:
    bundle = _multi_turn_bundle([
        "完成第一项分析", "补充第一项参数",
        "开始第二项实验", "补充第二项参数",
    ])
    invalid = {
        "trace_id": "trace_1",
        "task_episodes": [
            _screen_episode("turn_1", "turn_2", "turn_1", "完成第一项分析"),
            _screen_episode("turn_4", "turn_4", "turn_4", "完成第二项实验"),
        ],
    }
    repaired = {
        "trace_id": "trace_1",
        "task_episodes": [
            _screen_episode("turn_1", "turn_2", "turn_1", "完成第一项分析"),
            _screen_episode("turn_3", "turn_4", "turn_3", "完成第二项实验"),
        ],
    }
    client = FakeClient([invalid, repaired])

    result = analyze_user_trace(bundle, client, "run_test", 1_000_000)

    assert result.strategy == "compact_user_screen_boundary_repair"
    assert result.model_call_count == 2
    assert result.record["analysis_status"] == "complete"
    assert [
        (episode["episode_boundary"]["start_turn_id"],
         episode["episode_boundary"]["end_turn_id"])
        for episode in result.record["task_episodes"]
    ] == [("turn_1", "turn_2"), ("turn_3", "turn_4")]
    provenance = result.record["analysis_provenance"]
    assert provenance["boundary_repair_attempted"] is True
    assert provenance["boundary_fallback_used"] is False
    assert provenance["final_boundary_issues"] == []
    assert "structural:uncovered_turns:1:turn_3" in client.calls[1][1]


def test_structurally_invalid_repair_falls_back_to_full_trace_review() -> None:
    bundle = _multi_turn_bundle([
        "完成第一项分析", "补充第一项参数",
        "开始第二项实验", "补充第二项参数",
    ])
    invalid = {
        "trace_id": "trace_1",
        "task_episodes": [
            _screen_episode("turn_1", "turn_2", "turn_1", "完成第一项分析"),
            _screen_episode("missing_turn", "turn_4", "turn_4", "完成第二项实验"),
        ],
    }
    client = FakeClient([invalid, deepcopy(invalid)])

    result = analyze_user_trace(bundle, client, "run_test", 1_000_000)

    assert result.strategy == "compact_user_screen_boundary_fallback"
    assert result.model_call_count == 2
    assert result.record["analysis_status"] == "needs_review"
    assert result.record["route"]["judgment"] == "review"
    assert len(result.record["task_episodes"]) == 1
    boundary = result.record["task_episodes"][0]["episode_boundary"]
    assert boundary["judgment"] == "uncertain"
    assert (boundary["start_turn_id"], boundary["end_turn_id"]) == (
        "turn_1", "turn_4",
    )
    assert result.record["analysis_provenance"]["boundary_fallback_used"] is True


def test_normalizer_downgrades_reliable_but_overlapping_partition() -> None:
    bundle = _multi_turn_bundle(["任务一", "任务一补充", "任务二"])
    raw = {
        "trace_id": "trace_1",
        "task_episodes": [
            _screen_episode("turn_1", "turn_2", "turn_1", "任务一"),
            _screen_episode("turn_2", "turn_3", "turn_3", "任务二"),
        ],
    }

    record = normalise_user_analysis(raw, bundle, "run_test")

    assert record["analysis_status"] == "needs_review"
    assert all(
        episode["episode_boundary"]["judgment"] == "uncertain"
        for episode in record["task_episodes"]
    )
    assert any(
        warning.startswith("episode_partition:structural:overlap_turns:1")
        for warning in record["validation_warnings"]
    )


def test_route_is_computed_from_rubrics_not_model_route() -> None:
    raw = _user_analysis()
    raw["task_episodes"][0]["route"] = {
        "judgment": "skip", "reason": "模型不应控制路由",
    }

    record = normalise_user_analysis(raw, _bundle(), "run_test")

    assert record["task_episodes"][0]["route"]["judgment"] == "analyze"
    assert record["route"]["judgment"] == "analyze"


def test_explicit_low_value_rubric_is_skipped_even_with_signal() -> None:
    raw = _user_analysis()
    raw["task_episodes"][0]["analysis_value"] = {
        "judgment": "low", "reason": "明确是无研究价值的机械请求",
        "evidence_turn_ids": ["turn_1"],
    }

    record = normalise_user_analysis(raw, _bundle(), "run_test")

    assert record["task_episodes"][0]["route"]["judgment"] == "skip"
    assert record["route"]["judgment"] == "skip"


def test_trace_with_only_synthetic_user_messages_skips_model_call() -> None:
    synthetic = _turn("turn_1", 1, [_event(
        "event_reminder", "user_message",
        "[System reminder] This is a runtime system reminder, not user input. Do not answer it.",
        1,
    )])
    client = FakeClient([])

    result = analyze_trace(
        TraceBundle("batch_test", "trace_1", (synthetic,)),
        client, "run_test", 1_000_000, 1_000_000,
    )

    assert result.strategy == "no_real_user_turns"
    assert result.model_call_count == 0
    assert result.record["task_episodes"] == []
    assert client.calls == []


def test_compact_shape_omits_expensive_requirement_expansion() -> None:
    episode = COMPACT_USER_OUTPUT_SHAPE["task_episodes"][0]

    assert "requirements" not in episode["preliminary_goal"]
    assert "changes" not in episode["requirement_evolution"]


def test_compact_scope_covers_cross_disciplinary_research_workflow() -> None:
    assert "跨学科科研全流程" in COMPACT_USER_SYSTEM_PROMPT
    assert "质性分析" in COMPACT_USER_SYSTEM_PROMPT
    assert "不要求必须写代码" in COMPACT_USER_SYSTEM_PROMPT
    assert "构题价值由analysis_value判断" in COMPACT_USER_SYSTEM_PROMPT
    assert "每个Turn恰好出现一次" in COMPACT_USER_SYSTEM_PROMPT
    assert "好呢" in COMPACT_USER_SYSTEM_PROMPT
    assert "initial_query.turn_id必须位于本Episode" in COMPACT_USER_SYSTEM_PROMPT


def test_oversized_tool_body_is_indexed_then_restored_only_when_selected() -> None:
    huge = "large-tool-output-marker-" * 20_000
    bundle = TraceBundle("batch_test", "trace_1", (
        _turn("turn_1", 1, [
            _event("event_query", "user_message", "请把结果保存成 JSON。", 1),
            _event("event_answer", "assistant_message", "结果已经保存成 CSV。", 2),
            {
                "event_id": "event_tool_call", "sequence": 3,
                "timestamp": "2026-01-01T00:00:03Z", "type": "tool_call",
                "data": {"call_id": "call_1", "tool_name": "read_file", "path": "x.log"},
            },
            {
                "event_id": "event_tool_result", "sequence": 4,
                "timestamp": "2026-01-01T00:00:04Z", "type": "tool_result",
                "data": {"call_id": "call_1", "content": huge},
            },
        ]),
        _turn("turn_2", 2, [
            _event("event_feedback", "user_message", "不对，我明确要求的是 JSON。", 5),
        ]),
    ))
    screen = normalise_compact_user_analysis(_user_analysis(), bundle, "run_test")
    candidate = candidate_agent_payload(bundle, screen)
    indexed = indexed_candidate_payload(candidate)
    serialized_index = json.dumps(indexed, ensure_ascii=False)

    assert huge not in serialized_index
    assert "sha256" in serialized_index
    assert "retrieval_summary" in serialized_index
    chunk_id = "event_tool_result#chunk_0000"
    selected = normalise_selected_event_ids(
        {"selected_event_ids": [chunk_id, "not-real"]}, candidate
    )
    restored = selected_evidence_payload(candidate, selected)
    serialized_restored = json.dumps(restored, ensure_ascii=False)
    assert huge[:10_000] in serialized_restored
    assert len(serialized_restored) < len(huge) // 4
    assert "event_tool_call" in serialized_restored  # paired by call_id
    assert selected == [chunk_id]


def test_agent_payload_uses_complete_episode_boundary_and_outcome_lookahead() -> None:
    """A candidate signal must not cut off a later execution in the Episode."""
    bundle = TraceBundle("batch_test", "trace_1", (
        _turn("turn_1", 1, [
            _event("event_query", "user_message", "请生成温场图。", 1),
        ]),
        _turn("turn_2", 2, [
            _event("event_signal", "user_message", "数据齐了，可以直接生成吗？", 2),
        ]),
        _turn("turn_3", 3, [
            _event("event_final_request", "user_message", "你直接生成给我。", 3),
            _event("event_tool", "tool_result", "图片已生成：figure.png", 4),
            _event("event_answer", "assistant_message", "已经生成并保存。", 5),
        ]),
        _turn("turn_4", 4, [
            _event("event_review", "user_message", "图例再调整一下。", 6),
        ]),
    ))
    screen = normalise_compact_user_analysis(_user_analysis(), bundle, "run_test")
    episode = screen["task_episodes"][0]
    episode["episode_boundary"].update({
        "start_turn_id": "turn_1", "end_turn_id": "turn_3",
    })
    episode["initial_query"]["turn_id"] = "turn_1"
    episode["candidate_signals"]["signals"] = [{
        "type": "retry_after_failure", "turn_id": "turn_2",
        "event_id": "event_signal", "quote": "可以直接生成吗",
    }]
    episode["route"] = {"judgment": "analyze", "reason": "回归测试候选"}

    payload = candidate_agent_payload(bundle, screen)
    turn_ids = [turn["turn_id"] for turn in payload["episode_turns"]]
    assert turn_ids == ["turn_1", "turn_2", "turn_3", "turn_4"]
    assert payload["episode_evidence_spans"] == [{
        "episode_id": episode["episode_id"],
        "start_turn_id": "turn_1",
        "end_turn_id": "turn_3",
        "post_episode_context_turn_id": "turn_4",
    }]
    visible = json.dumps(payload, ensure_ascii=False)
    assert "图片已生成：figure.png" in visible
    assert "图例再调整一下" in visible


def test_oversized_candidate_uses_three_call_indexed_strategy() -> None:
    huge = "large-tool-output-marker-" * 1_000
    bundle = TraceBundle("batch_test", "trace_1", (
        _turn("turn_1", 1, [
            _event("event_query", "user_message", "请把结果保存成 JSON。", 1),
            _event("event_answer", "assistant_message", "结果已经保存成 CSV。", 2),
            {
                "event_id": "event_tool_result", "sequence": 3,
                "timestamp": "2026-01-01T00:00:03Z", "type": "tool_result",
                "data": {"call_id": "call_1", "content": huge},
            },
        ]),
        _turn("turn_2", 2, [
            _event("event_feedback", "user_message", "不对，我明确要求的是 JSON。", 4),
        ]),
    ))
    client = FakeClient([
        _user_analysis(),
        {"trace_id": "trace_1", "selected_event_ids": [
            "event_tool_result#chunk_0000"
         ],
         "reason": "需要检查工具产物"},
        _attribution(),
    ])

    result = analyze_trace(bundle, client, "run_test", 1_000, 1_000_000)

    assert result.strategy == "compact_user_screen_then_indexed_agent_verification"
    assert result.model_call_count == 3
    assert result.record["analysis_provenance"]["selected_tool_event_ids"] == [
        "event_tool_result#chunk_0000"
    ]
    assert huge not in client.calls[1][1]
    assert huge[:10_000] in client.calls[2][1]
    assert len(client.calls[2][1]) < len(huge)


def test_chunk_summary_finds_salient_text_in_the_middle() -> None:
    middle = "archive is empty and the result file is missing"
    huge = "x" * 5_000 + middle + "y" * 15_000
    candidate = {
        "trace_id": "trace_1", "candidate_episodes": [],
        "episode_turns": [{
            "turn_id": "turn_1", "events": [{
                "event_id": "event_tool_result", "type": "tool_result",
                "data": {"content": huge},
            }],
        }],
    }

    indexed = indexed_candidate_payload(candidate)
    first_chunk = indexed["episode_turns"][0]["events"][0]["data"]["chunks"][0]
    summary = json.dumps(first_chunk["retrieval_summary"], ensure_ascii=False)

    assert "archive is empty" in summary
    assert "missing" in summary


def test_deterministic_selector_keeps_error_event() -> None:
    candidate = {
        "trace_id": "trace_1", "candidate_episodes": [],
        "episode_turns": [{"turn_id": "turn_1", "events": [
            {"event_id": "ok", "type": "tool_result", "data": {"content": "done"}},
            {"event_id": "bad", "type": "tool_result", "data": {
                "content": "AttributeError: missing field"
            }},
        ]}],
    }
    assert deterministic_strong_evidence_ids(candidate) == ["bad"]


def test_deterministic_selector_keeps_user_referenced_numeric_config() -> None:
    candidate = {
        "trace_id": "trace_1", "candidate_episodes": [],
        "episode_turns": [
            {"turn_id": "turn_1", "events": [{
                "event_id": "query", "type": "user_message",
                "data": {"content": "不要限制到4096，至少应该是8192"},
            }]},
            {"turn_id": "turn_2", "events": [
                {"event_id": "relevant", "type": "tool_result",
                 "data": {"content": "MAX_PROMPT_LENGTH=4096"}},
                {"event_id": "unrelated", "type": "tool_result",
                 "data": {"content": "BATCH_SIZE=128"}},
            ]},
        ],
    }
    assert deterministic_strong_evidence_ids(candidate) == ["relevant"]


def test_index_partitions_preserve_all_chunk_ids_under_prompt_limit() -> None:
    candidate = {
        "trace_id": "trace_1", "candidate_episodes": [],
        "episode_turns": [{"turn_id": "turn_1", "events": [
            {"event_id": f"event_{index}", "type": "tool_result", "data": {
                "content": ("x" * 20_000) + str(index)
            }} for index in range(8)
        ]}],
    }
    indexed = indexed_candidate_payload(candidate)
    partitions = partition_indexed_candidate_payload(indexed, max_prompt_chars=20_000)
    original = {
        chunk["chunk_id"]
        for turn in indexed["episode_turns"] for event in turn["events"]
        for chunk in event["data"].get("chunks", [])
    }
    partitioned = {
        chunk["chunk_id"]
        for part in partitions for turn in part["episode_turns"]
        for event in turn["events"] for chunk in event["data"].get("chunks", [])
    }
    assert len(partitions) > 1
    assert original == partitioned
    assert all(len(evidence_selection_prompt(part)) <= 20_000 for part in partitions)


def test_failing_large_index_partition_is_subdivided_without_truncation() -> None:
    parent = {
        "trace_id": "trace_1",
        "input_view": "candidate_episode_event_index_partition",
        "candidate_episodes": [],
        "episode_turns": [{"turn_id": "turn_1", "events": [
            {"event_id": f"event_{index}", "type": "tool_result", "data": {
                "externalized": True, "preview": ("x" * 6_000) + str(index)
            }} for index in range(5)
        ]}],
        "partition": {"index": 1, "count": 1, "selection_limit": 5},
    }

    class RejectLargeClient:
        def __init__(self) -> None:
            self.prompt_lengths: list[int] = []

        def complete_json(self, system: str, user: str) -> tuple[dict, dict]:
            self.prompt_lengths.append(len(user))
            if len(user) > 25_000:
                raise TimeoutError("synthetic provider timeout")
            return {"selected_event_ids": [], "reason": "checked"}, {"usage": {}}

    client = RejectLargeClient()
    results = complete_index_partition_with_fallback(client, parent)
    original_event_ids = {
        event["event_id"]
        for turn in parent["episode_turns"] for event in turn["events"]
    }
    recovered_event_ids = {
        event["event_id"]
        for partition, _, _ in results
        for turn in partition["episode_turns"] for event in turn["events"]
    }

    assert client.prompt_lengths[0] > 25_000
    assert len(results) > 1
    assert all(len(evidence_selection_prompt(partition)) <= 25_000
               for partition, _, _ in results)
    assert recovered_event_ids == original_event_ids


def test_event_summary_surfaces_configuration_assignment_in_the_middle() -> None:
    candidate = {
        "trace_id": "trace_1", "candidate_episodes": [],
        "episode_turns": [{
            "turn_id": "turn_1", "events": [{
                "event_id": "event_tool_result", "type": "tool_result",
                "data": {"call_id": "call_1", "content": (
                    "header\n" + "x" * 300 + "\nMAX_PROMPT_LENGTH=4096\n" + "y" * 300
                )},
            }],
        }],
    }

    indexed = indexed_candidate_payload(candidate)
    summary = indexed["episode_turns"][0]["events"][0]["data"]["retrieval_summary"]

    assert "MAX_PROMPT_LENGTH=4096" in summary["structured_facts"]
    assert summary["line_count"] == 5


def test_insufficient_indexed_evidence_triggers_one_followup_retrieval() -> None:
    huge = "large-tool-output-marker-" * 1_000
    bundle = TraceBundle("batch_test", "trace_1", (
        _turn("turn_1", 1, [
            _event("event_query", "user_message", "请把结果保存成 JSON。", 1),
            _event("event_answer", "assistant_message", "结果已经保存成 CSV。", 2),
            {
                "event_id": "event_tool_result", "sequence": 3,
                "timestamp": "2026-01-01T00:00:03Z", "type": "tool_result",
                "data": {"call_id": "call_1", "content": huge},
            },
        ]),
        _turn("turn_2", 2, [
            _event("event_feedback", "user_message", "不对，我明确要求的是 JSON。", 4),
        ]),
    ))
    insufficient = _attribution()
    insufficient["evidence_review"] = {
        "judgment": "insufficient", "reason": "还需检查后续工具结果",
        "missing_evidence_queries": ["检查后续Chunk中的最终文件状态"],
    }
    client = FakeClient([
        _user_analysis(),
        {"trace_id": "trace_1", "selected_event_ids": [
            "event_tool_result#chunk_0000"
        ], "reason": "先检查开头"},
        insufficient,
        {"trace_id": "trace_1", "selected_event_ids": [
            "event_tool_result#chunk_0001"
        ], "reason": "补查后续文件状态"},
        _attribution(),
    ])

    result = analyze_trace(bundle, client, "run_test", 1_000, 1_000_000)

    provenance = result.record["analysis_provenance"]
    assert result.model_call_count == 5
    assert len(provenance["evidence_retrieval_rounds"]) == 2
    assert provenance["selected_tool_event_ids"] == [
        "event_tool_result#chunk_0000", "event_tool_result#chunk_0001",
    ]
    assert provenance["evidence_review"]["judgment"] == "sufficient"
    assert "missing_evidence_queries" in client.calls[3][1]


def test_still_insufficient_evidence_cannot_be_normalised_to_no_pain() -> None:
    huge = "large-tool-output-marker-" * 1_000
    bundle = TraceBundle("batch_test", "trace_1", (
        _turn("turn_1", 1, [
            _event("event_query", "user_message", "请把结果保存成 JSON。", 1),
            _event("event_answer", "assistant_message", "结果已经保存成 CSV。", 2),
            {
                "event_id": "event_tool_result", "sequence": 3,
                "timestamp": "2026-01-01T00:00:03Z", "type": "tool_result",
                "data": {"call_id": "call_1", "content": huge},
            },
        ]),
        _turn("turn_2", 2, [
            _event("event_feedback", "user_message", "不对，我明确要求的是 JSON。", 4),
        ]),
    ))
    insufficient = _attribution()
    insufficient["evidence_review"] = {
        "judgment": "insufficient", "reason": "没有看到最终文件检查",
        "missing_evidence_queries": ["查找最终文件检查"],
    }
    insufficient["episode_attributions"][0].update({
        "agent_related": False, "agent_related_reason": "暂未看到失败证据",
        "agent_failure": None, "capability_gaps": [], "outcome": "unknown",
    })
    client = FakeClient([
        _user_analysis(),
        {"trace_id": "trace_1", "selected_event_ids": [
            "event_tool_result#chunk_0000"
        ], "reason": "初查"},
        insufficient,
        {"trace_id": "trace_1", "selected_event_ids": [], "reason": "没有找到补充证据"},
    ])

    result = analyze_trace(bundle, client, "run_test", 1_000, 1_000_000)

    episode = result.record["task_episodes"][0]
    assert result.model_call_count == 4
    assert result.record["analysis_status"] == "needs_review"
    assert episode["agent_assessment"]["agent_related"] is None
    assert episode["pain_confirmed"] is None
    assert "agent_verification:evidence_not_confirmed_sufficient" in (
        result.record["validation_warnings"]
    )


def test_repair_appends_to_original_output_and_is_resumable(tmp_path: Path) -> None:
    batch = tmp_path / "preprocessed" / "batch_test"
    _write_jsonl(batch / "unified_turns.jsonl", list(_bundle().turns))
    run_id = "run_partial"
    run_dir = tmp_path / "analysis-runs" / "batch_test" / run_id
    stage = run_dir / "02_analyze"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text(json.dumps({
        "manifest_version": "1.0", "analysis_run_id": run_id,
        "batch_id": "batch_test", "status": "partial",
        "prompt_version": "trace-user-screen-rubrics-v1",
        "attribution_prompt_version": "trace-agent-verification-v2",
    }), encoding="utf-8")
    client = FakeClient([_user_analysis(), _attribution()])

    repaired = repair_trace_analysis(
        batch, run_dir, client, "fake", workers=1,
        direct_max_chars=1_000_000, request_max_chars=1_000_000,
        export_after=False,
    )

    main_output = stage / f"trace_analysis__batch_test__{run_id}.jsonl"
    main_screen = stage / f"user_screen__batch_test__{run_id}.jsonl"
    assert len(main_output.read_text(encoding="utf-8").splitlines()) == 1
    assert len(main_screen.read_text(encoding="utf-8").splitlines()) == 1
    assert repaired["status"] == "completed"
    assert repaired["total_pending_trace_count"] == 0
    assert Path(repaired["prompt_snapshot_path"]).is_file()

    again = repair_trace_analysis(
        batch, run_dir, FakeClient([]), "fake", workers=1, export_after=False,
    )
    assert again["status"] == "nothing_to_repair"
    assert len(main_output.read_text(encoding="utf-8").splitlines()) == 1


def test_invalid_agent_gap_is_downgraded_for_review() -> None:
    raw = _attribution()
    raw["episode_attributions"][0]["capability_gaps"][0]["evidence"][0]["quote"] = (
        "模型没有说过的文字"
    )
    base = normalise_user_analysis(_user_analysis(), _bundle(), "run_test")
    record = merge_agent_attribution(
        raw, _bundle(), base, {"event_query", "event_answer", "event_feedback"}
    )

    pain = record["task_episodes"][0]["agent_assessment"]
    assert record["analysis_status"] == "needs_review"
    assert pain["agent_related"] is None
    assert pain["capability_gaps"] == []


def test_run_export_and_status_are_resumable_and_auditable(tmp_path: Path) -> None:
    batch = tmp_path / "preprocessed" / "batch_test"
    _write_jsonl(batch / "unified_turns.jsonl", list(_bundle().turns))
    # Stage 01 compatibility check expects this file even though analysis streams Turns.
    _write_jsonl(batch / "unified_traces.jsonl", [{"trace_id": "trace_1", "sessions": []}])
    analysis_root = tmp_path / "workspace" / "analysis-runs"
    client = FakeClient([_user_analysis(), _attribution()])

    analyzed = run_trace_analysis(
        batch, analysis_root, client, "fake", workers=1,
        direct_max_chars=1_000_000, request_max_chars=1_000_000,
    )
    assert analyzed["status"] == "completed"
    run_dir = Path(analyzed["run_dir"])
    output = Path(analyzed["output_path"])
    assert output.name.startswith("trace_analysis__batch_test__run_")
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    screen_output = Path(analyzed["user_screen_output_path"])
    screen_row = json.loads(screen_output.read_text(encoding="utf-8").splitlines()[0])
    assert screen_row["route"]["judgment"] == "analyze"
    assert screen_row["task_episodes"][0]["agent_assessment"] is None

    exported = export_trace_analysis(run_dir, batch)
    export_dir = Path(exported["output_path"])
    requirements = list((export_dir / "requirements.jsonl").read_text(encoding="utf-8").splitlines())
    gaps = list((export_dir / "capability_gaps.jsonl").read_text(encoding="utf-8").splitlines())
    distribution = json.loads(
        (export_dir / "capability_gap_distribution.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert len(requirements) == 1
    assert len(gaps) == 1
    assert distribution["member_count"] == 1
    assert len(distribution["member_ids"]) == 1
    verified_traces = list(export_dir.glob("agent_verified_traces__*.jsonl"))
    verified_users = list(export_dir.glob("agent_verified_user_turns__*.jsonl"))
    assert len(verified_traces) == 1
    assert len(verified_users) == 1
    assert len(verified_traces[0].read_text(encoding="utf-8").splitlines()) == 1
    user_view = json.loads(verified_users[0].read_text(encoding="utf-8").splitlines()[0])
    assert user_view["trace_id"] == "trace_1"
    assert user_view["user_turn_count"] == 2

    status = pipeline_status(tmp_path / "preprocessed", analysis_root)
    assert status["batches"][0]["overall_status"] == "completed"
    assert status["batches"][0]["pending_trace_count"] == 0
