from __future__ import annotations

import json
from pathlib import Path

from trace_analysis.pipeline.stage_02_analyze.runner import (
    TraceBundle,
    _observable_turn,
    merge_agent_attribution,
)
from trace_analysis.pipeline.stage_04_build_results.builder import (
    _judgment_and_reasons,
)
from trace_analysis.review_remediation.planner import _classify_review
from trace_analysis.review_remediation.workflow import (
    approve_queue_items,
    build_validated_candidate_runs,
    prepare_approved_reanalysis,
)


def _review(note: str = "") -> dict:
    return {
        "review_id": "review_a",
        "case_id": "case_a",
        "batch_id": "batch_a",
        "analysis_run_id": "run_a",
        "trace_id": "trace_a",
        "episode_id": "episode_a",
        "decision": "review",
        "note": note,
    }


def test_interruption_is_visible_to_agent_verification_but_not_a_query():
    turn = {
        "turn_id": "turn_a",
        "turn_index": 1,
        "events": [{
            "event_id": "event_interrupt",
            "sequence": 1,
            "type": "user_message",
            "data": {"content": "[Request interrupted by user for tool use]"},
        }],
    }
    value = _observable_turn(turn)
    assert value["events"] == [{
        "event_id": "event_interrupt",
        "sequence": 1,
        "timestamp": None,
        "type": "user_interruption",
        "data": {
            "content": "[Request interrupted by user for tool use]",
            "action": "tool_use_rejected",
        },
    }]


def test_review_classification_uses_feedback_and_source_validation_state():
    case = {
        "pain_judgment": "review",
        "agent_verification": {
            "agent_related": None,
            "agent_failure": "回复被截断",
        },
        "capability_gaps": [],
        "validation_warnings": [
            "task_episodes[0].agent_assessment:unsupported_agent_attribution"
        ],
    }
    codes = {
        value["issue_code"]
        for value in _classify_review(
            _review("这是用户主动打断，且页面找不到证据turn"), case
        )
    }
    assert codes == {
        "user_interruption_unmodeled",
        "unsupported_attribution_evidence",
        "evidence_presentation_check",
    }


def test_review_status_has_machine_readable_reasons():
    judgment, reasons = _judgment_and_reasons(
        {"agent_related": None, "agent_failure": None},
        ["task_episodes[0].requirement_results[0]:missing"],
    )
    assert judgment == "review"
    assert reasons == [
        "required_analysis_fields_missing",
        "agent_failure_not_determined",
    ]


def test_interruption_assessment_keeps_exact_event_and_followup_evidence():
    bundle = TraceBundle("batch_a", "trace_a", (
        {
            "turn_id": "turn_query", "turn_index": 1,
            "events": [{
                "event_id": "event_query", "type": "user_message",
                "data": {"content": "请完成任务"},
            }],
        },
        {
            "turn_id": "turn_interrupt", "turn_index": 2,
            "events": [{
                "event_id": "event_interrupt", "type": "user_message",
                "data": {"content": "[Request interrupted by user]"},
            }],
        },
        {
            "turn_id": "turn_followup", "turn_index": 3,
            "events": [{
                "event_id": "event_followup", "type": "user_message",
                "data": {"content": "不要继续这个方向，先检查输入"},
            }],
        },
    ))
    user_record = {
        "schema_version": "3.0", "batch_id": "batch_a",
        "trace_id": "trace_a", "analysis_run_id": "run_a",
        "analysis_status": "complete", "route": {"judgment": "analyze"},
        "task_episodes": [{
            "episode_id": "episode_a", "query_turn_id": "turn_query",
            "episode_boundary": {
                "start_turn_id": "turn_query", "end_turn_id": "turn_followup",
            },
            "route": {"judgment": "analyze"},
            "preliminary_goal": {"requirements": []},
        }],
    }
    raw = {
        "trace_id": "trace_a",
        "episode_attributions": [{
            "query_turn_id": "turn_query",
            "goal_assessment": {
                "judgment": "confirmed", "goal": "完成任务", "reason": "明确",
                "evidence": [{
                    "event_id": "event_query", "quote": "请完成任务",
                }],
            },
            "requirement_results": [], "confirmed_requirements": [],
            "outcome": "unknown",
            "user_interruptions": [{
                "interruption_event": {
                    "event_id": "event_interrupt",
                    "quote": "[Request interrupted by user]",
                },
                "judgment": "dissatisfaction_signal",
                "reason": "后续用户否定当前方向",
                "followup_evidence": [{
                    "event_id": "event_followup", "quote": "不要继续这个方向",
                }],
            }],
            "agent_failure": None, "agent_related": None,
            "agent_related_reason": "", "attribution": "unknown",
            "capability_gaps": [],
        }],
    }
    visible = {
        event["event_id"]: event["data"]
        for turn in bundle.turns
        for event in _observable_turn(turn)["events"]
    }
    result = merge_agent_attribution(raw, bundle, user_record, set(visible), visible)
    interruptions = result["task_episodes"][0]["agent_assessment"]["user_interruptions"]
    assert interruptions[0]["judgment"] == "dissatisfaction_signal"
    assert interruptions[0]["interruption_event"]["event_id"] == "event_interrupt"
    assert interruptions[0]["followup_evidence"][0]["event_id"] == "event_followup"


def test_approval_and_prepare_reuse_only_selected_user_screens(tmp_path):
    cycle = tmp_path / "cycle_a"
    queue_dir = cycle / "04_reanalysis_queue"
    queue_dir.mkdir(parents=True)
    queue = {
        "queue_id": "queue_a",
        "batch_id": "batch_a",
        "baseline_analysis_run_id": "run_a",
        "trace_id": "trace_a",
        "issue_codes": ["user_interruption_unmodeled"],
        "reanalysis_mode": "agent_verify_only",
    }
    (queue_dir / "reanalysis_queue.jsonl").write_text(
        json.dumps(queue) + "\n", encoding="utf-8"
    )
    analysis = tmp_path / "analysis"
    screen_dir = analysis / "batch_a" / "run_a" / "02_user_screen"
    screen_dir.mkdir(parents=True)
    (screen_dir / "user_screen__batch_a__run_a.jsonl").write_text(
        json.dumps({
            "trace_id": "trace_a", "analysis_run_id": "run_a",
            "route": {"judgment": "analyze"},
        }) + "\n" + json.dumps({
            "trace_id": "trace_b", "analysis_run_id": "run_a",
            "route": {"judgment": "analyze"},
        }) + "\n",
        encoding="utf-8",
    )
    approval = approve_queue_items(
        cycle_dir=cycle, reviewer_name="tester", reason="pilot",
        queue_ids={"queue_a"},
    )
    assert approval["approved_count"] == 1
    result = prepare_approved_reanalysis(
        cycle_dir=cycle, analysis_root=analysis,
        preprocessed_root=tmp_path / "preprocessed",
    )
    assert result["approved_trace_count"] == 1
    jobs = list(
        json.loads(line)
        for line in (cycle / "05_reanalysis" / "jobs.jsonl").read_text().splitlines()
    )
    output = list(
        json.loads(line)
        for line in open(jobs[0]["input_user_screen"], encoding="utf-8")
    )
    assert [value["trace_id"] for value in output] == ["trace_a"]
    assert output[0]["analysis_run_id"].startswith("repair_cycle_a_")

    baseline_results = analysis / "batch_a" / "run_a" / "04_results"
    baseline_results.mkdir()
    baseline_path = baseline_results / "pain_cases__batch_a__run_a.jsonl"
    baseline_path.write_text("\n".join(json.dumps(value) for value in (
        {
            "case_id": "case_a", "batch_id": "batch_a",
            "analysis_run_id": "run_a", "trace_id": "trace_a",
            "episode_id": "episode_a", "pain_judgment": "review",
        },
        {
            "case_id": "case_b", "batch_id": "batch_a",
            "analysis_run_id": "run_a", "trace_id": "trace_b",
            "episode_id": "episode_b", "pain_judgment": "excluded",
        },
    )) + "\n", encoding="utf-8")
    jobs = list(
        json.loads(line)
        for line in (cycle / "05_reanalysis" / "jobs.jsonl").read_text().splitlines()
    )
    repair_run = Path(jobs[0]["repair_run_dir"])
    repair_results = repair_run / "04_results"
    repair_results.mkdir()
    repair_path = repair_results / "pain_cases__repair.jsonl"
    repair_path.write_text(json.dumps({
        "case_id": "case_a", "batch_id": "batch_a",
        "analysis_run_id": jobs[0]["repair_analysis_run_id"],
        "trace_id": "trace_a", "episode_id": "episode_a",
        "pain_judgment": "excluded",
    }) + "\n", encoding="utf-8")
    candidate = build_validated_candidate_runs(
        cycle_dir=cycle, analysis_root=analysis,
    )
    assert candidate["status"] == "needs_human_acceptance"
    assert candidate["changed_trace_count"] == 1
    candidate_path = next(
        (cycle / "06_diff_validation" / "candidate-runs").glob(
            "**/04_results/pain_cases__*.jsonl"
        )
    )
    candidate_cases = [json.loads(line) for line in candidate_path.open()]
    assert {value["trace_id"] for value in candidate_cases} == {"trace_a", "trace_b"}
    assert next(
        value for value in candidate_cases if value["trace_id"] == "trace_b"
    )["pain_judgment"] == "excluded"
