from __future__ import annotations

import json
import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trace_analysis.pipeline.stage_02_analyze.runner import (
    TraceBundle,
    analyze_user_trace,
    normalise_user_analysis,
)
from trace_analysis.pipeline.stage_03_agent_verify import run_agent_verify
from trace_analysis.pipeline.stage_02_user_screen import run_user_screen
from trace_analysis.pipeline.stage_01_preprocess import build_user_turns
from trace_analysis.pipeline.stage_01_preprocess.user_turns import (
    iter_user_trace_bundles,
)
from trace_analysis.pipeline.stage_04_build_results import build_pain_cases
from trace_analysis.pipeline.stage_05_report_publish import publish_pain_cases
from trace_analysis.model_api import RoundRobinJSONClient


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def event(event_id: str, kind: str, content: str, sequence: int) -> dict:
    return {
        "event_id": event_id, "sequence": sequence, "type": kind,
        "data": {"content": content},
        "source": {"source_record_id": f"source_{event_id}", "json_pointer": "/data/content"},
    }


def turn(trace_id: str, turn_id: str, index: int, events: list[dict]) -> dict:
    return {
        "batch_id": "batch_test", "trace_id": trace_id,
        "session_id": f"session_{trace_id}", "turn_id": turn_id,
        "turn_index": index, "harness": "claude_code",
        "agent_models": ["claude-opus"], "events": events,
    }


def user_raw(trace_id: str) -> dict:
    return {
        "trace_id": trace_id,
        "task_episodes": [{
            "episode_boundary": {"judgment": "reliable", "reason": "同一任务", "start_turn_id": "turn_1", "end_turn_id": "turn_2"},
            "domain": {"judgment": "in_scope", "reason": "科研代码任务", "evidence_turn_ids": ["turn_1"]},
            "analysis_value": {"judgment": "high", "reason": "可验证", "evidence_turn_ids": ["turn_1", "turn_2"]},
            "initial_query": {"judgment": "identified", "turn_id": "turn_1", "reason": "首次请求"},
            "initial_query_clarity": {"judgment": "clear", "reason": "格式明确", "evidence_turn_ids": ["turn_1"]},
            "preliminary_goal": {
                "judgment": "complete", "reason": "目标明确", "goal": "输出 JSON",
                "requirements": [{"text": "输出 JSON", "origin": "initial_query", "evidence": [{"turn_id": "turn_1", "event_id": "query", "quote": "输出 JSON"}]}],
            },
            "requirement_evolution": {"judgment": "stable", "reason": "未变化", "changes": []},
            "candidate_signals": {
                "judgment": "present", "reason": "明确纠正",
                "signals": [{"type": "explicit_rejection", "turn_id": "turn_2", "event_id": "feedback", "quote": "不是 CSV"}],
            },
        }],
    }


def attribution(trace_id: str) -> dict:
    return {
        "trace_id": trace_id,
        "evidence_review": {"judgment": "sufficient", "reason": "充分", "missing_evidence_queries": []},
        "episode_attributions": [{
            "query_turn_id": "turn_1",
            "goal_assessment": {"judgment": "confirmed", "goal": "输出 JSON", "reason": "未变化", "evidence": [{"turn_id": "turn_1", "event_id": "query", "quote": "输出 JSON"}]},
            "requirement_results": [{"requirement_index": 0, "status": "unmet", "evidence": [{"turn_id": "turn_1", "event_id": "answer", "quote": "输出 CSV"}]}],
            "confirmed_requirements": [{
                "text": "输出 JSON", "origin": "initial_query",
                "origin_evidence": [{"turn_id": "turn_1", "event_id": "query", "quote": "输出 JSON"}],
                "status": "unmet",
                "status_evidence": [{"turn_id": "turn_1", "event_id": "answer", "quote": "输出 CSV"}],
            }],
            "outcome": "failed", "agent_failure": "输出 CSV", "agent_related": True,
            "agent_related_reason": "违反明确格式", "attribution": "execution_or_verification",
            "capability_gaps": [{"capability_key": "output_format_compliance", "capability_name": "输出格式约束遵循", "reason": "没有输出 JSON", "evidence": [{"turn_id": "turn_1", "event_id": "answer", "quote": "输出 CSV"}]}],
        }],
    }


class FakeClient:
    model = "judge-model"
    config = SimpleNamespace(name="fake")

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def complete_json(self, system: str, user: str):
        return self.responses.pop(0), {"usage": {"total_tokens": 10}}


class PoolFakeClient:
    model = "judge-model"
    config = SimpleNamespace(name="fake")

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def complete_json(self, system: str, user: str):
        self.calls += 1
        return {"served_by": self.name}, {"usage": {"total_tokens": 1}}


class IncrementalPainPipelineTest(unittest.TestCase):
    def test_round_robin_client_distributes_calls_without_exposing_keys(self) -> None:
        clients = [PoolFakeClient("a"), PoolFakeClient("b"), PoolFakeClient("c")]
        pool = RoundRobinJSONClient(clients)
        results = [pool.complete_json("system", "user") for _ in range(7)]
        self.assertEqual([item[0]["served_by"] for item in results], [
            "a", "b", "c", "a", "b", "c", "a",
        ])
        self.assertEqual([client.calls for client in clients], [3, 2, 2])
        self.assertEqual(
            [item[1]["client_pool_index"] for item in results],
            [0, 1, 2, 0, 1, 2, 0],
        )

    def test_long_user_trace_uses_hierarchical_episode_analysis(self) -> None:
        turns = [
            turn(
                "trace_long", f"turn_{index}", index,
                [event(
                    f"event_{index}", "user_message",
                    f"用户消息{index}" + ("长上下文" * 150), index,
                )],
            )
            for index in range(1, 301)
        ]
        episode = user_raw("trace_long")["task_episodes"][0]
        episode["episode_boundary"].update(
            {"start_turn_id": "turn_1", "end_turn_id": "turn_300"}
        )
        episode["candidate_signals"] = {
            "judgment": "absent", "reason": "没有候选信号", "signals": []
        }
        result = analyze_user_trace(
            TraceBundle("batch_test", "trace_long", tuple(turns)),
            FakeClient([
                {"trace_id": "trace_long", "episode_boundaries": [{
                    "start_turn_id": "turn_1", "end_turn_id": "turn_300",
                    "reason": "同一连续目标",
                }]},
                {"trace_id": "trace_long", "task_episodes": [episode]},
            ]),
            "run_long", 1_000_000,
        )
        self.assertEqual(result.strategy, "hierarchical_user_screen")
        self.assertEqual(result.model_call_count, 2)
        self.assertEqual(
            result.record["analysis_provenance"]["episode_call_count"], 1
        )
        self.assertEqual(
            result.record["task_episodes"][0]["episode_boundary"]["end_turn_id"],
            "turn_300",
        )

    def test_stage_02_persists_user_screen_without_agent_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "preprocessed" / "batch_test"
            turns = [
                turn("trace_analyze", "turn_1", 1, [event("query", "user_message", "输出 JSON", 1), event("answer", "assistant_message", "输出 CSV", 2)]),
                turn("trace_analyze", "turn_2", 2, [event("feedback", "user_message", "不是 CSV", 3)]),
            ]
            write_jsonl(batch / "unified_turns.jsonl", turns)
            user_product = build_user_turns(batch)
            self.assertEqual(user_product["trace_count"], 1)
            self.assertEqual(user_product["user_turn_count"], 2)
            user_record = json.loads(
                (batch / "user_turns__batch_test.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["event_sequence"] for item in user_record["user_turns"]],
                [1, 3],
            )
            rebuilt = next(iter_user_trace_bundles(batch / "user_turns__batch_test.jsonl"))
            self.assertEqual(
                [item["events"][0]["sequence"] for item in rebuilt.turns],
                [1, 3],
            )
            self.assertEqual(
                [item["events"][0]["event_id"] for item in rebuilt.turns],
                ["query", "feedback"],
            )
            result = run_user_screen(
                batch, root / "analysis-runs", FakeClient([user_raw("trace_analyze")]),
                "fake", workers=1, request_max_chars=1_000_000, run_id="run_screen",
            )
            rows = list(Path(result["output_path"]).read_text(encoding="utf-8").splitlines())
            self.assertEqual(len(rows), 1)
            self.assertEqual(result["input_product"], "user_turns")
            record = json.loads(rows[0])
            self.assertEqual(record["route"]["judgment"], "analyze")
            self.assertIsNone(record["task_episodes"][0]["agent_assessment"])

    def test_stage_03_only_writes_analyze_and_builds_publishable_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "preprocessed" / "batch_test"
            turns = [
                turn("trace_analyze", "turn_1", 1, [event("query", "user_message", "输出 JSON", 1), event("answer", "assistant_message", "输出 CSV", 2)]),
                turn("trace_analyze", "turn_2", 2, [event("feedback", "user_message", "不是 CSV", 3)]),
                turn("trace_skip", "turn_s", 1, [event("skip_query", "user_message", "你好", 1)]),
            ]
            write_jsonl(batch / "unified_turns.jsonl", turns)
            run = root / "analysis-runs" / "batch_test" / "run_test"
            bundle = TraceBundle("batch_test", "trace_analyze", tuple(turns[:2]))
            analyzed = normalise_user_analysis(user_raw("trace_analyze"), bundle, "run_test")
            skipped = {
                "schema_version": "3.0", "batch_id": "batch_test",
                "trace_id": "trace_skip", "analysis_run_id": "run_test",
                "analysis_status": "complete",
                "route": {"judgment": "skip", "reason": "无分析价值"},
                "task_episodes": [],
            }
            write_jsonl(
                run / "02_user_screen" / "user_screen__batch_test__run_test.jsonl",
                [analyzed, skipped],
            )
            verified = run_agent_verify(
                batch, run, FakeClient([attribution("trace_analyze")]), "fake",
                workers=1, direct_max_chars=1_000_000, request_max_chars=1_000_000,
            )
            self.assertEqual(verified["selected_trace_count"], 1)
            output = Path(verified["output_path"])
            rows = list(output.read_text(encoding="utf-8").splitlines())
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["trace_id"], "trace_analyze")

            built = build_pain_cases(run, batch)
            pain_rows = list(Path(built["output_path"]).read_text(encoding="utf-8").splitlines())
            self.assertEqual(len(pain_rows), 1)
            case = json.loads(pain_rows[0])
            self.assertEqual(case["pain_judgment"], "confirmed")
            self.assertEqual(case["models"], ["claude-opus"])
            self.assertTrue(case["evidence_references"])
            self.assertTrue(case["source_references"])
            requirement = case["user_screen"]["preliminary_goal"]["requirements"][0]
            self.assertEqual(requirement["text"], "输出 JSON")
            self.assertEqual(requirement["status"], "unmet")
            self.assertEqual(case["capability_gaps"][0]["capability_key"], "output_format_compliance")
            self.assertEqual(case["capability_gaps"][0]["capability_name"], "输出格式约束遵循")

            published = publish_pain_cases(run, root / "publications" / "pain-report")
            catalog = json.loads(Path(published["catalog_path"]).read_text(encoding="utf-8"))
            self.assertEqual(catalog["batches"][0]["active_run_id"], "run_test")
            self.assertTrue((Path(published["publication_path"]) / "case-index.json").is_file())
            detail = json.loads(next(
                (Path(published["publication_path"]) / "cases").glob("*.json")
            ).read_text(encoding="utf-8"))
            self.assertEqual(detail["title"], "输出格式约束遵循")
            self.assertEqual(detail["initial_query_clarity_reason"], "格式明确")
            self.assertEqual(detail["requirements"][0]["status"], "unmet")
            self.assertEqual(detail["capability_gap_details"][0]["capability"], "输出格式约束遵循")
            self.assertIn("建议进入 Benchmark 候选池", detail["benchmark_note"])
            self.assertEqual(detail["goal_reason"], "未变化")
            transcript_path = next(
                (Path(published["publication_path"]) / "transcripts").glob("*.json.gz")
            )
            with gzip.open(transcript_path, "rt", encoding="utf-8") as source:
                transcript = json.load(source)
            self.assertEqual([item["user"]["role"] for item in transcript["turns"]], ["user", "user"])
            self.assertEqual(transcript["turns"][0]["agent_events"][0]["role"], "assistant")

            fallback_case = dict(case)
            fallback_case["user_screen"] = json.loads(json.dumps(case["user_screen"]))
            fallback_case["user_screen"]["preliminary_goal"]["requirements"] = []
            _, fallback_detail = __import__(
                "trace_analysis.pipeline.stage_05_report_publish.publisher",
                fromlist=["_public_case"],
            )._public_case(fallback_case)
            self.assertEqual(fallback_detail["requirements"][0]["origin"], "goal_fallback")
            self.assertEqual(fallback_detail["requirements"][0]["status"], "unknown")

            batch_two = root / "preprocessed" / "batch_two"
            batch_two.mkdir(parents=True)
            write_jsonl(batch_two / "unified_turns.jsonl", [
                turn("trace_two", "turn_1", 1, [event("query_two", "user_message", "输出 JSON", 1), event("answer_two", "assistant_message", "输出 CSV", 2)]),
                turn("trace_two", "turn_2", 2, [event("feedback_two", "user_message", "不是 CSV", 3)]),
            ])
            run_two = root / "analysis-runs" / "batch_two" / "run_two"
            result_two = run_two / "04_results"
            copied = dict(case)
            copied.update({
                "case_id": "case_two", "batch_id": "batch_two",
                "analysis_run_id": "run_two", "trace_id": "trace_two",
            })
            pain_two = result_two / "pain_cases__batch_two__run_two.jsonl"
            write_jsonl(pain_two, [copied])
            result_two.mkdir(parents=True, exist_ok=True)
            (result_two / "manifest.json").write_text(json.dumps({
                "batch_id": "batch_two", "analysis_run_id": "run_two",
                "output_path": str(pain_two), "status": "completed",
            }), encoding="utf-8")
            publish_pain_cases(run_two, root / "publications" / "pain-report")
            catalog = json.loads(Path(published["catalog_path"]).read_text(encoding="utf-8"))
            self.assertEqual({item["batch_id"] for item in catalog["batches"]}, {"batch_test", "batch_two"})
            global_summary = json.loads(
                (root / "publications" / "pain-report" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(global_summary["batch_count"], 2)
            self.assertEqual(global_summary["case_count"], 2)


if __name__ == "__main__":
    unittest.main()
