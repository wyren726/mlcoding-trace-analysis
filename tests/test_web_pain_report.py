import json
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from trace_analysis.web import (
    ReviewConflictError,
    SQLiteReviewStore,
    create_review_server,
    publish_pain_report,
    sync_reviews_to_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_publish_pain_report_builds_public_safe_episode_views(tmp_path: Path):
    analysis = tmp_path / "trace_analysis.jsonl"
    user_turns = tmp_path / "user_turns.jsonl"
    unified_turns = tmp_path / "unified_turns.jsonl"
    manifest = tmp_path / "manifest.json"
    site = tmp_path / "site"
    _write_jsonl(analysis, [{
        "trace_id": "trace_1",
        "route": {"judgment": "analyze"},
        "validation_warnings": ["warning_1"],
        "task_episodes": [{
            "episode_id": "episode_1",
            "query_turn_id": "turn_1",
            "episode_boundary": {"start_turn_id": "turn_1", "end_turn_id": "turn_2"},
            "initial_query_clarity": {"judgment": "clear", "reason": "目标明确"},
            "analysis_value": {"judgment": "high", "reason": "可以验收"},
            "preliminary_goal": {
                "goal": "完成训练并保存到 /data/users/private/model.pt",
                "requirements": [{
                    "requirement_id": "requirement_1",
                    "text": "完成训练",
                    "origin": "initial_query",
                    "status": "unmet",
                    "evidence": [{"turn_id": "turn_1", "event_id": "event_1", "quote": "请完成训练"}],
                }],
            },
            "candidate_signals": {
                "reason": "用户纠正",
                "signals": [{"type": "explicit_dissatisfaction", "turn_id": "turn_2", "event_id": "event_2", "quote": "这不是我要的"}],
            },
            "outcome": "failed",
            "agent_assessment": {
                "agent_related": True,
                "agent_failure": "未完成训练",
                "agent_related_reason": "要求清晰但 Agent 提前结束",
                "attribution": "execution_or_verification",
                "goal_assessment": {"goal": "完成训练", "reason": "用户直接提出"},
                "capability_gaps": [{
                    "capability": "长任务执行",
                    "reason": "提前结束",
                    "evidence": [{"turn_id": "turn_2", "event_id": "event_3", "quote": "保存到 https://private.example/output"}],
                }],
            },
        }],
    }])
    _write_jsonl(user_turns, [{
        "trace_id": "trace_1",
        "session_id": "private_session",
        "harness": "claude_code",
        "agent_models": ["anthropic/claude-4.6-opus-20260205"],
        "source_files": ["/private/session.jsonl"],
        "user_turns": [{"content": "完整私人对话"}],
    }])
    _write_jsonl(unified_turns, [{
        "trace_id": "trace_1",
        "harness": "claude_code",
        "agent_models": ["anthropic/claude-4.6-opus-20260205"],
        "events": [{"data": {"content": "完整 Tool Result"}}],
    }])
    manifest.write_text(json.dumps({
        "batch_id": "batch_1",
        "analysis_run_id": "run_1",
        "status": "completed",
        "total_trace_count": 10,
        "completed_trace_count": 10,
        "model": "judge-model",
        "prompt_version_history": ["screen-v1"],
        "attribution_prompt_version_history": ["verify-v2"],
    }), encoding="utf-8")

    result = publish_pain_report(analysis, user_turns, unified_turns, manifest, site)

    assert result["trace_count"] == 1
    assert result["confirmed_episode_count"] == 1
    root = site / "data" / "pain-report"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_trace_count"] == 10
    assert summary["verified_episode_count"] == 1
    index = json.loads((root / "case-index.json").read_text(encoding="utf-8"))
    assert index[0]["model_label"] == "Claude 4.6 Opus"
    assert index[0]["status"] == "confirmed"
    detail_text = (root / "cases" / "episode_1.json").read_text(encoding="utf-8")
    detail = json.loads(detail_text)
    assert detail["goal"] == "完成训练"
    assert detail["initial_query_clarity_reason"] == "目标明确"
    assert detail["requirements"][0]["text"] == "完成训练"
    assert detail["evidence_chain"][0]["event_id"] == "event_2"
    assert "[链接]" in detail_text
    assert "private_session" not in detail_text
    assert "完整私人对话" not in detail_text
    assert "完整 Tool Result" not in detail_text
    assert "/data/users/private/model.pt" not in detail_text


class HumanReviewSyncTest(unittest.TestCase):
    def test_save_refresh_sync_and_idempotence_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = root / "reviews.sqlite3"
            source = root / "trace_analysis.jsonl"
            output = root / "04_review" / "human_review_decisions.jsonl"
            manifest = root / "04_review" / "sync_manifest.json"
            source.write_text('{"episode_id":"episode_1","status":"review"}\n', encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            store = SQLiteReviewStore(database)

            first = store.save_review(
                case_id="batch_1::run_1::episode_1",
                batch_id="batch_1",
                analysis_run_id="run_1",
                trace_id="trace_1",
                episode_id="episode_1",
                decision="confirmed",
                note="要求清晰，Agent 未满足。",
                reviewer_user_id="reviewer_1",
                reviewer_email="reviewer@example.com",
                expected_revision=0,
            )
            self.assertEqual(first["revision"], 1)
            self.assertEqual(
                SQLiteReviewStore(database).latest_review(first["case_id"])["decision"],
                "confirmed",
            )

            first_sync = sync_reviews_to_jsonl(store, output, manifest)
            self.assertTrue(first_sync["changed"])
            exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["decision"], "confirmed")

            second_sync = sync_reviews_to_jsonl(store, output, manifest)
            self.assertFalse(second_sync["changed"])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

            second = store.save_review(
                case_id=first["case_id"],
                batch_id="batch_1",
                analysis_run_id="run_1",
                trace_id="trace_1",
                episode_id="episode_1",
                decision="excluded",
                note="复核后发现属于外部服务故障。",
                reviewer_user_id="reviewer_1",
                reviewer_email="reviewer@example.com",
                expected_revision=1,
            )
            self.assertEqual(second["revision"], 2)
            self.assertEqual(second["supersedes_review_id"], first["review_id"])
            third_sync = sync_reviews_to_jsonl(store, output, manifest)
            self.assertTrue(third_sync["changed"])
            exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["decision"], "excluded")
            self.assertEqual(store.event_count(), 2)

            with self.assertRaises(ReviewConflictError):
                store.save_review(
                    case_id=first["case_id"],
                    batch_id="batch_1",
                    analysis_run_id="run_1",
                    trace_id="trace_1",
                    episode_id="episode_1",
                    decision="confirmed",
                    note="过期页面提交。",
                    reviewer_user_id="reviewer_2",
                    reviewer_email="reviewer2@example.com",
                    expected_revision=1,
                )

            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)


class HumanReviewHTTPTest(unittest.TestCase):
    def test_http_save_refresh_conflict_and_automatic_jsonl_sync(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            site = root / "site"
            report = site / "data" / "pain-report"
            bundle = report / "batches" / "batch_1" / "run_1"
            bundle.mkdir(parents=True)
            (site / "index.html").write_text("<!doctype html><title>review</title>", encoding="utf-8")
            catalog = {
                "batches": [{
                    "batch_id": "batch_1",
                    "active_run_id": "run_1",
                    "case_index": "batches/batch_1/run_1/case-index.json",
                }],
            }
            (report / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            source = bundle / "case-index.json"
            source.write_text(json.dumps([{
                "case_id": "case_1",
                "batch_id": "batch_1",
                "analysis_run_id": "run_1",
                "trace_id": "trace_1",
                "episode_id": "episode_1",
            }]), encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            database = root / "reviews.sqlite3"
            output = root / "04_review" / "human_review_decisions.jsonl"
            manifest = root / "04_review" / "sync_manifest.json"
            server = create_review_server(
                host="127.0.0.1",
                port=0,
                site_dir=site,
                database_path=database,
                export_path=output,
                manifest_path=manifest,
                reviewer_user_id="local_user",
                reviewer_email="local@example.com",
                quiet=True,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            open_local = build_opener(ProxyHandler({})).open
            try:
                with open_local(f"{base}/api/pain-reviews") as response:
                    self.assertEqual(json.load(response)["reviews"], [])

                request = Request(
                    f"{base}/api/pain-reviews",
                    data=json.dumps({
                        "case_id": "case_1",
                        "decision": "confirmed",
                        "note": "要求清晰，Agent 未满足。",
                        "expected_revision": 0,
                        "trace_id": "spoofed_trace",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with open_local(request) as response:
                    saved = json.load(response)
                self.assertEqual(saved["review"]["revision"], 1)
                self.assertTrue(saved["sync"]["changed"])
                self.assertNotIn("reviewer_email", saved["review"])

                with open_local(f"{base}/api/pain-reviews") as response:
                    refreshed = json.load(response)["reviews"]
                self.assertEqual(len(refreshed), 1)
                self.assertEqual(refreshed[0]["decision"], "confirmed")

                exported = json.loads(output.read_text(encoding="utf-8").strip())
                self.assertEqual(exported["trace_id"], "trace_1")
                self.assertEqual(exported["decision"], "confirmed")
                self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["review_count"], 1)

                with self.assertRaises(HTTPError) as conflict:
                    open_local(request)
                self.assertEqual(conflict.exception.code, 409)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
