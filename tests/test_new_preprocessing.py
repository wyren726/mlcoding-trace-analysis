from __future__ import annotations

import csv
import json
import os
import time
import pytest
from pathlib import Path

from trace_analysis.preprocessing.pipeline import run_preprocess
from trace_analysis.features.basic import run_basic_features
from trace_analysis.features.basic.extractor import extract_turn
from trace_analysis.features.incremental import consolidate_feature_set
from trace_analysis.features.custom import run_semantic_features
from trace_analysis.features.custom.semantic import redact_for_api, searchable_event_text
from trace_analysis.features.model_api.client import OpenAICompatibleClient
from trace_analysis.analysis.core import (build_candidate_core, publish_reviewed_core,
                                           organize_hierarchy, review_suggestions_with_llm,
                                           suggest_reviews)
from trace_analysis.orchestration import run_update
from trace_analysis.analysis.extensions import run_extension
from trace_analysis.env import load_project_env
from trace_analysis.preprocessing.adapters.sls_proxy import SLSProxyAdapter
from trace_analysis.preprocessing.adapters.session_jsonl import SessionJSONLAdapter
from trace_analysis.registries import build_artifact_manifest, rebuild_registry_csv, write_immutable_record
from trace_analysis.artifact_sync import load_sync_config, sync_artifacts


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_immutable_registry_records_and_csv_rebuild(tmp_path: Path) -> None:
    artifact = tmp_path / "preprocessed" / "batch-1" / "unified_turns.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    record = {"batch_id": "batch-1", "status": "success", "event_count": 1}
    path = write_immutable_record("datasets", "batch-1", record, (artifact,), root=tmp_path)
    assert path.is_file()
    assert write_immutable_record("datasets", "batch-1", record, (artifact,), root=tmp_path) == path
    with pytest.raises(ValueError, match="different content"):
        write_immutable_record("datasets", "batch-1", {**record, "event_count": 2},
                               (artifact,), root=tmp_path)
    output = tmp_path / "data_registry.csv"
    result = rebuild_registry_csv("datasets", output, tmp_path)
    assert result["record_count"] == 1
    with output.open(encoding="utf-8-sig") as handle:
        assert next(csv.DictReader(handle))["batch_id"] == "batch-1"
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_result = build_artifact_manifest(manifest_path, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest_result["available_count"] == 1
    assert manifest["artifacts"][0]["sha256"]


def test_artifact_sync_status_is_dry_run_and_non_destructive(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "sync.toml"
    config.write_text('[artifacts]\nremote="user@example:/artifacts"\nroots=["features"]\n', encoding="utf-8")
    (tmp_path / "features").mkdir()
    commands = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr("trace_analysis.artifact_sync.subprocess.run", fake_run)
    assert load_sync_config(config)["roots"] == ["features"]
    result = sync_artifacts("status", config, tmp_path)
    assert result["status"] == "success" and result["operation_count"] == 2
    assert all("--dry-run" in command and "--delete" not in command for command in commands)


def test_session_jsonl_adapter_reconstructs_turn_and_tools(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    records = [
        {"type": "queue-operation", "operation": "enqueue", "content": "system prompt",
         "timestamp": "2026-01-01T00:00:00Z", "sessionId": "s1"},
        {"type": "user", "promptId": "p1", "uuid": "u1", "timestamp": "2026-01-01T00:00:01Z",
         "sessionId": "s1", "message": {"role": "user", "content": "请读取文件"}},
        {"type": "assistant", "uuid": "a1", "timestamp": "2026-01-01T00:00:02Z", "sessionId": "s1",
         "message": {"id": "m1", "role": "assistant", "model": "model-a", "usage": {"input_tokens": 3, "output_tokens": 2},
                     "content": [{"type": "thinking", "thinking": "需要读取"},
                                 {"type": "tool_use", "id": "c1", "name": "Read", "input": {"path": "/Users/a/x"}}],
                     "stop_reason": "tool_use"}},
        {"type": "user", "uuid": "u2", "timestamp": "2026-01-01T00:00:03Z", "sessionId": "s1",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "done"}]}},
        {"type": "assistant", "uuid": "a2", "timestamp": "2026-01-01T00:00:04Z", "sessionId": "s1",
         "message": {"id": "m2", "role": "assistant", "model": "model-a", "usage": {"input_tokens": 4, "output_tokens": 2},
                     "content": [{"type": "text", "text": "已读取"}], "stop_reason": "end_turn"}},
    ]
    write_jsonl(source, records)
    result = run_preprocess([str(source)], tmp_path / "out", None, None)
    turns = [json.loads(x) for x in Path(result["output_path"], "unified_turns.jsonl").open() if x.strip()]
    assert result["adapter"] == "session_jsonl" and len(turns) == 1
    turn = turns[0]
    assert turn["status"] == "completed" and turn["harness"] == "claude_code"
    assert turn["agent_models"] == ["model-a"]
    assert [e["type"] for e in turn["events"] if e["type"] in {"user_message", "tool_call", "tool_result", "assistant_message"}] == [
        "user_message", "tool_call", "tool_result", "assistant_message"]
    call = next(e for e in turn["events"] if e["type"] == "tool_call")
    assert call["data"]["arguments"]["path"] == "<LOCAL_PATH>"
    trace = json.loads(Path(result["output_path"], "unified_traces.jsonl").read_text())
    assert any(e["data"]["kind"] == "queue-operation" for e in trace["sessions"][0]["unassigned_events"])


def test_all_standard_output_schemas_exist_and_parse() -> None:
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1]
    paths = [
        "preprocessed/unified_traces.schema.yaml",
        "preprocessed/unified_turns.schema.yaml",
        "preprocessed/data_registry.schema.yaml",
        "preprocessed/rejected_records.schema.yaml",
        "features/basic_turn_features.schema.yaml",
        "features/semantic_features.schema.yaml",
        "features/feature_registry.schema.yaml",
        "analysis/capability_taxonomy.schema.yaml",
        "analysis/capability_cases.schema.yaml",
        "analysis/capability_distribution.schema.yaml",
        "analysis/model_capability_comparison.schema.yaml",
        "analysis/harness_capability_comparison.schema.yaml",
        "analysis/training_trace_candidates.schema.yaml",
        "analysis/unmet_need_priority.schema.yaml",
        "pipeline-runs/run_manifest.schema.yaml",
        "registries/registry_record.schema.yaml",
        "artifacts/manifest.schema.yaml",
        "OUTPUT_SCHEMA_INDEX.yaml",
    ]
    for relative in paths:
        path = root / relative
        assert path.is_file(), relative
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict) and document.get("schema_document_version") == "1.0", relative


def test_project_env_loads_without_overriding_process_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TRACE_TEST_FROM_FILE=file-value\nTRACE_TEST_EXISTING=file-value\n", encoding="utf-8")
    os.environ["TRACE_TEST_EXISTING"] = "process-value"
    os.environ.pop("TRACE_TEST_FROM_FILE", None)
    try:
        assert load_project_env(env_file) == env_file.resolve()
        assert os.environ["TRACE_TEST_FROM_FILE"] == "file-value"
        assert os.environ["TRACE_TEST_EXISTING"] == "process-value"
    finally:
        os.environ.pop("TRACE_TEST_FROM_FILE", None)
        os.environ.pop("TRACE_TEST_EXISTING", None)


def test_collector_auto_detection_and_turn_propagation(tmp_path: Path) -> None:
    source = tmp_path / "collector.jsonl"
    base = {"trace_schema_version": "v0.2.0", "session_id": "s1", "trace_id": None,
            "timestamp": "2026-01-01T00:00:00Z", "project_hmac": {},
            "project_identity_source": "cwd", "source": {"adapter": "codex_rollout_jsonl"}}
    write_jsonl(source, [
        {**base, "event_id": "e1", "turn_id": "t1", "sequence": 1, "event_type": "event_msg",
         "payload": {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}}},
        {**base, "event_id": "e2", "turn_id": None, "sequence": 2, "event_type": "response_item",
         "payload": {"type": "response_item", "payload": {"type": "function_call", "call_id": "c1", "name": "shell"}}},
        {**base, "event_id": "e3", "turn_id": "t1", "sequence": 3, "event_type": "event_msg",
         "payload": {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t1"}}},
    ])
    result = run_preprocess([str(source)], tmp_path / "out", None, None)
    turns = Path(result["output_path"], "unified_turns.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(turns[0])
    assert [event["type"] for event in record["events"]] == ["system_event", "tool_call", "system_event"]
    assert all("raw" not in event and "raw_ref" in event for event in record["events"])
    assert result["adapter"] == "collector_events"
    with (tmp_path / "out" / "data_registry.csv").open(encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["session_count"] == "1"


def test_collector_v020_normalizes_codex_tools_model_harness_and_duplicate_messages(tmp_path: Path) -> None:
    source = tmp_path / "collector-v020.jsonl"
    base = {"trace_schema_version": "v0.2.0", "session_id": "s1", "trace_id": None,
            "timestamp": "2026-08-11T00:00:00Z", "project_hmac": {},
            "project_identity_source": "cwd",
            "source": {"adapter": "codex_rollout_jsonl", "collector_version": "0.3.7",
                       "capture_kind": "local"}}

    def event(index: int, event_type: str, payload: dict, turn_id: str | None = None) -> dict:
        return {**base, "event_id": f"e{index}", "turn_id": turn_id, "sequence": index,
                "event_type": event_type, "payload": {"type": event_type, "payload": payload}}

    write_jsonl(source, [
        event(1, "event_msg", {"type": "task_started", "turn_id": "t1"}, "t1"),
        event(2, "turn_context", {"turn_id": "t1", "model": "gpt-5.6-sol"}),
        event(3, "response_item", {"type": "message", "role": "user", "content": "duplicate user"}),
        event(4, "event_msg", {"type": "user_message", "message": "真实用户消息"}),
        event(5, "response_item", {"type": "custom_tool_call", "call_id": "c1",
                                    "name": "exec", "input": "{\"cmd\":\"pwd\"}", "status": "completed"}),
        event(6, "response_item", {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"}),
        event(7, "response_item", {"type": "function_call", "call_id": "c2",
                                    "name": "apply_patch", "arguments": "patch"}),
        event(8, "response_item", {"type": "function_call_output", "call_id": "c2", "output": "done"}),
        event(9, "event_msg", {"type": "web_search_end", "call_id": "c2", "results": []}),
        event(10, "event_msg", {"type": "patch_apply_end", "call_id": "c2", "success": True}),
        event(11, "response_item", {"type": "message", "role": "assistant", "content": "duplicate agent"}),
        event(12, "event_msg", {"type": "agent_message", "message": "真实Agent回复"}),
        event(13, "event_msg", {"type": "token_count", "info": {
            "last_token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "total_token_usage": {"input_tokens": 300, "output_tokens": 50, "total_tokens": 350},
            "model_context_window": 200000}}),
        event(14, "compacted", {"window_number": 2, "replacement_history": []}),
        event(15, "event_msg", {"type": "task_complete", "turn_id": "t1"}, "t1"),
    ])
    result = run_preprocess([str(source)], tmp_path / "out", None, None)
    turn = json.loads(Path(result["output_path"], "unified_turns.jsonl").read_text(encoding="utf-8"))
    assert turn["status"] == "completed"
    assert turn["harness"] == "codex"
    assert turn["agent_models"] == ["gpt-5.6-sol"]
    assert len([x for x in turn["events"] if x["type"] == "user_message"]) == 1
    assert len([x for x in turn["events"] if x["type"] == "assistant_message"]) == 1
    assert not [x for x in turn["events"] if x["type"] == "unknown"]
    calls = [x for x in turn["events"] if x["type"] == "tool_call"]
    results = [x for x in turn["events"] if x["type"] == "tool_result"]
    assert [(x["data"]["call_id"], x["data"]["tool_name"]) for x in calls] == [
        ("c1", "exec"), ("c2", "apply_patch")]
    assert [x["data"]["call_id"] for x in results] == ["c1", "c2"]
    assert all(x["type"] == "system_event" for x in turn["events"] if
               (x.get("data") or {}).get("type") in {"message", "web_search_end", "patch_apply_end"})
    features = extract_turn(turn, "test-run")
    assert features["values"]["tool_pairing"] == {
        "matched_count": 2, "unmatched_call_ids": [], "unmatched_result_call_ids": []}
    assert features["values"]["user_input_length"] == len("真实用户消息")
    assert features["values"]["assistant_output_length"] == len("真实Agent回复")
    assert features["values"]["token_usage"] == {
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    with (tmp_path / "out" / "data_registry.csv").open(encoding="utf-8-sig") as handle:
        registry = next(csv.DictReader(handle))
    assert json.loads(registry["model_distribution"])["counts"] == {"gpt-5.6-sol": 1}
    assert json.loads(registry["harness_distribution"])["counts"] == {"codex": 1}


def test_sls_auto_detection(tmp_path: Path) -> None:
    source = tmp_path / "sls.jsonl"
    payload = {"id": "r1", "model": "m1", "messages": [{"role": "user", "content": "hello"}], "response": "done"}
    write_jsonl(source, [{"_time_": "now", "content": json.dumps(payload)}])
    result = run_preprocess([str(source)], tmp_path / "out", None, None)
    assert result["adapter"] == "sls_proxy"
    assert result["event_count"] == 5
    turn = json.loads(Path(result["output_path"], "unified_turns.jsonl").read_text(encoding="utf-8"))
    assert turn["agent_models"] == ["m1"]
    assert turn["status"] == "completed"


def test_sls_cumulative_snapshots_normalize_and_do_not_repeat_history() -> None:
    adapter = SLSProxyAdapter()
    common = {"model": "m1", "request_metadata": {"headers": {
        "x-claude-code-session-id": "session-1", "user-agent": "claude-code"}}}
    first = {"content": json.dumps({**common, "id": "r1", "timestamp": "2026-01-01T00:00:00Z",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "检查引用格式",
                                                       "cache_control": {"type": "ephemeral"}}]}],
        "reasoning": "检查 APA", "response": "建议使用 APA 7"})}
    second = {"content": json.dumps({**common, "id": "r2", "timestamp": "2026-01-01T00:01:00Z",
        "messages": [
            {"role": "user", "content": "检查引用格式"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "检查 APA"},
                                                  {"type": "text", "text": "建议使用 APA 7"}]},
            {"role": "user", "content": [{"type": "text", "text": "继续核查 DOI",
                                             "cache_control": {"type": "ephemeral"}}]},
        ], "reasoning": "开始核查 DOI", "response": "已核查 DOI"})}
    events = []
    events.extend(adapter.convert(first, batch_id="b1", source_file="sls.jsonl", record_index=0))
    events.extend(adapter.convert(second, batch_id="b1", source_file="sls.jsonl", record_index=1))
    user_messages = [event["data"]["content"] for event in events if event["type"] == "user_message"]
    assistant_messages = [event["data"]["content"] for event in events if event["type"] == "assistant_message"]
    assert user_messages == ["检查引用格式", "继续核查 DOI"]
    assert assistant_messages == ["建议使用 APA 7", "已核查 DOI"]


def test_sls_platform_instructions_do_not_create_user_turns() -> None:
    adapter = SLSProxyAdapter()
    payload = {"id": "r1", "model": "m1", "request_metadata": {"headers": {
        "x-claude-code-session-id": "session-1"}}, "messages": [
        {"role": "user", "content": "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."},
        {"role": "user", "content": "真实用户需求"},
    ], "response": "完成"}
    events = adapter.convert({"content": json.dumps(payload)}, batch_id="b1",
                             source_file="sls.jsonl", record_index=0)
    platform = [event for event in events if event["type"] == "system_event"
                and (event.get("data") or {}).get("content")]
    users = [event for event in events if event["type"] == "user_message"]
    assert len(platform) == 1
    assert [event["data"]["content"] for event in users] == ["真实用户需求"]

    quoted_upload = {"content": json.dumps({"id": "r2", "model": "m1",
        "request_metadata": {"headers": {"x-claude-code-session-id": "session-2"}},
        "messages": [{"role": "user", "content":
            "CRITICAL: The phrase 'The user just uploaded 10 files' is only an instruction."}],
        "response": "done"})}
    quoted_events = adapter.convert(quoted_upload, batch_id="b1", source_file="sls.jsonl", record_index=1)
    assert not any(event["type"] == "system_event" and (event.get("data") or {}).get("kind") == "uploaded_files"
                   for event in quoted_events)


def test_sls_normalizes_top_level_tool_calls_and_splits_upload_metadata() -> None:
    adapter = SLSProxyAdapter()
    payload = {"id": "r1", "model": "m1", "request_metadata": {"headers": {
        "x-claude-code-session-id": "session-1"}}, "messages": [{"role": "user", "content":
        "继续上传\n\nThe user just uploaded 1 files.\n- paper.pdf: local path /mnt/agent-workspace/x/paper.pdf"}],
        "tool_calls": [{"id": "c1", "name": "Read", "input": {"path": "/mnt/agent-workspace/x/paper.pdf"}}],
        "response": "收到"}
    events = adapter.convert({"content": json.dumps(payload)}, batch_id="b1",
                             source_file="sls.jsonl", record_index=0)
    user = next(event for event in events if event["type"] == "user_message")
    upload = next(event for event in events if event["type"] == "system_event"
                  and (event.get("data") or {}).get("kind") == "uploaded_files")
    call = next(event for event in events if event["type"] == "tool_call")
    assert user["data"]["content"] == "继续上传"
    assert upload["data"]["files"][0]["name"] == "paper.pdf"
    assert call["data"] == {"call_id": "c1", "tool_name": "Read",
                            "arguments": {"path": "<LOCAL_PATH>"}}


def test_platform_title_request_remains_unassigned(tmp_path: Path) -> None:
    source = tmp_path / "title.jsonl"
    payload = {"id": "r1", "model": "m1", "messages": [{"role": "user",
        "content": "Generate a concise chat-conversation title from the user's first message."}],
        "response": "会话标题"}
    write_jsonl(source, [{"content": json.dumps(payload)}])
    result = run_preprocess([str(source)], tmp_path / "out", "sls_proxy", None)
    assert Path(result["output_path"], "unified_turns.jsonl").read_text(encoding="utf-8") == ""
    trace = json.loads(Path(result["output_path"], "unified_traces.jsonl").read_text(encoding="utf-8"))
    assert len(trace["sessions"][0]["unassigned_events"]) >= 3


def test_basic_tool_features(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    turn = {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "status": "completed", "harness": "codex",
            "events": [
                {"type": "tool_call", "data": {"call_id": "c1", "tool_name": "shell", "arguments": "abc"}},
                {"type": "tool_result", "data": {"call_id": "c1", "output": "done"}},
            ]}
    write_jsonl(batch / "unified_turns.jsonl", [turn])
    result = run_basic_features(batch, tmp_path / "features")
    feature = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    assert feature["values"]["tool_usage"]["shell"] == {
        "call_count": 1, "result_count": 1, "argument_length": 3, "result_length": 4,
    }


def test_basic_tool_features_deduplicate_logical_call_ids(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    turn = {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "events": [
        {"type": "tool_call", "data": {"call_id": "c1", "tool_name": "shell", "arguments": "abc"}},
        {"type": "tool_call", "data": {"call_id": "c1", "tool_name": "shell", "arguments": "abc"}},
        {"type": "tool_result", "data": {"call_id": "c1", "content": "done"}},
        {"type": "tool_result", "data": {"call_id": "c1", "content": "done"}},
    ]}
    write_jsonl(batch / "unified_turns.jsonl", [turn])
    result = run_basic_features(batch, tmp_path / "features")
    values = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))["values"]
    assert values["tool_usage"]["shell"] == {
        "call_count": 1, "result_count": 1, "argument_length": 3, "result_length": 4,
    }


def test_semantic_feature_run_with_mock_client(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "turn_index": 1,
         "events": [{"event_id": "a1", "type": "assistant_message", "data": {"content": "已完成"}}]},
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t2", "turn_index": 2,
         "events": [{"event_id": "e1", "type": "user_message", "data": {"content": "没有完成"}}]},
    ])

    class MockClient:
        model = "mock-model"
        config = object()

        def complete_json(self, system: str, user: str):
            has_feedback = "[NEXT_TURN_USER_FOLLOW_UP]" in user
            return ({"sentiment": "negative" if has_feedback else "none", "explicit": has_feedback,
                     "user_correction": False, "complaints": ["未完成"] if has_feedback else [],
                     "evidence": ([{"event_id": "e1", "quote": "没有完成"}] if has_feedback else [])},
                    {"cache_hit": False})

    result = run_semantic_features(batch, tmp_path / "features", "user_feedback", MockClient(), "mock", None)
    assert result["record_count"] == 2
    feature = json.loads(Path(result["output_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert feature["values"]["sentiment"] == "negative"


def test_consolidate_incremental_feature_runs_by_batch(tmp_path: Path) -> None:
    root = tmp_path / "features"
    feature_dir = root / "demand_capability_instances"
    for run_name, batch, turn_id in (("run_a", "batch_a", "t1"), ("run_b", "batch_b", "t2")):
        run_dir = feature_dir / run_name
        run_dir.mkdir(parents=True)
        output = run_dir / "features.jsonl"
        write_jsonl(output, [{"turn_id": turn_id, "feature_run_id": run_name,
                              "feature_set": "demand_capability_instances",
                              "feature_version": "v3", "values": {"instances": []}}])
        (run_dir / "run_manifest.json").write_text(json.dumps({
            "feature_set": "demand_capability_instances", "feature_version": "v3",
            "input_batch": batch, "provider": "mock", "model": "m",
            "output_path": str(output.resolve()), "status": "success",
        }), encoding="utf-8")
    result = consolidate_feature_set(
        root, "demand_capability_instances", "v3", ["batch_a", "batch_b"], "mock", "m"
    )
    assert result["batch_count"] == 2 and result["record_count"] == 2
    assert len((feature_dir / "consolidated" / "batch_a.jsonl").read_text().splitlines()) == 1
    assert len((feature_dir / "consolidated" / "batch_b.jsonl").read_text().splitlines()) == 1


def test_semantic_features_are_flushed_after_each_record(tmp_path: Path) -> None:
    batch = tmp_path / "batch_live"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "turn_index": 1,
         "events": [{"event_id": "u1", "type": "user_message", "data": {"content": "需求一"}}]},
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t2", "turn_index": 2,
         "events": [{"event_id": "u2", "type": "user_message", "data": {"content": "需求二"}}]},
    ])
    output_root = tmp_path / "features"

    class ObservingClient:
        model = "mock-model"
        config = object()
        calls = 0

        def complete_json(self, system: str, user: str):
            self.calls += 1
            if "需求二" in user:
                outputs = list((output_root / "demand_capability_instances").glob("*/features.jsonl"))
                assert len(outputs) == 1
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not outputs[0].read_text(encoding="utf-8").splitlines():
                    time.sleep(0.01)
                assert len(outputs[0].read_text(encoding="utf-8").splitlines()) == 1
            event_id = "u1" if "需求一" in user else "u2"
            quote = "需求一" if "需求一" in user else "需求二"
            return ({"user_goal": quote, "instances": [{
                "need": quote, "object": "代码", "expected_outcome": "完成", "constraints": [],
                "requirement_source": "explicit_user", "fulfillment": "unclear", "agent_gap": "",
                "candidate_capability": "需求处理",
                "requirement_evidence": [{"event_id": event_id, "quote": quote}],
                "fulfillment_evidence": [],
            }]}, {"cache_hit": False})

    result = run_semantic_features(
        batch, output_root, "demand_capability_instances", ObservingClient(), "mock", None
    )
    manifest = json.loads(Path(result["output_path"]).with_name("run_manifest.json").read_text(encoding="utf-8"))
    assert result["record_count"] == 2
    assert manifest["status"] == "success" and manifest["record_count"] == 2


def test_interrupted_semantic_run_reuses_flushed_records(tmp_path: Path) -> None:
    batch = tmp_path / "batch_resume"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": turn_id, "turn_index": index,
         "events": [{"event_id": event_id, "type": "user_message", "data": {"content": quote}}]}
        for index, (turn_id, event_id, quote) in enumerate(
            [("t1", "u1", "需求一"), ("t2", "u2", "需求二")], 1
        )
    ])
    output_root = tmp_path / "features"

    def response(user: str):
        quote, event_id = ("需求一", "u1") if "需求一" in user else ("需求二", "u2")
        return ({"user_goal": quote, "instances": [{
            "need": quote, "object": "代码", "expected_outcome": "完成", "constraints": [],
            "requirement_source": "explicit_user", "fulfillment": "unclear", "agent_gap": "",
            "candidate_capability": "需求处理",
            "requirement_evidence": [{"event_id": event_id, "quote": quote}],
            "fulfillment_evidence": [],
        }]}, {"cache_hit": False})

    class InterruptingClient:
        model = "mock-model"
        config = object()
        calls = 0

        def complete_json(self, system: str, user: str):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return response(user)

    with pytest.raises(KeyboardInterrupt):
        run_semantic_features(
            batch, output_root, "demand_capability_instances", InterruptingClient(), "mock", None
        )

    class ResumeClient:
        model = "mock-model"
        config = object()
        calls = 0

        def complete_json(self, system: str, user: str):
            self.calls += 1
            return response(user)

    client = ResumeClient()
    resumed = run_semantic_features(
        batch, output_root, "demand_capability_instances", client, "mock", None
    )
    assert client.calls == 1
    assert resumed["record_count"] == 1
    assert resumed["skipped_existing_count"] == 1


def test_user_feedback_uses_next_turn_user_message_as_follow_up(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "turn_index": 1,
         "events": [{"event_id": "answer", "type": "assistant_message", "data": {"content": "已完成"}}]},
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t2", "turn_index": 2,
         "events": [{"event_id": "feedback", "type": "user_message", "data": {"content": "完全不是我要的"}}]},
    ])

    class MockClient:
        model = "mock-model"
        config = object()
        prompts: list[str] = []

        def complete_json(self, system: str, user: str):
            self.prompts.append(user)
            follow_up = (user.split("[NEXT_TURN_USER_FOLLOW_UP]", 1)[1]
                         if "[NEXT_TURN_USER_FOLLOW_UP]" in user else "")
            evidence = ([{"event_id": "feedback", "quote": "完全不是我要的"}]
                        if "完全不是我要的" in follow_up else [])
            return ({"sentiment": "negative" if evidence else "none", "explicit": bool(evidence),
                     "user_correction": bool(evidence), "complaints": [], "evidence": evidence},
                    {"cache_hit": False})

    client = MockClient()
    result = run_semantic_features(batch, tmp_path / "features", "user_feedback", client, "mock", None)
    assert result["record_count"] == 2
    assert "[CURRENT_TURN]" in client.prompts[0]
    assert "[NEXT_TURN_USER_FOLLOW_UP]" in client.prompts[0]
    assert "[feedback] user_message" in client.prompts[0]
    assert "[NO_NEXT_TURN_USER_FOLLOW_UP]" in client.prompts[1]


def test_task_outcome_can_use_next_turn_feedback(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "turn_index": 1,
         "events": [{"event_id": "a1", "type": "assistant_message", "data": {"content": "已修复"}}]},
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t2", "turn_index": 2,
         "events": [{"event_id": "f1", "type": "user_message", "data": {"content": "问题仍然存在"}}]},
    ])

    class MockClient:
        model = "mock-model"
        config = object()
        prompts: list[str] = []

        def complete_json(self, system: str, user: str):
            self.prompts.append(user)
            failed = "[NEXT_TURN_USER_FOLLOW_UP]" in user
            return ({"outcome": "failed" if failed else "unclear", "verified": False if failed else "unclear",
                     "unmet_parts": ["问题未解决"] if failed else [],
                     "evidence": ([{"event_id": "f1", "quote": "问题仍然存在"}] if failed else [])},
                    {"cache_hit": False})

    client = MockClient()
    result = run_semantic_features(batch, tmp_path / "features", "task_outcome", client, "mock", None)
    assert result["record_count"] == 2
    assert "[NEXT_TURN_USER_FOLLOW_UP]" in client.prompts[0]
    assert "[f1] user_message" in client.prompts[0]
    assert "[NO_NEXT_TURN_USER_FOLLOW_UP]" in client.prompts[1]


def test_task_outcome_drops_only_inexact_redundant_evidence(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [{
        "trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "turn_index": 1,
        "events": [{"event_id": "e1", "type": "tool_result", "data": {"content": "测试通过"}}],
    }])

    class MockClient:
        model = "mock-model"
        config = object()

        def complete_json(self, system: str, user: str):
            return ({"outcome": "completed", "verified": True, "unmet_parts": [], "evidence": [
                {"event_id": "e1", "quote": "测试通过"},
                {"event_id": "e1", "quote": "测试已经成功通过"},
            ]}, {"cache_hit": False})

    result = run_semantic_features(batch, tmp_path / "features", "task_outcome", MockClient(), "mock", None)
    record = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    assert record["values"]["evidence"] == [{"event_id": "e1", "quote": "测试通过"}]
    assert record["api_meta"]["dropped_inexact_evidence_count"] == 1


def test_difficulty_drops_only_inexact_redundant_evidence(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [{
        "trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "turn_index": 1,
        "events": [{"event_id": "e1", "type": "user_message", "data": {"content": "修改三个文件并测试"}}],
    }])

    class MockClient:
        model = "mock-model"
        config = object()

        def complete_json(self, system: str, user: str):
            return ({"intrinsic_difficulty": "medium", "observed_difficulty": "low",
                     "factors": ["跨文件修改"], "evidence": [
                         {"event_id": "e1", "quote": "修改三个文件"},
                         {"event_id": "e1", "quote": "需要修改多个文件并运行测试"},
                     ]}, {"cache_hit": False})

    result = run_semantic_features(batch, tmp_path / "features", "difficulty", MockClient(), "mock", None)
    record = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    assert record["values"]["evidence"] == [{"event_id": "e1", "quote": "修改三个文件"}]
    assert record["api_meta"]["dropped_inexact_evidence_count"] == 1


def test_open_ended_demand_capability_instances(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [{"trace_id": "tr1", "session_id": "s1", "turn_id": "t1",
        "events": [
            {"event_id": "e1", "type": "user_message", "data": {"content": "请修复问题并运行测试"}},
            {"event_id": "e2", "type": "assistant_message", "data": {"content": "只修改了代码，没有测试"}},
        ]}])

    class MockClient:
        model = "mock-model"
        config = object()

        def complete_json(self, system: str, user: str):
            return ({"user_goal": "修复并验证问题", "instances": [{
                "need": "验证修复有效", "object": "修改后的代码", "expected_outcome": "相关测试通过",
                "constraints": ["必须运行测试"], "requirement_source": "explicit_user",
                "fulfillment": "unmet", "agent_gap": "未执行测试",
                "candidate_capability": "修改后主动运行针对性测试",
                "requirement_evidence": [{"event_id": "e1", "quote": "运行测试"},
                                         {"event_id": "e1", "quote": "用户要求执行完整测试"}],
                "fulfillment_evidence": [{"event_id": "e2", "quote": "没有测试"}],
            }]}, {"cache_hit": False})

    first = run_semantic_features(batch, tmp_path / "features", "demand_capability_instances",
                                  MockClient(), "mock", None)
    feature = json.loads(Path(first["output_path"]).read_text(encoding="utf-8"))
    instance = feature["values"]["instances"][0]
    assert instance["instance_id"].startswith("capinst_")
    assert instance["fulfillment"] == "unmet"
    assert {item["event_id"] for item in instance["requirement_evidence"]} == {"e1"}
    assert instance["requirement_evidence"] == [{"event_id": "e1", "quote": "运行测试"}]
    assert {item["event_id"] for item in instance["fulfillment_evidence"]} == {"e2"}
    assert feature["api_meta"]["dropped_inexact_evidence_count"] == 1


def test_capability_instance_downgrades_unsupported_fulfillment(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [{
        "trace_id": "tr1", "session_id": "s1", "turn_id": "t1",
        "events": [{"event_id": "e1", "type": "user_message", "data": {"content": "请运行测试"}}],
    }])

    class MockClient:
        model = "mock-model"
        config = object()

        def complete_json(self, system: str, user: str):
            return ({"user_goal": "验证修改", "instances": [{
                "need": "运行测试", "object": "代码", "expected_outcome": "测试通过", "constraints": [],
                "requirement_source": "explicit_user", "fulfillment": "met", "agent_gap": "",
                "candidate_capability": "运行测试",
                "requirement_evidence": [{"event_id": "e1", "quote": "运行测试"}],
                "fulfillment_evidence": [{"event_id": "e1", "quote": "测试已经通过"}],
            }]}, {"cache_hit": False})

    result = run_semantic_features(batch, tmp_path / "features", "demand_capability_instances",
                                   MockClient(), "mock", None)
    record = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    instance = record["values"]["instances"][0]
    assert instance["fulfillment"] == "unclear"
    assert instance["fulfillment_evidence"] == []
    assert record["api_meta"]["downgraded_fulfillment_count"] == 1


def test_api_redaction() -> None:
    value = redact_for_api("sk-abcdefghijklmnop user@example.com 10.0.0.1 /Users/alice/private.txt")
    assert "sk-" not in value
    assert "user@example.com" not in value
    assert "10.0.0.1" not in value
    assert "/Users/alice" not in value
    assert redact_for_api("/mnt/agent-workspace/") == "<LOCAL_PATH>"


def test_searchable_event_text_includes_nested_tool_text() -> None:
    nested = {"content": [{"type": "text", "text": '{"id": "abc", "dir": "/Users/alice/x"}'}]}
    value = searchable_event_text(nested)
    assert '{"id": "abc", "dir": "<LOCAL_PATH>"}' in value
    assert "/Users/alice" not in value


def test_preprocess_redacts_raw_deduplicates_and_registers_rejections(tmp_path: Path) -> None:
    source = tmp_path / "collector.jsonl"
    valid = {"trace_schema_version": "v0.2.0", "event_id": "e1", "session_id": "s1",
             "turn_id": "t1", "sequence": 1, "timestamp": "2026-01-01T00:00:00Z",
             "event_type": "event_msg", "source": {"adapter": "codex_rollout_jsonl"},
             "payload": {"type": "event_msg", "payload": {
                 "type": "task_complete", "text": "sk-abcdefghijklmnop user@example.com"}}}
    source.write_text(json.dumps(valid) + "\n{bad json\n", encoding="utf-8")
    first = run_preprocess([str(source)], tmp_path / "out", None, None)
    trace_text = Path(first["output_path"], "unified_traces.jsonl").read_text(encoding="utf-8")
    assert "sk-" not in trace_text and "user@example.com" not in trace_text
    rejected = Path(first["rejected_records_path"]).read_text(encoding="utf-8")
    assert "JSONDecodeError" in rejected
    second = run_preprocess([str(source)], tmp_path / "out", None, None)
    assert second["deduplicated"] is True
    assert second["batch_id"] == first["batch_id"]
    with (tmp_path / "out" / "data_registry.csv").open(encoding="utf-8-sig") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_basic_token_timing_pairing_and_incremental_skip(tmp_path: Path) -> None:
    batch = tmp_path / "batch_x"
    batch.mkdir()
    write_jsonl(batch / "unified_turns.jsonl", [{"trace_id": "tr1", "session_id": "s1",
        "turn_id": "t1", "status": "completed", "events": [
            {"type": "user_message", "timestamp": "2026-01-01T00:00:00Z", "data": {"content": "go"}},
            {"type": "model_call", "timestamp": "2026-01-01T00:00:01Z", "data": {}},
            {"type": "token_usage", "timestamp": "2026-01-01T00:00:02Z",
             "data": {"prompt_tokens": 10, "completion_tokens": 5}},
            {"type": "tool_call", "data": {"call_id": "c1", "tool_name": "shell", "arguments": "x"}},
            {"type": "tool_result", "data": {"call_id": "missing", "content": "x"}},
        ]}])
    first = run_basic_features(batch, tmp_path / "features")
    values = json.loads(Path(first["output_path"]).read_text(encoding="utf-8"))["values"]
    assert values["token_usage"]["total_tokens"] == 15
    assert values["duration_ms"] == 2000 and values["first_response_ms"] == 1000
    assert values["tool_pairing"]["unmatched_call_ids"] == ["c1"]
    second = run_basic_features(batch, tmp_path / "features")
    assert second["record_count"] == 0 and second["skipped_existing_count"] == 1


def test_basic_incremental_does_not_reuse_a_different_batch(tmp_path: Path) -> None:
    features = tmp_path / "features"
    for batch_name in ("batch_old", "batch_new"):
        batch = tmp_path / batch_name
        batch.mkdir()
        write_jsonl(batch / "unified_turns.jsonl", [{
            "trace_id": "tr1", "session_id": "s1", "turn_id": "stable_t1",
            "status": "completed", "events": [],
        }])
        result = run_basic_features(batch, features)
        assert result["record_count"] == 1
        assert result["skipped_existing_count"] == 0


def test_model_json_prefix_repair() -> None:
    response = {"choices": [{"message": {"content": '{"{"intrinsic_difficulty":"hard","evidence":[]}'}}]}
    assert OpenAICompatibleClient._parse(response)["intrinsic_difficulty"] == "hard"


def test_candidate_core_preserves_all_cases_and_evidence(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.jsonl"
    base_instance = {"candidate_capability": "运行针对性测试", "need": "验证修复", "expected_outcome": "测试通过",
                     "constraints": [], "fulfillment": "unmet", "agent_gap": "未运行测试",
                     "requirement_evidence": [{"event_id": "e1", "quote": "运行测试"}],
                     "fulfillment_evidence": [{"event_id": "e2", "quote": "没有测试"}]}
    write_jsonl(feature_path, [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "feature_set": "demand_capability_instances",
         "feature_version": "v2", "feature_run_id": "r1", "values": {"instances": [{**base_instance, "instance_id": "i1"}]}},
        {"trace_id": "tr2", "session_id": "s2", "turn_id": "t2", "feature_set": "demand_capability_instances",
         "feature_version": "v2", "feature_run_id": "r1", "values": {"instances": [{**base_instance, "instance_id": "i2"}]}},
    ])
    result = build_candidate_core([feature_path], tmp_path / "analysis", "test_v1")
    core = Path(result["output_path"])
    cases = [json.loads(line) for line in (core / "capability_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 2
    assert len(cases[0]["evidence"]) == 2
    with (core / "capability_distribution.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["case_count"] == "2"
    assert row["trace_count"] == "2"


def test_candidate_core_deduplicates_runs_and_matches_previous_taxonomy(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.jsonl"
    instance = {"instance_id": "i1", "candidate_capability": "运行针对性测试", "need": "验证修复",
                "expected_outcome": "测试通过", "constraints": [], "fulfillment": "met", "agent_gap": "",
                "requirement_evidence": [{"event_id": "e1", "quote": "运行测试"}],
                "fulfillment_evidence": [{"event_id": "e2", "quote": "测试通过"}]}
    write_jsonl(feature_path, [
        {"trace_id": "tr1", "session_id": "s1", "turn_id": "t1", "feature_set": "demand_capability_instances",
         "feature_version": "v2", "feature_run_id": run_id,
         "generated_by": {"model": model}, "values": {"instances": [instance]}}
        for run_id, model in (("r1", "m1"), ("r2", "m2"))
    ])
    taxonomy_path = tmp_path / "previous.json"
    taxonomy_path.write_text(json.dumps({"root": {"capability_id": "capability_root", "children": [{
        "capability_id": "cap_existing", "name": "运行针对性测试", "children": []
    }]}}), encoding="utf-8")
    result = build_candidate_core([feature_path], tmp_path / "analysis", "test_v2", taxonomy_path)
    core = Path(result["output_path"])
    cases = [json.loads(line) for line in (core / "capability_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 1
    assert {source["feature_run_id"] for source in cases[0]["source_features"]} == {"r1", "r2"}
    taxonomy = json.loads((core / "capability_taxonomy.json").read_text(encoding="utf-8"))
    node = taxonomy["root"]["children"][0]
    assert node["capability_id"] == "cap_existing"
    assert node["status"] == "matched_existing"


def test_review_core_accepts_and_merges_without_losing_cases(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.jsonl"
    records = []
    for index, capability in enumerate(("测试执行", "运行测试"), 1):
        records.append({"trace_id": f"tr{index}", "session_id": f"s{index}", "turn_id": f"t{index}",
                        "feature_set": "demand_capability_instances", "feature_version": "v2",
                        "feature_run_id": "r1", "values": {"instances": [{
                            "instance_id": f"i{index}", "candidate_capability": capability,
                            "need": "验证修改", "expected_outcome": "测试通过", "constraints": [],
                            "fulfillment": "met", "agent_gap": "",
                            "requirement_evidence": [{"event_id": f"e{index}", "quote": "运行测试"}],
                            "fulfillment_evidence": []}]}})
    write_jsonl(feature_path, records)
    draft = build_candidate_core([feature_path], tmp_path / "draft", "draft_review")
    taxonomy = json.loads(Path(draft["output_path"], "capability_taxonomy.json").read_text(encoding="utf-8"))
    nodes = {node["name"]: node["capability_id"] for node in taxonomy["root"]["children"]}
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps({"reviewer": "tester",
        "publication_policy": {"require_boundaries": False}, "decisions": [
        {"capability_id": nodes["测试执行"], "action": "accept", "reason": "边界明确",
         "definition": "执行与任务相关的测试"},
        {"capability_id": nodes["运行测试"], "action": "merge", "reason": "语义重复",
         "target_capability_id": nodes["测试执行"]},
    ]}), encoding="utf-8")
    result = publish_reviewed_core(Path(draft["output_path"]), decisions_path,
                                   tmp_path / "published", "published_v1")
    assert result["published_capability_count"] == 1
    assert result["published_case_count"] == 2
    cases = _read_jsonl(Path(result["output_path"], "capability_cases.jsonl"))
    assert {case["capability_id"] for case in cases} == {nodes["测试执行"]}
    assert all(case["assignment_status"] == "published" for case in cases)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_review_suggestions_find_merge_and_split_candidates(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.jsonl"
    specs = [
        ("i1", "测试执行", "运行测试验证代码"), ("i2", "测试执行", "运行测试验证代码"),
        ("i3", "执行测试", "运行测试验证代码"), ("i4", "执行测试", "运行测试验证代码"),
        ("i5", "综合处理", "检查测试执行结果"), ("i6", "综合处理", "检查测试执行结果"),
        ("i7", "综合处理", "编辑文档修改文字"), ("i8", "综合处理", "编辑文档修改文字"),
    ]
    records = []
    for index, (instance_id, capability, need) in enumerate(specs, 1):
        records.append({"trace_id": f"tr{index}", "session_id": f"s{index}", "turn_id": f"t{index}",
                        "feature_set": "demand_capability_instances", "feature_version": "v2",
                        "feature_run_id": "r1", "values": {"instances": [{
                            "instance_id": instance_id, "candidate_capability": capability, "need": need,
                            "expected_outcome": need, "constraints": [], "fulfillment": "met", "agent_gap": "",
                            "requirement_evidence": [], "fulfillment_evidence": []}]}})
    write_jsonl(feature_path, records)
    draft = build_candidate_core([feature_path], tmp_path / "draft", "suggestion_draft")
    output = tmp_path / "suggestions.json"
    result = suggest_reviews(Path(draft["output_path"]), output, merge_threshold=0.45,
                             split_similarity_threshold=0.3, split_min_cases=4,
                             split_min_cluster_size=2)
    suggestions = json.loads(output.read_text(encoding="utf-8"))
    assert result["merge_candidate_count"] >= 1
    assert any({item["left_name"], item["right_name"]} == {"测试执行", "执行测试"}
               for item in suggestions["merge_candidates"])
    split = next(item for item in suggestions["split_candidates"] if item["name"] == "综合处理")
    assert sorted(len(cluster["case_ids"]) for cluster in split["clusters"]) == [2, 2]


def test_llm_review_keeps_rule_candidate_and_validates_case_ids(tmp_path: Path) -> None:
    draft_core = tmp_path / "core"
    draft_core.mkdir()
    write_jsonl(draft_core / "capability_cases.jsonl", [
        {"case_id": "i1", "capability_id": "c1", "user_need": "运行测试"},
        {"case_id": "i2", "capability_id": "c2", "user_need": "执行测试"},
    ])
    suggestions = tmp_path / "suggestions.json"
    suggestions.write_text(json.dumps({"source_taxonomy_version": "draft1", "merge_candidates": [{
        "left_capability_id": "c1", "left_name": "运行测试", "left_case_ids": ["i1"],
        "right_capability_id": "c2", "right_name": "执行测试", "right_case_ids": ["i2"],
        "combined_score": 0.9,
    }], "split_candidates": []}), encoding="utf-8")

    class MockClient:
        model = "mock-model"

        def complete_json(self, system: str, user: str):
            return ({"decision": "merge", "reasoning": "边界相同", "boundary_difference": "无",
                     "suggested_name": "测试执行", "inclusion_criteria": ["需要运行测试"],
                     "exclusion_criteria": [], "cited_case_ids": ["i1", "i2"]}, {"cache_hit": False})

    result = review_suggestions_with_llm(draft_core, suggestions, tmp_path / "out", MockClient(), "mock")
    review = _read_jsonl(Path(result["output_path"]))[0]
    assert review["rule_candidate"]["combined_score"] == 0.9
    assert review["llm_review"]["decision"] == "merge"
    assert result["record_count"] == 1


def test_hierarchy_parent_aggregates_leaf_cases(tmp_path: Path) -> None:
    core = tmp_path / "core"
    core.mkdir()
    (core / "capability_taxonomy.json").write_text(json.dumps({
        "taxonomy_version": "flat_v1", "status": "published", "candidates": [],
        "root": {"capability_id": "capability_root", "name": "Coding Agent 能力", "children": [
            {"capability_id": "c_test", "name": "测试执行", "status": "published", "children": []},
            {"capability_id": "c_debug", "name": "错误诊断", "status": "published", "children": []},
        ]}}), encoding="utf-8")
    write_jsonl(core / "capability_cases.jsonl", [
        {"case_id": "i1", "trace_id": "tr1", "capability_id": "c_test", "assignment_status": "published",
         "evidence": [{"event_id": "e1"}]},
        {"case_id": "i2", "trace_id": "tr2", "capability_id": "c_debug", "assignment_status": "published",
         "evidence": [{"event_id": "e2"}, {"event_id": "e3"}]},
    ])
    operations = tmp_path / "hierarchy.json"
    operations.write_text(json.dumps({"reviewer": "tester", "operations": [{
        "action": "create_parent", "capability_id": "c_quality", "name": "质量保障",
        "child_ids": ["c_test", "c_debug"], "reason": "两个能力均服务于修改质量保障"
    }]}), encoding="utf-8")
    result = organize_hierarchy(core, operations, tmp_path / "out", "tree_v1")
    taxonomy = json.loads(Path(result["output_path"], "capability_taxonomy.json").read_text(encoding="utf-8"))
    parent = taxonomy["root"]["children"][0]
    assert parent["capability_id"] == "c_quality"
    assert parent["case_count"] == 2
    assert parent["trace_count"] == 2
    assert parent["evidence_count"] == 3
    assert {child["parent_id"] for child in parent["children"]} == {"c_quality"}
    assert result["max_level"] == 2


def test_update_pipeline_stops_at_human_gate_and_resumes_without_rerun(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"_time_": "now", "content": json.dumps({
        "id": "r1", "model": "m1", "messages": [{"role": "user", "content": "修复并测试"}],
        "response": "完成"
    })}])
    capability_features = tmp_path / "capability_features.jsonl"
    write_jsonl(capability_features, [{
        "trace_id": "tr1", "session_id": "s1", "turn_id": "t1",
        "feature_set": "demand_capability_instances", "feature_version": "v2", "feature_run_id": "r1",
        "values": {"instances": [{"instance_id": "i1", "candidate_capability": "测试执行",
                                    "need": "运行测试", "expected_outcome": "测试通过", "constraints": [],
                                    "fulfillment": "met", "agent_gap": "", "requirement_evidence": [],
                                    "fulfillment_evidence": []}]}
    }])
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "update.json"
    config_path.write_text(json.dumps({
        "sources": [str(source)], "semantic_features": [],
        "enable_human_review": True,
        "historical_capability_features": [str(capability_features)],
        "generate_review_suggestions": True,
        "extensions": ["model_comparison", "harness_comparison", "training_selection", "unmet_need_priority"],
        "roots": {"preprocessed": str(tmp_path / "preprocessed"), "features": str(tmp_path / "features"),
                  "analysis": str(tmp_path / "analysis")},
        "runs_root": str(tmp_path / "runs")
    }), encoding="utf-8")
    first = run_update(config_path)
    assert first["status"] == "waiting_for_human_review"
    manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == {"preprocess", "basic_features", "candidate_core", "review_suggestions"}
    registry = tmp_path / "preprocessed" / "data_registry.csv"
    rows_before = registry.read_text(encoding="utf-8").splitlines()
    resumed = run_update(config_path, resume_run_id=first["run_id"])
    assert resumed["status"] == "waiting_for_human_review"
    assert registry.read_text(encoding="utf-8").splitlines() == rows_before
    resumed_manifest = json.loads(Path(resumed["manifest_path"]).read_text(encoding="utf-8"))
    draft_core = Path(resumed_manifest["stages"]["candidate_core"]["result"]["output_path"])
    draft_taxonomy = json.loads((draft_core / "capability_taxonomy.json").read_text(encoding="utf-8"))
    capability_id = draft_taxonomy["root"]["children"][0]["capability_id"]
    decisions = tmp_path / "review_decisions.json"
    decisions.write_text(json.dumps({"reviewer": "tester", "publication_policy": {
        "min_independent_cases": 1, "min_independent_traces": 1, "min_evidence_count": 0,
        "require_definition": False, "require_boundaries": False}, "decisions": [{
        "capability_id": capability_id, "action": "accept", "reason": "定义和案例证据明确"
    }]}), encoding="utf-8")
    completed = run_update(config_path, resume_run_id=first["run_id"], review_decisions=decisions)
    assert completed["status"] == "completed"
    final_manifest = json.loads(Path(completed["manifest_path"]).read_text(encoding="utf-8"))
    assert final_manifest["stages"]["publish_reviewed_core"]["status"] == "completed"
    assert final_manifest["stages"]["validate_core"]["result"]["status"] == "valid"
    assert all(final_manifest["stages"][f"extension:{name}"]["status"] == "completed" for name in (
        "model_comparison", "harness_comparison", "training_selection", "unmet_need_priority"))
    assert Path(final_manifest["final_core_path"], "capability_taxonomy.json").exists()
    assert registry.read_text(encoding="utf-8").splitlines() == rows_before


def test_all_analysis_extensions_produce_explainable_outputs(tmp_path: Path) -> None:
    core = tmp_path / "core"
    core.mkdir()
    (core / "capability_taxonomy.json").write_text(json.dumps({
        "taxonomy_version": "v1", "root": {"capability_id": "capability_root", "children": [{
            "capability_id": "c1", "name": "测试执行", "children": [], "status": "published"
        }]}}), encoding="utf-8")
    write_jsonl(core / "capability_cases.jsonl", [
        {"case_id": "i1", "trace_id": "tr1", "turn_ids": ["t1"], "capability_id": "c1",
         "assignment_status": "published", "fulfillment": "met", "unmet_need": "", "evidence": [{"event_id": "e1"}]},
        {"case_id": "i2", "trace_id": "tr2", "turn_ids": ["t2"], "capability_id": "c1",
         "assignment_status": "published", "fulfillment": "unmet", "unmet_need": "未运行测试", "evidence": [{"event_id": "e2"}]},
    ])
    feature_file = tmp_path / "all_features.jsonl"
    records = []
    for turn_id, model, harness, outcome, verified, sentiment, difficulty in (
        ("t1", "model-a", "h1", "completed", True, "positive", "hard"),
        ("t2", "model-b", "h2", "failed", False, "negative", "medium"),
    ):
        for feature_set, values in (
            ("basic_turn_features", {"agent_models": [model], "harness": harness}),
            ("task_outcome", {"outcome": outcome, "verified": verified}),
            ("user_feedback", {"sentiment": sentiment}),
            ("difficulty", {"intrinsic_difficulty": difficulty}),
            ("behavior_quality", {"quality": "good" if turn_id == "t1" else "poor"}),
        ):
            records.append({"trace_id": turn_id.replace("t", "tr"), "turn_id": turn_id,
                            "feature_set": feature_set, "values": values})
    write_jsonl(feature_file, records)
    outputs = tmp_path / "extensions"
    results = {name: run_extension(name, core, [feature_file], outputs) for name in (
        "model_comparison", "harness_comparison", "training_selection", "unmet_need_priority")}
    assert all(Path(result["output_path"]).exists() for result in results.values())
    model_rows = list(csv.DictReader(Path(results["model_comparison"]["output_path"]).open(encoding="utf-8-sig")))
    assert {row["model"] for row in model_rows} == {"model-a", "model-b"}
    training = _read_jsonl(Path(results["training_selection"]["output_path"]))
    assert {row["trace_id"]: row["recommended_use"] for row in training if row.get("trace_id")} == {
        "tr1": "positive_sft", "tr2": "repair_trace"}
    assert any(row["recommended_use"] == "preference_pair" for row in training)
    unmet_rows = list(csv.DictReader(Path(results["unmet_need_priority"]["output_path"]).open(encoding="utf-8-sig")))
    assert unmet_rows[0]["priority_tier"] == "insufficient_evidence"


def test_update_daily_mode_completes_draft_without_human_gate(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"_time_": "now", "content": json.dumps({
        "id": "r1", "model": "m1", "messages": [{"role": "user", "content": "运行测试"}],
        "response": "完成"
    })}])
    feature_file = tmp_path / "capabilities.jsonl"
    write_jsonl(feature_file, [{
        "trace_id": "tr1", "session_id": "s1", "turn_id": "t1",
        "feature_set": "demand_capability_instances", "feature_version": "v3", "feature_run_id": "r1",
        "values": {"instances": [{"instance_id": "i1", "candidate_capability": "测试执行",
            "need": "运行测试", "expected_outcome": "测试通过", "constraints": [],
            "fulfillment": "met", "agent_gap": "", "requirement_evidence": [{"event_id": "e1"}],
            "fulfillment_evidence": []}]}
    }])
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "daily.json"
    config.write_text(json.dumps({
        "sources": [str(source)], "semantic_features": [], "enable_human_review": False,
        "historical_capability_features": [str(feature_file)],
        "extensions": ["model_comparison", "training_selection"],
        "roots": {"preprocessed": str(tmp_path / "preprocessed"),
                  "features": str(tmp_path / "features"), "analysis": str(tmp_path / "analysis")},
        "runs_root": str(tmp_path / "runs")
    }), encoding="utf-8")
    result = run_update(config)
    assert result["status"] == "completed"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["final_core_path"].endswith("/core")
    assert manifest["stages"]["extension:model_comparison"]["status"] == "completed"
    assert manifest["stages"]["extension:training_selection"]["status"] == "completed"
