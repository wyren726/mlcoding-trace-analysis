from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from trace_analysis.pipeline.stage_06_trace_quality_labeling.mapping import load_taxonomy
from trace_analysis.pipeline.stage_06_trace_quality_labeling.prompt import build_capability_messages
from trace_analysis.pipeline.stage_06_trace_quality_labeling.runner import run_labeling


COLUMNS = ["是否考虑", "需求", "子需求", "定义", "原子能力", "输入", "输出", "边界", "阶段编号", "阶段", ""]


def table(path: Path, rows=None):
    if rows is None:
        rows = [
            {"需求": "文献调研@作者", "子需求": "检索文献", "定义": "找到相关文献", "原子能力": "应用-实施；分析-区分", "边界": "不做评价", "阶段编号": "1", "阶段": "调研"},
            {"子需求": "核验引用", "定义": "核对引用", "原子能力": "评价-核查"},
            {"需求": "实验设计", "子需求": "制定方案", "定义": "制定实验方案", "原子能力": "创造-计划"},
            {"阶段编号": "8", "阶段": "跨阶段"},
        ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def case_file(path: Path):
    value = {
        "schema_version": "pain-case-v1", "case_id": "case_internal",
        "trace_id": "trace_internal", "episode_id": "episode_internal",
        "session_ids": ["original-session"], "pain_judgment": "confirmed",
        "episode_boundary": {"judgment": "reliable"},
        "agent_verification": {"agent_related": True, "attribution": "execution_or_verification"},
        "evidence_references": [{"event_id": "evt_internal"}],
        "capability_gaps": [{"capability_key": "old_key"}],
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_load_forward_fill_and_exact_names(tmp_path):
    path = table(tmp_path / "taxonomy.csv")
    before = path.read_bytes()
    taxonomy = load_taxonomy(path)
    assert taxonomy.requirements == ["文献调研@作者", "实验设计"]
    assert len(taxonomy.entries) == 3
    assert taxonomy.entries[("文献调研@作者", "核验引用")].source_row == 3
    assert taxonomy.entries[("文献调研@作者", "核验引用")].stage_name == ""
    assert taxonomy.skipped_rows == [5]
    assert taxonomy.sha256 == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_resolve_atoms_locally_and_deduplicate(tmp_path):
    taxonomy = load_taxonomy(table(tmp_path / "taxonomy.csv"))
    choice = {"requirement_name": "文献调研@作者", "subrequirement_name": "检索文献"}
    values = taxonomy.resolve([choice, choice])
    assert len(values) == 1
    assert values[0]["atomic_capabilities"] == ["应用-实施", "分析-区分"]
    assert values[0]["taxonomy_row"] == 2


@pytest.mark.parametrize("choice", [
    {"requirement_name": "实验设计", "subrequirement_name": "检索文献"},
    {"requirement_name": "missing", "subrequirement_name": None},
    {"requirement_name": "实验设计", "subrequirement_name": "制定方案", "atomic_capabilities": ["fake"]},
    {"requirement_name": "实验设计", "subrequirement_name": []},
])
def test_invalid_selections_rejected(tmp_path, choice):
    taxonomy = load_taxonomy(table(tmp_path / "taxonomy.csv"))
    with pytest.raises(ValueError):
        taxonomy.resolve([choice])


def test_parent_only_is_not_all_children(tmp_path):
    taxonomy = load_taxonomy(table(tmp_path / "taxonomy.csv"))
    values = taxonomy.resolve([{"requirement_name": "实验设计", "subrequirement_name": None}])
    assert values[0]["mapping_status"] == "partial"
    assert values[0]["atomic_capabilities"] is None


@pytest.mark.parametrize("rows", [
    [{"子需求": "孤立", "定义": "无父级", "原子能力": "核查"}],
    [{"需求": "父", "子需求": "子", "定义": "定义", "原子能力": ""}],
    [{"需求": "父", "子需求": "子", "定义": "定义", "原子能力": "核查"}] * 2,
])
def test_invalid_csv_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        load_taxonomy(table(tmp_path / "taxonomy.csv", rows))


def test_prompt_uses_definitions_not_atomic_choices(tmp_path):
    taxonomy = load_taxonomy(table(tmp_path / "taxonomy.csv"))
    messages = build_capability_messages({"user_goal": "查文献"}, taxonomy)
    payload = json.loads(messages[1]["content"])
    assert payload["taxonomy"][0]["boundary"] == "不做评价"
    assert "atomic_capabilities" not in payload["taxonomy"][0]


def test_prepare_is_not_semantic_success_and_resume(tmp_path):
    source = case_file(tmp_path / "cases.jsonl")
    taxonomy = table(tmp_path / "taxonomy.csv")
    before = source.read_bytes()
    result = run_labeling(source, tmp_path / "run", taxonomy_path=taxonomy)
    assert result["prepared"] == 1
    assert result["semantic_completed"] == 0
    output = Path(result["output"])
    record = json.loads(output.read_text())
    assert record["labels"]["trace_quality"] == "unknown"
    assert record["labels"]["episode_integrity"] == "unknown"
    assert record["labels"]["failure_pattern"] == ["unknown"]
    assert record["labels"]["capability_mapping"] == []
    assert record["reused"]["capability_gaps"][0]["capability_key"] == "old_key"
    assert record["label_provenance"]["taxonomy_sha256"] == load_taxonomy(taxonomy).sha256
    assert record["label_provenance"]["labeling_status"] == "pending_semantic_analysis"
    run_labeling(source, tmp_path / "run", taxonomy_path=taxonomy)
    assert len(output.read_text().splitlines()) == 1
    assert source.read_bytes() == before


def test_changed_taxonomy_or_input_requires_new_run(tmp_path):
    source = case_file(tmp_path / "cases.jsonl")
    taxonomy = table(tmp_path / "taxonomy.csv")
    result = run_labeling(source, tmp_path / "run", taxonomy_path=taxonomy)
    output_before = Path(result["output"]).read_bytes()
    taxonomy.write_text(taxonomy.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="changed"):
        run_labeling(source, tmp_path / "run", taxonomy_path=taxonomy)
    assert Path(result["output"]).read_bytes() == output_before
    with pytest.raises(ValueError, match="overwriting"):
        run_labeling(source, tmp_path / "run", resume=False, taxonomy_path=taxonomy)


@pytest.mark.parametrize("selection", [
    [{"requirement_name": "实验设计", "subrequirement_name": "制定方案"}],
    [{"taxonomy_row": True}], [{"taxonomy_row": "2"}],
    [{"taxonomy_row": 5}], [{"taxonomy_row": 999}],
    [{"taxonomy_row": 2, "requirement_name": "invented"}], None,
])
def test_model_boundary_accepts_only_existing_rows(tmp_path, selection):
    taxonomy = load_taxonomy(table(tmp_path / "taxonomy.csv"))
    with pytest.raises(ValueError):
        taxonomy.resolve_rows(selection)


def test_row_lookup_preserves_source_names(tmp_path):
    taxonomy = load_taxonomy(table(tmp_path / "taxonomy.csv"))
    mapped = taxonomy.resolve_rows([{"taxonomy_row": 2}, {"taxonomy_row": 2}])
    assert len(mapped) == 1
    assert mapped[0]["requirement_name"] == "文献调研@作者"
    assert mapped[0]["subrequirement_name"] == "检索文献"
    assert mapped[0]["atomic_capabilities"] == ["应用-实施", "分析-区分"]


def test_selected_retry_counts_all_successes_and_preserves_prefix(tmp_path):
    source = case_file(tmp_path / "cases.jsonl")
    source.write_text(source.read_text() * 3)
    taxonomy = table(tmp_path / "taxonomy.csv")
    first = run_labeling(source, tmp_path / "run", taxonomy_path=taxonomy, include_lines={1, 3})
    output = Path(first["output"])
    prefix = output.read_bytes()
    assert first["prepared"] == 2
    second = run_labeling(source, tmp_path / "run", taxonomy_path=taxonomy, include_lines={2})
    assert second["prepared"] == 3
    assert second["total"] == 3
    assert output.read_bytes().startswith(prefix)
    assert len(output.read_text().splitlines()) == 3
    progress = json.loads((tmp_path / "run/progress__run.json").read_text())
    assert progress["pending"] == 0
    assert progress["attempted_this_run"] == 1


def test_semantic_retry_saves_gates_and_rejects_unknown_enum(tmp_path):
    import copy
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.runner import GATE_KEYS

    response = {
        "quality_gates": {**dict.fromkeys(GATE_KEYS, None), "episode_integrity": "unknown"},
        "quality_gate_reasons": dict.fromkeys(GATE_KEYS, "材料不足，无法判断"),
        "episode_integrity": "unknown", "evidence_status": "unknown",
        "domain_primary": "计算机", "domain_secondary": [],
        "research_stage_primary": "unknown", "research_stage_secondary": [],
        "difficulty": "unknown", "difficulty_factors": [], "failure_pattern": [],
        "capability_mapping": [{"taxonomy_row": 2}], "reason": "仅支持文献检索需求",
    }

    class Client:
        model = "fake"
        calls = []

        def complete_json(self, system, user):
            self.calls.append(user)
            result = copy.deepcopy(response)
            if len(self.calls) == 1:
                result["domain_secondary"] = ["医学"]
            return result, {}

    client = Client()
    source = case_file(tmp_path / "cases.jsonl")
    raw = tmp_path / "run" / "_internal" / "session.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"type": "user", "message": "test"}) + "\n", encoding="utf-8")
    value = json.loads(source.read_text())
    turns = tmp_path / "unified_turns.jsonl"
    turns.write_text(json.dumps({"trace_id": "trace_internal", "turn_id": "t1", "turn_index": 1,
                                "events": [{"event_id": "e1", "type": "assistant_message", "content": "test"}]}) + "\n")
    value["lineage"] = {"preprocessed_input": str(turns)}
    value["episode_boundary"] = {"start_turn_id": "t1", "end_turn_id": "t1"}
    value["source_references"] = [{"source": {"input_file": str(raw), "record_index": 1}}]
    source.write_text(json.dumps(value) + "\n", encoding="utf-8")
    result = run_labeling(source, tmp_path / "run",
                          taxonomy_path=table(tmp_path / "taxonomy.csv"), client=client)
    assert result["semantic_completed"] == 1
    saved = json.loads(Path(result["output"]).read_text())
    assert len(client.calls) == 2
    assert "Invalid domain_secondary" in client.calls[1]
    assert saved["quality_gates"] == response["quality_gates"]
    assert saved["quality_gate_reasons"] == response["quality_gate_reasons"]
    assert saved["labels"]["trace_quality"] == "unknown"
    assert saved["labels"]["capability_mapping"][0]["subrequirement_name"] == "检索文献"
    assert saved["label_provenance"]["model_metadata"]["validation_retries"] == 1
    assert saved["stage_04_labels"] == value


def test_excluded_skipped_preserving_original_line(tmp_path):
    source = case_file(tmp_path / "cases.jsonl")
    row = json.loads(source.read_text())
    source.write_text(json.dumps({**row, "pain_judgment": "excluded"}) + "\n" + json.dumps(row) + "\n")
    result = run_labeling(source, tmp_path / "output")
    assert result["total"] == 1
    assert json.loads(Path(result["output"]).read_text())["source"]["source_line_number"] == 2


def test_episode_uses_boundaries_not_evidence_range():
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.runner import _episode_payload
    from trace_analysis.pipeline.stage_02_analyze.runner import TraceBundle, candidate_agent_payload
    bundle = TraceBundle("batch", "trace", tuple({"turn_id": f"t{i}", "turn_index": i,
        "events": [{"type": "assistant_message", "event_id": f"e{i}", "content": str(i)}]} for i in range(4)))
    row = {"episode_id": "episode", "query_turn_id": "t1", "episode_boundary": {"start_turn_id": "t1", "end_turn_id": "t3"},
           "source_references": [{"turn_id": "t2", "source": {"record_index": 0}}]}
    payload = _episode_payload(bundle, row)
    assert [t["turn_id"] for t in payload["episode_turns"]] == ["t1", "t2", "t3"]
    assert payload == candidate_agent_payload(bundle, {"task_episodes": [{k: row[k] for k in ("episode_id", "query_turn_id", "episode_boundary")} | {"route": {"judgment": "analyze"}}]})
    row["episode_boundary"] = {}
    with pytest.raises(ValueError, match="authoritative"):
        _episode_payload(bundle, row)


def test_long_episode_retrieves_again_when_evidence_insufficient(tmp_path, monkeypatch):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling import runner
    from trace_analysis.pipeline.stage_02_analyze import runner as stage03
    episode = {"trace_id": "t", "episode_evidence_spans": [], "candidate_episodes": [],
        "episode_turns": [{"turn_id": "t1", "events": [{"type": "tool_result", "event_id": "e1", "data": {"content": "x" * 210000}}]}]}
    selections = []
    def select(client, partition, **kwargs):
        selections.append(kwargs["retrieval_context"].copy())
        return [(partition, {"selected_tool_event_ids": [], "reason": "test"}, {})]
    monkeypatch.setattr(stage03, "complete_index_partition_with_fallback", select)
    calls = []
    def label(row, taxonomy, client, evidence):
        calls.append(evidence)
        return {"labels": {"evidence_status": "insufficient"}, "quality_gates": {"evidence_sufficient": None},
                "quality_gate_reasons": {"evidence_sufficient": "Need actual output"}, "reason": "unclear", "metadata": {}}
    monkeypatch.setattr(runner, "_semantic_labels_once", label)
    result = runner._semantic_labels({}, load_taxonomy(table(tmp_path / "taxonomy.csv")), object(), episode)
    assert len(calls) == 2
    assert any(item.get("round") == 2 for item in selections)
    assert result["metadata"]["full_episode_read"] is False
    assert result["labels"]["evidence_status"] == "insufficient"


def test_run_lock_blocks_second_writer(tmp_path):
    import fcntl
    output = tmp_path / "output"
    output.mkdir()
    with (output / ".labeling.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active writer"):
            run_labeling(case_file(tmp_path / "cases.jsonl"), output)


@pytest.mark.parametrize("invalid", ["true", 1, [], {}])
def test_quality_gates_do_not_accept_truthy_values(invalid):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.runner import GATE_KEYS, _quality_from_gates
    gates = {**dict.fromkeys(GATE_KEYS, True), "episode_integrity": "complete"}
    gates["outcome_observable"] = invalid
    with pytest.raises(ValueError):
        _quality_from_gates(gates)
