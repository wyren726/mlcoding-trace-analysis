import json

from trace_analysis.analysis.core.instance_pool import prepare_instance_pool


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_instance_pool_merges_and_reports_coverage(tmp_path):
    scope = tmp_path / "scope"
    (scope / "filtered_features").mkdir(parents=True)
    (scope / "scope_summary.json").write_text(json.dumps({
        "status": "completed", "unaudited_included": 0, "scope_version": "v3", "included": 1,
    }), encoding="utf-8")
    _write(scope / "filtered_features" / "batch.jsonl", [{
        "feature_set": "demand_capability_instances", "feature_version": "v4",
        "trace_id": "tr1", "session_id": "s1", "turn_id": "t1",
        "values": {"instances": [{"instance_id": "i1", "need": "n",
                                    "ml_llm_coding_scope": {"scope": "included",
                                                             "scope_version": "v3_strict"}}]},
    }])
    _write(tmp_path / "features" / "difficulty" / "consolidated" / "batch.jsonl", [{
        "feature_set": "difficulty", "feature_version": "v2", "turn_id": "t1",
    }])
    result = prepare_instance_pool([scope], tmp_path / "out", tmp_path / "features")
    assert result["instance_count"] == 1
    assert result["turn_record_count"] == 1
    assert result["turn_feature_coverage"] == {"difficulty": 1}
    index = json.loads((tmp_path / "out" / "instance_index.jsonl").read_text())
    assert index["available_turn_features"] == {"difficulty": "v2"}
    pooled = json.loads((tmp_path / "out" / "demand_capability_instances.jsonl").read_text())
    assert pooled["feature_set"] == "demand_capability_instances"
    assert pooled["values"]["instances"][0]["instance_id"] == "i1"
