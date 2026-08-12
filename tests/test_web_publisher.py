import json
from pathlib import Path

import pytest

from trace_analysis.web import publish_capability_site


def _fixture_core(tmp_path: Path) -> Path:
    core = tmp_path / "core"
    core.mkdir()
    taxonomy = {
        "taxonomy_version": "test_v1",
        "status": "analysis_snapshot",
        "generated_at": "2026-08-11T00:00:00+08:00",
        "root": {
            "capability_id": "capability_root", "name": "根能力", "query_count": 1,
            "children": [{
                "capability_id": "cap_leaf", "name": "验证数据集", "definition": "检查数据完整性。",
                "path": ["根能力", "验证数据集"], "query_count": 1, "is_leaf": True, "children": [],
            }],
        },
    }
    (core / "capability_taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False), encoding="utf-8")
    case = {
        "capability_id": "cap_leaf", "case_id": "case_1", "source_case_id": "source_1",
        "trace_id": "trace_1", "turn_ids": ["turn_1"], "user_need": "验证数据集",
        "fulfillment": "met", "assignment_status": "published",
        "assignment_support": {"support_type": "direct", "entailment_audit": "supported"},
        "evidence": [
            {"event_id": "event_1", "evidence_type": "requirement", "quote": "请检查数据集是否完整"},
            {"event_id": "event_2", "evidence_type": "fulfillment", "quote": "检查通过"},
        ],
    }
    (core / "capability_cases.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    (core / "capability_distribution.csv").write_text("capability_id,query_count\ncap_leaf,1\n", encoding="utf-8")
    return core


def test_publish_capability_site_builds_self_contained_snapshot(tmp_path: Path):
    core = _fixture_core(tmp_path)
    output = tmp_path / "site"
    result = publish_capability_site(core, output, "能力调研测试")

    assert result["status"] == "success"
    assert result["leaf_count"] == 1
    assert (output / "index.html").is_file()
    assert (output / "app.js").is_file()
    assert (output / "styles.css").is_file()
    html = (output / "index.html").read_text(encoding="utf-8")
    javascript = (output / "app.js").read_text(encoding="utf-8")
    assert "能力分布" in html
    assert "能力森林" not in html
    assert "能力关系图" in html
    assert "d3.forceSimulation" in javascript
    assert "d3.forceLink" in javascript
    assert "d3.drag" in javascript
    assert 'class="id-row"' in javascript
    cases = json.loads((output / "data/cases/cap_leaf.json").read_text())
    assert cases[0]["user_query"] == ["请检查数据集是否完整"]
    assert cases[0]["requirement_origin"] == "direct"
    assert "source_features" not in cases[0]
    evidence_index = json.loads((output / "data/evidence-index.json").read_text())
    assert evidence_index[0]["capability_name"] == "验证数据集"
    assert evidence_index[0]["user_query"] == ["请检查数据集是否完整"]


def test_publish_requires_complete_core(tmp_path: Path):
    core = tmp_path / "core"
    core.mkdir()
    with pytest.raises(ValueError, match="missing required files"):
        publish_capability_site(core, tmp_path / "site", "test")
