import json
from pathlib import Path

from trace_analysis.analysis.core.dynamic import _validate_atomic, _validate_merge, build_dynamic_taxonomy
from trace_analysis.features.custom.semantic import add_negotiation_ids, is_negotiation_candidate, validate


class FakeClient:
    model = "fake-model"

    def complete_json(self, system: str, user: str):
        payload = json.loads(user)
        if "cases" in payload:
            decisions = []
            for case in payload["cases"]:
                capabilities = [{
                    "name": "将数学机制实现为可执行代码",
                    "definition": "把研究描述中的数学机制转换为可运行实现并验证基本行为",
                    "inclusion_criteria": "明确要求实现数学或算法机制",
                    "exclusion_criteria": "仅讨论理论且不要求计算实现",
                    "support_type": "direct",
                    "supporting_event_ids": [case["evidence"][0]["event_id"]],
                }]
                if case["case_id"] == "case_1":
                    capabilities.append({
                        "name": "构造最小实验验证算法实现",
                        "definition": "设计最小实验检查算法实现是否符合预期",
                        "inclusion_criteria": "明确要求实验验证实现",
                        "exclusion_criteria": "只要求编写实现而不验证",
                        "support_type": "direct",
                        "supporting_event_ids": [case["evidence"][0]["event_id"]],
                    })
                decisions.append({"case_id": case["case_id"], "status": "included",
                                  "reason": "包含明确编码行为", "capabilities": capabilities})
            return {"decisions": decisions}, {"cache_hit": False}
        if "proposals" in payload:
            by_name = {}
            for proposal in payload["proposals"]:
                by_name.setdefault(proposal["name"], []).append(proposal["id"])
            groups = []
            for name, members in by_name.items():
                proposal = next(row for row in payload["proposals"] if row["id"] == members[0])
                groups.append({"name": name, "definition": proposal["definition"],
                               "inclusion_criteria": proposal["inclusion_criteria"],
                               "exclusion_criteria": proposal["exclusion_criteria"],
                               "member_ids": members})
            return {"groups": groups}, {"cache_hit": False}
        if "assignments" in payload:
            return {"decisions": [{"proposal_id": row["proposal_id"], "supported": True,
                                    "reason": "用户证据直接支持"} for row in payload["assignments"]]}, {
                                        "cache_hit": False}
        if "capabilities" in payload and payload["capabilities"] and "capability_id" in payload["capabilities"][0]:
            return {"decisions": [{"capability_id": row["capability_id"], "action": "keep",
                                    "reason": "通用原子行为"} for row in payload["capabilities"]]}, {
                                        "cache_hit": False}
        raise AssertionError("Unexpected request")


class StrictEntailmentClient(FakeClient):
    def complete_json(self, system: str, user: str):
        payload = json.loads(user)
        if "cases" in payload:
            return {"decisions": [{
                "case_id": case["case_id"], "status": "included", "reason": "候选提取",
                "capabilities": [{"name": "获取并验证数据集", "definition": "下载并检查数据集",
                                  "inclusion_criteria": "明确要求获取并检查数据",
                                  "exclusion_criteria": "仅提出宽泛复现目标",
                                  "support_type": "direct",
                                  "supporting_event_ids": [case["evidence"][0]["event_id"]]}]
            } for case in payload["cases"]]}, {"cache_hit": False}
        if "assignments" in payload:
            return {"decisions": [{"proposal_id": row["proposal_id"],
                                    "supported": "下载数据集" in row["user_need"],
                                    "reason": "只保留原文直接表达的行为"}
                                   for row in payload["assignments"]]}, {"cache_hit": False}
        return super().complete_json(system, user)


def test_merge_validation_coalesces_duplicate_names():
    base = {"name": "解析训练日志", "definition": "提取训练指标",
            "inclusion_criteria": "要求解析日志", "exclusion_criteria": "不含日志"}
    groups = _validate_merge({"groups": [
        {**base, "member_ids": ["a"]},
        {**base, "definition": "读取日志并提取指标", "member_ids": ["b"]},
    ]}, {"a", "b"})
    assert len(groups) == 1
    assert groups[0]["member_ids"] == ["a", "b"]


def test_requirement_negotiation_links_accepted_plan_item_with_exact_evidence():
    values = {
        "user_goals": [{"description": "复现论文", "evidence": [{"event_id": "u1", "quote": "复现论文"}]}],
        "plan_items": [{"local_id": "p1", "description": "下载并验证数据集",
                        "proposal_evidence": [{"event_id": "a1", "quote": "先下载并验证数据集"}]}],
        "responses": [{"response_type": "explicit_acceptance", "accepted_item_ids": ["p1"],
                       "rejected_item_ids": [], "response_evidence": [{"event_id": "u2", "quote": "可以，开始吧"}],
                       "reason": "用户明确接受计划"}],
    }
    validate("requirement_negotiation", values, {"u1", "a1", "u2"},
             {"u1": "我要复现论文", "a1": "先下载并验证数据集", "u2": "可以，开始吧"})
    stable = add_negotiation_ids("turn_1", values)
    item_id = stable["plan_items"][0]["item_id"]
    assert stable["responses"][0]["accepted_item_ids"] == [item_id]


def test_negotiation_prefilter_requires_plan_and_follow_up():
    turn = {"events": [{"type": "assistant_message", "data": {"content": "计划分两步：\n1. 下载数据\n2. 运行实验"}}]}
    follow_up = [{"type": "user_message", "data": {"content": "可以，开始吧"}}]
    assert is_negotiation_candidate(turn, follow_up)
    assert not is_negotiation_candidate(turn, [])
    assert not is_negotiation_candidate(
        {"events": [{"type": "assistant_message", "data": {"content": "你好"}}]}, follow_up)


def test_negotiated_atomic_capability_requires_both_sides_of_evidence_chain():
    expectation = {"case": {"event_ids": {"proposal", "accept"},
                            "proposal_event_ids": {"proposal"},
                            "acceptance_event_ids": {"accept"}, "negotiated": True}}
    capability = {"name": "下载并验证数据集", "definition": "下载并检查数据",
                  "inclusion_criteria": "用户接受该计划项", "exclusion_criteria": "计划未被接受",
                  "support_type": "negotiated", "supporting_event_ids": ["proposal", "accept"]}
    rows = _validate_atomic({"decisions": [{"case_id": "case", "status": "included",
                                             "reason": "证据链完整", "capabilities": [capability]}]}, expectation)
    assert rows[0]["capabilities"][0]["support_type"] == "negotiated"


def test_dynamic_taxonomy_is_global_atomic_and_multilabel(tmp_path: Path):
    core = tmp_path / "source" / "core"
    core.mkdir(parents=True)
    taxonomy = {"root": {"capability_id": "capability_root", "name": "能力", "children": [{
        "capability_id": "old", "name": "基于流模型的聚类与路由", "children": []
    }]}}
    (core / "capability_taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    cases = [
        {"case_id": "case_1", "capability_id": "old", "trace_id": "t1", "turn_ids": ["q1"],
         "user_need": "实现流匹配方法并做最小实验",
         "evidence": [{"event_id": "e1", "quote": "实现流匹配方法并做最小实验",
                       "evidence_type": "requirement"}]},
        {"case_id": "case_2", "capability_id": "old", "trace_id": "t2", "turn_ids": ["q2"],
         "user_need": "实现扩散模型中的约束项",
         "evidence": [{"event_id": "e2", "quote": "实现扩散模型中的约束项",
                       "evidence_type": "requirement"}]},
    ]
    (core / "capability_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")

    result = build_dynamic_taxonomy(core, tmp_path / "analysis", FakeClient(), "fake", "v2")
    output = Path(result["output_path"])
    produced = json.loads((output / "capability_taxonomy.json").read_text(encoding="utf-8"))
    names = [node["name"] for node in produced["root"]["children"]]
    assert "基于流模型的聚类与路由" not in names
    assert names.count("将数学机制实现为可执行代码") == 1
    assert all(node["status"] == "supported" for node in produced["root"]["children"])
    assert result["multi_label_source_cases"] == 1
    mappings = [json.loads(line) for line in (output / "capability_cases.jsonl").read_text().splitlines()]
    assert len(mappings) == 3
    assert len({row["case_id"] for row in mappings}) == 3
    assert {row["source_case_id"] for row in mappings} == {"case_1", "case_2"}


def test_broad_reproduction_goal_does_not_support_dataset_capability(tmp_path: Path):
    core = tmp_path / "source" / "core"
    core.mkdir(parents=True)
    taxonomy = {"root": {"capability_id": "capability_root", "name": "能力", "children": [{
        "capability_id": "old", "name": "论文复现", "children": []
    }]}}
    (core / "capability_taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    cases = [
        {"case_id": "broad", "capability_id": "old", "trace_id": "t1", "turn_ids": ["q1"],
         "user_need": "启动论文复现工作流",
         "evidence": [{"event_id": "e1", "quote": "我要开启论文复现的研究",
                       "evidence_type": "requirement"}]},
        {"case_id": "explicit", "capability_id": "old", "trace_id": "t2", "turn_ids": ["q2"],
         "user_need": "下载数据集并检查完整性",
         "evidence": [{"event_id": "e2", "quote": "请下载数据集并检查完整性",
                       "evidence_type": "requirement"}]},
    ]
    (core / "capability_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")
    result = build_dynamic_taxonomy(core, tmp_path / "analysis", StrictEntailmentClient(), "fake", "strict")
    mappings = [json.loads(line) for line in
                Path(result["output_path"], "capability_cases.jsonl").read_text().splitlines()]
    assert {row["source_case_id"] for row in mappings} == {"explicit"}
