from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _turn_summary(batch_dirs: list[Path]) -> tuple[int, Counter[str]]:
    count = 0
    statuses: Counter[str] = Counter()
    for batch_dir in batch_dirs:
        with (batch_dir / "unified_turns.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                turn = json.loads(line)
                count += 1
                statuses[str(turn.get("status") or "unknown")] += 1
    return count, statuses


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _registry_row(batch_dir: Path) -> dict[str, str]:
    registry = batch_dir.parent / "data_registry.csv"
    if not registry.exists():
        return {}
    with registry.open(encoding="utf-8-sig") as handle:
        return next((row for row in csv.DictReader(handle) if row.get("batch_id") == batch_dir.name), {})


def _write_demand_chart(top_nodes: list[dict[str, Any]], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trace_analysis_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    names = [str(node["name"]) for node in top_nodes]
    counts = [int(node.get("case_count") or 0) for node in top_nodes]
    height = max(5.5, len(names) * 0.55)
    fig, ax = plt.subplots(figsize=(12, height))
    positions = list(range(len(names)))
    bars = ax.barh(positions, counts, color="#4C78A8")
    ax.set_yticks(positions, labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("需求案例数")
    ax.set_title("科研 ML/LLM Coding 一级能力需求分布")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.bar_label(bars, labels=[str(value) for value in counts], padding=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_capability_report(core_dir: Path, batch_dirs: Path | list[Path], output: Path,
                               scope_summary: Path | None = None,
                               grounding_summary: Path | None = None) -> dict[str, Any]:
    if isinstance(batch_dirs, Path):
        batch_dirs = [batch_dirs]
    taxonomy = json.loads((core_dir / "capability_taxonomy.json").read_text(encoding="utf-8"))
    cases = _jsonl(core_dir / "capability_cases.jsonl")
    turn_count, turn_statuses = _turn_summary(batch_dirs)
    registries = [_registry_row(batch_dir) for batch_dir in batch_dirs]
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_capability[str(case["capability_id"])].append(case)

    nodes: list[dict[str, Any]] = []
    def collect(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node.get("children") or []:
            collect(child)
    for child in taxonomy["root"].get("children") or []:
        collect(child)
    leaves = [node for node in nodes if not node.get("children")]
    evidence_total = sum(len(case.get("evidence") or []) for case in cases)
    source_case_count = len({str(case.get("source_case_id") or case.get("case_id")) for case in cases})
    fulfillment = Counter(case.get("fulfillment", "unknown") for case in cases)
    leaf_support = Counter(int(node.get("case_count") or 0) for node in leaves)
    model_counts: Counter[str] = Counter()
    harness_counts: Counter[str] = Counter()
    sources = []
    for registry in registries:
        model_counts.update(json.loads(registry.get("model_distribution") or "{}").get("counts", {}))
        harness_counts.update(json.loads(registry.get("harness_distribution") or "{}").get("counts", {}))
        sources.extend(json.loads(registry.get("source") or "[]"))
    source_names = [Path(item).name for item in sources]
    batch_names = [batch_dir.name for batch_dir in batch_dirs]
    raw_size_bytes = sum(int(row.get("raw_size_bytes") or 0) for row in registries)
    session_count = sum(int(row.get("session_count") or 0) for row in registries)
    event_count = sum(int(row.get("event_count") or 0) for row in registries)

    root_name = str(taxonomy["root"].get("name") or "Coding Agent 能力")
    lines = [f"# {root_name}分布分析报告", "", "## 1. 数据说明", "",
             f"本报告基于预处理批次 `{', '.join(batch_names)}` 及能力体系版本 "
             f"`{taxonomy.get('taxonomy_version')}` 生成。", "",
             f"- 原始数据：{', '.join(source_names) or '未登记'}",
             f"- 原始数据大小：{raw_size_bytes / 1024 / 1024:.1f} MB",
             f"- Session：{session_count} 个；统一 Event：{event_count} 个",
             f"- 有效 Turn：{turn_count} 个（" + "、".join(f"{k} {v} 个" for k, v in turn_statuses.items()) + "）",
             f"- 纳入需求案例：{source_case_count} 个；案例—能力映射：{len(cases)} 条；"
             f"能力节点：{len(nodes)} 个；叶子能力：{len(leaves)} 个",
             f"- Agent 模型分布：`{json.dumps(dict(model_counts), ensure_ascii=False)}`",
             f"- Harness 分布：`{json.dumps(dict(harness_counts), ensure_ascii=False)}`",
             f"- 需求满足分布：`{json.dumps(dict(fulfillment), ensure_ascii=False)}`", "",
             f"叶子能力支持度为：{dict(sorted(leaf_support.items()))}。Query 数只表示当前数据中的需求频次，"
             "不用于区分正式或候选能力。"]
    if grounding_summary:
        grounding = json.loads(grounding_summary.read_text(encoding="utf-8"))
        lines.extend(["", "### 用户需求证据与语义落地审计", "",
                      f"- 待审计需求实例：{grounding.get('total_instances', 0)} 个",
                      f"- 用户原文足以支持：{grounding.get('grounded', 0)} 个",
                      f"- 存在过度推断：{grounding.get('ungrounded', 0)} 个",
                      f"- 暂不采用（不确定）：{grounding.get('uncertain', 0)} 个",
                      f"- 未审计：{grounding.get('unclassified', 0)} 个",
                      "- 审计口径：证据必须来自用户消息，且归纳后的需求对象、动作和关键约束必须能由用户原文直接陈述或必然推出。"])
    if scope_summary:
        scope = json.loads(scope_summary.read_text(encoding="utf-8"))
        lines.extend(["", "### 科研 ML/LLM Coding 范围筛选", "",
                      f"- 待判定需求实例：{scope.get('total_instances', 0)} 个",
                      f"- 纳入能力统计：{scope.get('included', 0)} 个",
                      f"- 排除：{scope.get('excluded', 0)} 个",
                      f"- 暂不统计（不确定）：{scope.get('uncertain', 0)} 个",
                      f"- 未分类：{scope.get('unclassified', 0)} 个",
                      "- 纳入口径：需求明确涉及 ML/LLM 模型代码、数据管线、训练、微调、推理、评测、部署、实验复现或故障诊断。",
                      "- 排除口径：纯写作排版、普通文献综述、领域问答、普通统计/绘图、通用软件开发及仅有概念讨论的需求。"])
    lines.extend(["", "## 2. 能力分布", "",
                  "叶子能力名称后的数字表示支持该能力的独立用户 Query 数；所有叶子采用相同展示方式。",
                  "", "```text", root_name])

    def render_tree(children: list[dict[str, Any]], prefix: str = "") -> None:
        for index, node in enumerate(children):
            last = index == len(children) - 1
            connector = "└── " if last else "├── "
            children = node.get("children") or []
            label = str(node["name"])
            if not children:
                query_count = int(node.get("query_count") or len({turn for case in by_capability.get(str(node["capability_id"]), [])
                                                                  for turn in case.get("turn_ids") or []}))
                label += f"（{query_count}）"
            lines.append(prefix + connector + label)
            render_tree(children, prefix + ("    " if last else "│   "))

    render_tree(taxonomy["root"].get("children") or [])
    lines.extend(["```", "", "## 3. 能力与证据", ""])

    evidence_rows = 0
    leaf_sections = 0
    fulfillment_labels = {"met": "已满足", "partially_met": "部分满足", "unmet": "未满足", "unclear": "无法判断"}
    evidence_labels = {"requirement": "用户需求", "fulfillment": "Agent执行或结果", "unspecified": "其他证据"}
    evidence_labels.update({"negotiation_proposal": "Agent提出的计划项",
                            "requirement_acceptance": "用户接受计划的证据"})

    def render(node: dict[str, Any], numbers: list[int]) -> None:
        nonlocal evidence_rows, leaf_sections
        level = min(2 + len(numbers), 6)
        number = ".".join(map(str, numbers))
        lines.extend([f"{'#' * level} {number} {node['name']}", "",
                      "**释义：**  ", str(node.get("definition") or "当前数据支持下的能力。"), ""])
        children = node.get("children") or []
        if children:
            for index, child in enumerate(children, 1):
                render(child, numbers + [index])
            return
        leaf_sections += 1
        node_cases = by_capability.get(str(node["capability_id"]), [])
        lines.extend([f"- 能力状态：`{node.get('status', 'unknown')}`",
                      f"- 支持案例数：{len(node_cases)}", "",
                      "用户直接提出的需求，或“Agent计划项 + 用户接受证据”共同支持能力归属；"
                      "Agent执行或结果仅用于说明满足情况，不单独支持能力归属。", "",
                      "| 案例 | 用户需求 | 满足情况 | 上下文类型 | 证据原文 | Trace / Turn / Event |",
                      "|---|---|---|---|---|---|"])
        for case in node_cases:
            index_prefix = f"{case.get('trace_id')} / {','.join(case.get('turn_ids') or [])} / "
            for evidence in case.get("evidence") or []:
                evidence_rows += 1
                lines.append("| " + " | ".join([
                    _cell(case.get("case_id")), _cell(case.get("user_need")),
                    _cell(fulfillment_labels.get(case.get("fulfillment"), case.get("fulfillment"))),
                    _cell(evidence_labels.get(evidence.get("evidence_type"), evidence.get("evidence_type"))),
                    _cell(evidence.get("quote")), _cell(index_prefix + str(evidence.get("event_id"))),
                ]) + " |")
        if not node_cases:
            lines.append("| — | — | — | — | 当前没有直接案例 | — |")
        lines.append("")

    for index, node in enumerate(taxonomy["root"].get("children") or [], 1):
        render(node, [index])
    if leaf_sections != len(leaves) or evidence_rows != evidence_total:
        raise ValueError(f"Report coverage mismatch: leaves {leaf_sections}/{len(leaves)}, "
                         f"evidence {evidence_rows}/{evidence_total}")
    output.parent.mkdir(parents=True, exist_ok=True)
    chart_path = output.parent / "demand_distribution.png"
    priority_path = output.parent / "unmet_demand_priority.csv"
    top_nodes = taxonomy["root"].get("children") or []
    _write_demand_chart(top_nodes, chart_path)

    priority_rows = []
    for leaf in leaves:
        leaf_cases = by_capability.get(str(leaf["capability_id"]), [])
        status_counts = Counter(str(case.get("fulfillment") or "unclear") for case in leaf_cases)
        known = status_counts["met"] + status_counts["partially_met"] + status_counts["unmet"]
        completion_rate = ((status_counts["met"] + 0.5 * status_counts["partially_met"]) / known
                           if known else None)
        priority_score = (len(leaf_cases) * (1 - completion_rate)
                          if completion_rate is not None and known >= 3 else None)
        if priority_score is not None:
            priority_rows.append({
                "capability_id": leaf["capability_id"],
                "capability_path": " > ".join(leaf.get("path") or [leaf["name"]]),
                "capability_name": leaf["name"], "case_count": len(leaf_cases),
                "known_case_count": known, "met_count": status_counts["met"],
                "partially_met_count": status_counts["partially_met"],
                "unmet_count": status_counts["unmet"], "unclear_count": status_counts["unclear"],
                "completion_rate": round(completion_rate, 4),
                "priority_score": round(priority_score, 4),
            })
    priority_rows.sort(key=lambda row: (-row["priority_score"], -row["case_count"], row["capability_name"]))
    priority_fields = ["rank", "capability_id", "capability_path", "capability_name", "case_count",
                       "known_case_count", "met_count", "partially_met_count", "unmet_count",
                       "unclear_count", "completion_rate", "priority_score"]
    with priority_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=priority_fields)
        writer.writeheader()
        for rank, row in enumerate(priority_rows, 1):
            writer.writerow({"rank": rank, **row})

    lines.extend(["", "## 4. 需求分布与重点未满足需求", "",
                  "### 4.1 一级能力需求分布", "",
                  "![科研 ML/LLM Coding 一级能力需求分布](demand_distribution.png)", "",
                  "柱长表示归入该一级能力域的需求案例数。一个需求实例只归入一个叶子能力，因此一级能力数量之和与纳入案例数一致。",
                  "", "### 4.2 高频且完成度较低的需求", "",
                  "完成度按 `(已满足 + 0.5 × 部分满足) / 可判断案例数` 计算，其中可判断案例不包含 `unclear`。",
                  "重点分数为 `需求案例数 × (1 - 完成度)`；仅统计至少有 3 条可判断案例的叶子能力。", "",
                  "| 排名 | 叶子能力 | 需求量 | 已满足 | 部分满足 | 未满足 | 完成度 | 重点分数 |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for rank, row in enumerate(priority_rows[:20], 1):
        lines.append("| " + " | ".join([
            str(rank), _cell(row["capability_path"]), str(row["case_count"]), str(row["met_count"]),
            str(row["partially_met_count"]), str(row["unmet_count"]),
            f"{row['completion_rate']:.1%}", f"{row['priority_score']:.2f}",
        ]) + " |")
    if not priority_rows:
        lines.append("| — | 当前没有满足最小样本量条件的叶子能力 | — | — | — | — | — | — |")
    lines.extend(["", f"完整排名数据见 `{priority_path.name}`。"])
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"status": "success", "output_path": str(output.resolve()), "node_count": len(nodes),
            "leaf_count": leaf_sections, "case_count": len(cases), "evidence_row_count": evidence_rows,
            "demand_chart_path": str(chart_path.resolve()),
            "unmet_priority_path": str(priority_path.resolve()),
            "unmet_priority_count": len(priority_rows)}
