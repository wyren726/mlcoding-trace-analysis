#!/usr/bin/env python3
"""Build a human-readable Markdown report for Agent-verified Trace candidates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUS_LABELS = {
    "confirmed": "确认痛点",
    "excluded": "排除候选",
    "review": "需人工复核",
}
ATTRIBUTION_LABELS = {
    "reasoning_or_response": "推理与响应",
    "execution_or_verification": "执行与验证",
    "tool_use": "工具使用",
    "context_handling": "上下文处理",
    "permission": "权限",
    "external": "外部限制",
    "user_input": "用户输入或需求变化",
    "unknown": "未知",
}
OUTCOME_LABELS = {
    "completed": "完成",
    "partially_completed": "部分完成",
    "failed": "失败",
    "unknown": "未知",
}
JUDGMENT_LABELS = {
    "clear": "清晰",
    "partly_clear": "部分清晰",
    "unclear": "不清晰",
    "high": "高",
    "medium": "中",
    "low": "低",
    "uncertain": "不确定",
    "met": "满足",
    "partially_met": "部分满足",
    "unmet": "未满足",
    "unknown": "未知",
    "initial_query": "初始要求",
    "clarified_later": "后续澄清",
    "new_later": "后续新增",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not an object: {path}")
            records.append(value)
    return records


def md_cell(value: Any) -> str:
    text = str(value if value is not None else "—").replace("\n", "<br>")
    return text.replace("|", "\\|")


def shorten(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text or "—"
    return text[: limit - 1].rstrip() + "…"


def quote_block(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return ["> —"]
    lines = []
    for line in text.splitlines():
        lines.append("> " + line if line else ">")
    return lines


def episode_status(episode: dict[str, Any]) -> str:
    assessment = episode.get("agent_assessment")
    related = assessment.get("agent_related") if isinstance(assessment, dict) else None
    if related is True:
        return "confirmed"
    if related is False:
        return "excluded"
    return "review"


def trace_status(episodes: list[dict[str, Any]]) -> str:
    statuses = {episode_status(episode) for episode in episodes}
    if "confirmed" in statuses:
        return "confirmed"
    if "review" in statuses:
        return "review"
    return "excluded"


def real_models(values: set[str]) -> list[str]:
    models = sorted(value for value in values if value and value != "<synthetic>")
    return models or ["unknown"]


def model_display(models: list[str]) -> str:
    labels = {
        "anthropic/claude-4.6-opus-20260205": "Claude 4.6 Opus",
        "deepseek-v4-pro": "DeepSeek V4 Pro",
        "unknown": "未知模型",
    }
    return " + ".join(labels.get(model, model) for model in models)


def episode_title(episode: dict[str, Any], status: str) -> str:
    assessment = episode.get("agent_assessment") or {}
    gaps = assessment.get("capability_gaps") or []
    if status == "confirmed" and gaps:
        return "、".join(str(gap.get("capability") or "能力缺口") for gap in gaps[:2])
    if status == "review":
        return shorten(assessment.get("agent_failure") or "现有证据不足", 54)
    signals = episode.get("candidate_signals") or {}
    kinds = [str(item.get("type")) for item in signals.get("signals") or []]
    return "候选信号未构成 Agent 痛点" + (f"（{', '.join(kinds)}）" if kinds else "")


def evidence_rows(episode: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, item: Any) -> None:
        if not isinstance(item, dict):
            return
        event_id = str(item.get("event_id") or "")
        quote = str(item.get("quote") or "").strip()
        if not event_id or not quote or (event_id, quote) in seen:
            return
        seen.add((event_id, quote))
        rows.append({
            "kind": kind,
            "turn_id": str(item.get("turn_id") or ""),
            "event_id": event_id,
            "quote": quote,
        })

    for signal in (episode.get("candidate_signals") or {}).get("signals") or []:
        add("用户候选信号", signal)
    assessment = episode.get("agent_assessment") or {}
    for item in (assessment.get("goal_assessment") or {}).get("evidence") or []:
        add("目标确认", item)
    for gap in assessment.get("capability_gaps") or []:
        for item in gap.get("evidence") or []:
            add("能力缺口", item)
    return rows


def benchmark_note(episode: dict[str, Any], status: str) -> str:
    if status == "excluded":
        return "不建议作为能力缺陷 Benchmark；当前证据更符合正常等待、任务推进、外部限制或用户新增要求。"
    if status == "review":
        return "暂不进入 Benchmark 候选池；应先人工复核失败是否真实发生、是否由 Agent 导致。"
    clarity = (episode.get("initial_query_clarity") or {}).get("judgment")
    value = (episode.get("analysis_value") or {}).get("judgment")
    if clarity == "clear" and value in {"high", "medium"}:
        return "建议进入 Benchmark 候选池；后续需要结合 Workspace 固化输入、环境、验收标准和参考结果。"
    return "可以保留为 Benchmark 候选，但需要先补足任务输入、验收标准或初始 Query 的明确性。"


def build_report(
    analysis_path: Path,
    user_turn_path: Path,
    unified_turn_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    analysis_records = [
        record for record in read_jsonl(analysis_path)
        if record.get("route", {}).get("judgment") == "analyze"
    ]
    user_records = {
        str(record.get("trace_id")): record for record in read_jsonl(user_turn_path)
    }
    analysis_by_id = {str(record["trace_id"]): record for record in analysis_records}
    metadata: dict[str, dict[str, set[str]]] = {
        trace_id: {"models": set(), "harnesses": set()}
        for trace_id in analysis_by_id
    }
    with unified_turn_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            turn = json.loads(line)
            trace_id = str(turn.get("trace_id") or "")
            if trace_id not in metadata:
                continue
            metadata[trace_id]["models"].update(
                str(value) for value in turn.get("agent_models") or [] if value
            )
            harness = turn.get("harness")
            if harness:
                metadata[trace_id]["harnesses"].add(str(harness))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    judge_models = {str(manifest.get("model") or "unknown")}
    user_prompt_versions = manifest.get("prompt_version_history") or [
        manifest.get("prompt_version")
    ]
    agent_prompt_versions = manifest.get("attribution_prompt_version_history") or [
        manifest.get("attribution_prompt_version")
    ]
    traces = []
    for record in analysis_records:
        trace_id = str(record["trace_id"])
        episodes = [
            episode for episode in record.get("task_episodes") or []
            if isinstance(episode.get("agent_assessment"), dict)
        ]
        if not episodes:
            continue
        models = real_models(metadata[trace_id]["models"])
        traces.append({
            "record": record,
            "episodes": episodes,
            "status": trace_status(episodes),
            "models": models,
            "model_label": model_display(models),
            "harnesses": sorted(metadata[trace_id]["harnesses"]) or ["unknown"],
            "user": user_records.get(trace_id, {}),
            "judge_model": str(
                (record.get("analysis_provenance") or {}).get(
                    "agent_verification_model"
                ) or manifest.get("model") or "unknown"
            ),
        })
        judge_models.add(traces[-1]["judge_model"])
    order = {"confirmed": 0, "review": 1, "excluded": 2}
    traces.sort(key=lambda item: (order[item["status"]], str(item["record"]["trace_id"])))

    episode_counts = Counter(
        episode_status(episode) for trace in traces for episode in trace["episodes"]
    )
    trace_counts = Counter(trace["status"] for trace in traces)
    model_trace_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        for model in trace["models"]:
            model_trace_counts[model][trace["status"]] += 1

    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Agent 候选痛点验证分析报告",
        "",
        "> 本报告面向不了解 Pipeline 细节的读者，汇总用户侧候选筛选后结合 Agent/Tool Turn 完成的验证结果。",
        "",
        "## 1. 数据范围与阅读说明",
        "",
        f"- Batch：`{manifest.get('batch_id')}`",
        f"- Analysis Run：`{manifest.get('analysis_run_id')}`",
        f"- 报告生成时间：`{generated_at}`",
        f"- 当前 Run 状态：`{manifest.get('status')}`",
        f"- 全批次 Trace：{manifest.get('total_trace_count')} 条",
        f"- 已完成 Trace：{manifest.get('completed_trace_count')} 条",
        f"- 尚未完成 Trace：{max(0, int(manifest.get('total_trace_count') or 0) - int(manifest.get('completed_trace_count') or 0))} 条",
        f"- 进入 Agent 验证：{len(traces)} 条 Trace、{sum(len(trace['episodes']) for trace in traces)} 个 Episode",
        f"- 被分析 Agent：Trace 原始运行中的模型；Judge 模型：`{', '.join(sorted(judge_models))}`",
        f"- 用户筛选 Prompt：`{', '.join(str(value) for value in user_prompt_versions if value)}`",
        f"- Agent 验证 Prompt：`{', '.join(str(value) for value in agent_prompt_versions if value)}`",
        "",
        (
            "> 注意：当前 Run 已完成全批次分析，报告分母包含全部 Trace。"
            if manifest.get("status") == "completed"
            else "> 注意：这是阶段性快照。尚未完成的 Trace 不进入分母；补跑结束后应重新生成报告。"
        ),
        "",
        "判定口径：",
        "",
        "- **确认痛点**：`agent_related=true`，存在可观察的 Agent 失败、归因理由和证据。",
        "- **排除候选**：`agent_related=false`，候选信号可由正常等待、任务变化、外部限制等解释。",
        "- **需人工复核**：`agent_related=null`，观察到潜在失败但证据不足以安全归因。",
        "",
        "## 2. 总览",
        "",
        "### 2.1 模型分布（按 Trace）",
        "",
        "| 被分析 Agent 模型 | Trace 数 | 占比 | 确认痛点 | 排除候选 | 需人工复核 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, counts in sorted(
        model_trace_counts.items(), key=lambda item: (-sum(item[1].values()), item[0])
    ):
        total = sum(counts.values())
        lines.append(
            f"| {md_cell(model_display([model]))} | {total} | {total / len(traces):.1%} | "
            f"{counts['confirmed']} | {counts['excluded']} | {counts['review']} |"
        )
    lines.extend([
        f"| **合计** | **{len(traces)}** | **100%** | **{trace_counts['confirmed']}** | **{trace_counts['excluded']}** | **{trace_counts['review']}** |",
        "",
        "本批候选全部由 `claude_code` Harness 产生。当前数据不是相同任务上的受控模型实验，因此只能展示样本分布，不能据此比较模型能力高低。",
        "",
        "### 2.2 Episode 判定分布",
        "",
        "| 判定 | Episode 数 |",
        "|---|---:|",
        f"| 确认痛点 | {episode_counts['confirmed']} |",
        f"| 排除候选 | {episode_counts['excluded']} |",
        f"| 需人工复核 | {episode_counts['review']} |",
        "",
        "### 2.3 已确认痛点概览",
        "",
        "| Case | 被分析 Agent 模型 | 痛点概述 | 归因 | 结果 | Trace / Episode |",
        "|---|---|---|---|---|---|",
    ])
    case_numbers = {str(trace["record"]["trace_id"]): index for index, trace in enumerate(traces, 1)}
    for trace in traces:
        case_number = case_numbers[str(trace["record"]["trace_id"])]
        for episode in trace["episodes"]:
            if episode_status(episode) != "confirmed":
                continue
            assessment = episode["agent_assessment"]
            lines.append(
                f"| {case_number} | {md_cell(trace['model_label'])} | "
                f"{md_cell(shorten(assessment.get('agent_failure'), 105))} | "
                f"{md_cell(ATTRIBUTION_LABELS.get(assessment.get('attribution'), assessment.get('attribution')))} | "
                f"{md_cell(OUTCOME_LABELS.get(episode.get('outcome'), episode.get('outcome')))} | "
                f"`{trace['record']['trace_id']}`<br>`{episode.get('episode_id')}` |"
            )
    lines.extend([
        "",
        "### 2.4 排除与待复核候选概览",
        "",
        "| Case | 被分析 Agent 模型 | 判定 | 候选或潜在问题 | Trace / Episode |",
        "|---|---|---|---|---|",
    ])
    for trace in traces:
        case_number = case_numbers[str(trace["record"]["trace_id"])]
        for episode in trace["episodes"]:
            status = episode_status(episode)
            if status == "confirmed":
                continue
            assessment = episode["agent_assessment"]
            summary = assessment.get("agent_failure") or (episode.get("candidate_signals") or {}).get("reason")
            lines.append(
                f"| {case_number} | {md_cell(trace['model_label'])} | {STATUS_LABELS[status]} | "
                f"{md_cell(shorten(summary, 110))} | `{trace['record']['trace_id']}`<br>`{episode.get('episode_id')}` |"
            )

    lines.extend(["", "## 3. 逐条案例分析", ""])
    for trace in traces:
        record = trace["record"]
        trace_id = str(record["trace_id"])
        case_number = case_numbers[trace_id]
        user = trace["user"]
        lines.extend([
            f"## Case {case_number:02d}｜{STATUS_LABELS[trace['status']]}｜{trace['model_label']}",
            "",
            "### Trace 基本信息",
            "",
            "| 字段 | 内容 |",
            "|---|---|",
            f"| Trace ID | `{trace_id}` |",
            f"| Session ID | `{user.get('session_id') or 'unknown'}` |",
            f"| 被分析 Agent 模型 | {md_cell(', '.join(trace['models']))} |",
            f"| Harness | {md_cell(', '.join(trace['harnesses']))} |",
            f"| Judge 模型 | `{trace['judge_model']}` |",
            f"| Analysis 状态 | `{record.get('analysis_status')}` |",
            f"| User Turn 数 | {user.get('user_turn_count', 'unknown')} |",
            f"| 验证 Episode 数 | {len(trace['episodes'])} |",
            "",
        ])
        source_files = user.get("source_files") or []
        if source_files:
            lines.append("原始 Session：")
            lines.append("")
            for source_file in source_files:
                lines.append(f"- [`session.jsonl`](<{source_file}>)")
            lines.append("")

        for episode_index, episode in enumerate(trace["episodes"], 1):
            status = episode_status(episode)
            assessment = episode["agent_assessment"]
            preliminary = episode.get("preliminary_goal") or {}
            lines.extend([
                f"### Episode {case_number}.{episode_index}｜{STATUS_LABELS[status]}｜{episode_title(episode, status)}",
                "",
                "#### 结论摘要",
                "",
            ])
            if status == "confirmed":
                conclusion = assessment.get("agent_failure") or "存在可归因于 Agent 的失败。"
            elif status == "excluded":
                conclusion = assessment.get("agent_related_reason") or "现有候选信号不构成 Agent 能力缺陷。"
            else:
                conclusion = assessment.get("agent_failure") or "现有证据不足以确认或排除 Agent 痛点。"
            lines.extend(quote_block(conclusion))
            lines.extend([
                "",
                "| 项目 | 结论 |",
                "|---|---|",
                f"| Episode ID | `{episode.get('episode_id')}` |",
                f"| Query Turn | `{episode.get('query_turn_id')}` |",
                f"| 痛点判定 | **{STATUS_LABELS[status]}** |",
                f"| 初始 Query 清晰度 | {JUDGMENT_LABELS.get((episode.get('initial_query_clarity') or {}).get('judgment'), (episode.get('initial_query_clarity') or {}).get('judgment'))} |",
                f"| 分析价值 | {JUDGMENT_LABELS.get((episode.get('analysis_value') or {}).get('judgment'), (episode.get('analysis_value') or {}).get('judgment'))} |",
                f"| 任务结果 | {OUTCOME_LABELS.get(episode.get('outcome'), episode.get('outcome'))} |",
                f"| 归因类型 | {ATTRIBUTION_LABELS.get(assessment.get('attribution'), assessment.get('attribution'))} |",
                "",
                "#### 用户目标",
                "",
            ])
            lines.extend(quote_block(preliminary.get("goal") or "无法完整还原"))
            lines.extend([
                "",
                f"初始 Query 判断：{(episode.get('initial_query_clarity') or {}).get('reason') or '—'}",
                "",
                "#### 用户要求及完成情况",
                "",
                "| 要求 | 来源 | 状态 | 证据定位 |",
                "|---|---|---|---|",
            ])
            requirements = preliminary.get("requirements") or []
            if requirements:
                for requirement in requirements:
                    refs = ", ".join(
                        f"`{item.get('turn_id')}` / `{item.get('event_id')}`"
                        for item in requirement.get("evidence") or []
                    ) or "—"
                    lines.append(
                        f"| {md_cell(requirement.get('text'))} | "
                        f"{md_cell(JUDGMENT_LABELS.get(requirement.get('origin'), requirement.get('origin')))} | "
                        f"{md_cell(JUDGMENT_LABELS.get(requirement.get('status'), requirement.get('status')))} | {refs} |"
                    )
            else:
                lines.append("| — | — | — | — |")

            signals = (episode.get("candidate_signals") or {}).get("signals") or []
            lines.extend([
                "",
                "#### 进入 Agent 验证的原因",
                "",
                f"{(episode.get('candidate_signals') or {}).get('reason') or '—'}",
                "",
                "| 信号 | 用户原话 | Turn / Event |",
                "|---|---|---|",
            ])
            if signals:
                for signal in signals:
                    lines.append(
                        f"| `{signal.get('type')}` | {md_cell(signal.get('quote'))} | "
                        f"`{signal.get('turn_id')}` / `{signal.get('event_id')}` |"
                    )
            else:
                lines.append("| — | — | — |")

            lines.extend([
                "",
                "#### Agent 行为验证",
                "",
                "**观察到的失败**",
                "",
            ])
            lines.extend(quote_block(assessment.get("agent_failure") or "未观察到可确认的 Agent 失败。"))
            lines.extend([
                "",
                "**归因判断**",
                "",
            ])
            lines.extend(quote_block(assessment.get("agent_related_reason") or "现有证据不足。"))
            lines.extend(["", "#### 能力缺口", ""])
            gaps = assessment.get("capability_gaps") or []
            if gaps:
                for gap_index, gap in enumerate(gaps, 1):
                    lines.extend([
                        f"**{gap_index}. {gap.get('capability') or '未命名能力缺口'}**",
                        "",
                        str(gap.get("reason") or "—"),
                        "",
                    ])
                    for evidence in gap.get("evidence") or []:
                        lines.append(
                            f"- `{evidence.get('turn_id')}` / `{evidence.get('event_id')}`："
                            f"{evidence.get('quote')}"
                        )
                    lines.append("")
            else:
                lines.extend(["未确认能力缺口。", ""])

            rows = evidence_rows(episode)
            lines.extend(["#### 关键证据链", ""])
            if rows:
                for row in rows:
                    lines.append(
                        f"- **{row['kind']}**｜`{row['turn_id']}` / `{row['event_id']}`"
                    )
                    lines.extend(quote_block(row["quote"]))
                    lines.append("")
            else:
                lines.extend(["当前没有通过校验的关键证据。", ""])

            lines.extend([
                "#### Benchmark 构造备注",
                "",
                benchmark_note(episode, status),
                "",
            ])

        warnings = record.get("validation_warnings") or []
        lines.extend(["### 数据质量提示", ""])
        if warnings:
            for warning in warnings:
                lines.append(f"- `{warning}`")
        else:
            lines.append("- 未记录结构或证据校验警告。")
        lines.extend(["", "### 完整 User Turn（折叠）", "", "<details>", "<summary>展开该 Trace 的全部真实 User Turn</summary>", ""])
        for turn in user.get("user_turns") or []:
            lines.append(
                f"**Turn {turn.get('turn_index')}**｜`{turn.get('turn_id')}` / `{turn.get('event_id')}`"
            )
            lines.append("")
            lines.extend(quote_block(turn.get("content")))
            lines.append("")
        lines.extend(["</details>", "", "---", ""])

    scope_limit = (
        "- 当前报告覆盖已完成全批次中进入 Agent 验证的候选；它仍不是全量失败率统计。"
        if manifest.get("status") == "completed"
        else "- 当前报告只覆盖已成功完成 Agent 验证的候选，不能代表完整批次的最终分布。"
    )
    lines.extend([
        "## 4. 数据与结论局限",
        "",
        scope_limit,
        "- 候选样本经过用户行为信号筛选，不适合直接计算模型总体失败率。",
        "- 模型分布不是同任务、同环境下的受控比较，不能据此得出 Claude 与 DeepSeek 的相对能力结论。",
        "- `needs_review` 不等于没有痛点；它表示当前 Trace 证据不足以安全归因。",
        "- Benchmark 构造备注是报告层的初步建议，后续仍需结合 Workspace、交付物和可执行验收标准复核。",
        "",
        "## 5. 机器可读数据",
        "",
        f"- Agent 验证结果：`{analysis_path}`",
        f"- 对应 User Turn：`{user_turn_path}`",
        f"- 预处理 Turn：`{unified_turn_path}`",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(lines)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output_path)
    except PermissionError:
        # Some shared NFS directories permit updating an existing file but not
        # creating a sibling temporary file.  The report is reproducible, so a
        # pre-created destination remains a safe fallback in that environment.
        if not output_path.is_file():
            raise
        output_path.write_text(rendered, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "trace_count": len(traces),
        "episode_count": sum(len(trace["episodes"]) for trace in traces),
        "confirmed_episode_count": episode_counts["confirmed"],
        "excluded_episode_count": episode_counts["excluded"],
        "review_episode_count": episode_counts["review"],
        "generated_at": generated_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--user-turns", required=True, type=Path)
    parser.add_argument("--unified-turns", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_report(
        args.analysis,
        args.user_turns,
        args.unified_turns,
        args.manifest,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
