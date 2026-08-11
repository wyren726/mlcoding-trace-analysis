from __future__ import annotations

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCOPE_VERSION = "ml_llm_coding_scope_v3"
AUDIT_VERSION = "ml_llm_coding_scope_v3_strict_audit"


def _records(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _validate(value: dict[str, Any], expected: set[str]) -> list[dict[str, Any]]:
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")
    ids = [str(row.get("instance_id")) for row in decisions if isinstance(row, dict)]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Every input instance_id must occur exactly once")
    for row in decisions:
        if row.get("scope") not in {"included", "excluded", "uncertain"}:
            raise ValueError(f"Invalid scope: {row.get('scope')!r}")
        confidence = row.get("confidence")
        if isinstance(confidence, str):
            cleaned = confidence.strip().rstrip("%")
            try:
                confidence = float(cleaned)
                row["confidence"] = confidence
            except ValueError:
                confidence = 0.5
                row["confidence"] = confidence
        if isinstance(confidence, (int, float)) and 1 < confidence <= 100:
            row["confidence"] = confidence / 100
            confidence = row["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            row["confidence"] = 0.5
        if not str(row.get("reason") or "").strip():
            raise ValueError("Every decision requires a reason")
    return decisions


def filter_ml_llm_coding_scope(feature_paths: list[Path], output_dir: Path, client: Any,
                               provider: str, workers: int = 16, batch_size: int = 25,
                               audit_included: bool = False,
                               seed_decisions: Path | None = None) -> dict[str, Any]:
    """Classify extracted demand instances and retain only ML/LLM coding cases.

    The append-only decision file is the resume checkpoint. Filtered feature files retain
    the original demand-feature schema so the existing core builder can consume them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "ml_llm_coding_scope.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "scope_summary.json"
    filtered_dir = output_dir / "filtered_features"
    filtered_dir.mkdir(exist_ok=True)
    records = _records(feature_paths)
    instances: dict[str, dict[str, Any]] = {}
    for record in records:
        for instance in record.get("values", {}).get("instances") or []:
            instance_id = str(instance.get("instance_id") or "")
            if instance_id:
                instances.setdefault(instance_id, instance)
    existing: dict[str, dict[str, Any]] = {}
    if seed_decisions and seed_decisions.exists():
        with seed_decisions.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["instance_id"])] = row
    if decisions_path.exists():
        with decisions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["instance_id"])] = row
    if audit_included:
        pending = [(key, value) for key, value in instances.items()
                   if existing.get(key, {}).get("scope") == "included"
                   and existing.get(key, {}).get("scope_version") != AUDIT_VERSION]
    else:
        pending = [(key, value) for key, value in instances.items() if key not in existing]
    batches = [pending[index:index + max(1, batch_size)]
               for index in range(0, len(pending), max(1, batch_size))]
    system = (
        "你是科研ML/LLM Coding需求范围分类器。输入是用户需求实例，必须返回JSON对象decisions。"
        "每项包含instance_id、scope、reason、confidence；scope只能是included、excluded、uncertain。"
        "included指需求的实质工作属于机器学习/深度学习/大模型或ML/LLM Coding Agent工程，包括模型代码、"
        "数据管线、训练、微调、推理、评测、部署、实验复现，以及Coding Agent的Harness、Benchmark、模型API、"
        "评测脚本、环境依赖、任务提交、运行监控、故障诊断和结果分析。跨生物、材料、医学等领域仍可纳入，"
        "但实例自身必须明确指向ML/LLM计算实现或Agent开发评测对象，不能只凭所在项目上下文推断。纯论文写作润色、排版、"
        "表格整理、文档扩展、文献检索、普通文献综述、领域知识问答、研究叙事、"
        "通用办公、无ML/LLM内容的软件开发，以及只有概念讨论而没有计算实现需求的案例必须excluded。"
        "证据不足或是否需要ML/LLM Coding无法确定才使用uncertain。不要因为文本出现AI、模型、数据、研究等词就纳入。"
        "所有输入instance_id必须全部且仅出现一次。reason用一句简洁中文说明判据。"
    )
    if audit_included:
        system = (
            "你是科研ML/LLM Coding范围的严格复核员。输入均曾被初筛为included，但可能存在误纳。"
            "必须返回JSON对象decisions，每项包含instance_id、scope、reason、confidence。"
            "以下任一类均可included：ML/深度学习/LLM模型实现、训练、微调、推理、评测或部署；直接服务模型的数据管线；"
            "ML实验复现与性能诊断；ML/LLM Coding Agent的Harness、Benchmark、模型API、评测脚本、数据集环境、"
            "任务提交、运行监控、错误诊断和指标分析。后者即使表现为配置、依赖安装、日志分析或脚本修改，只要实例文本本身"
            "明确指出具体模型、Agent、Harness、Benchmark、推理API或评测流水线，也应included。不得仅根据所在项目背景补足关联。"
            "纯表格整理、格式化、文档写作扩展、文献检索综述，以及与ML/LLM或Coding Agent对象无明确关系的普通Python开发、"
            "文件操作、绘图、统计回归和通用运维应excluded。"
            "不得靠猜测上下文补足ML属性；边界证据不足使用uncertain。scope只能是included、excluded、uncertain。"
            "所有instance_id必须全部且仅出现一次，reason用一句简洁中文说明直接判据。"
        )

    def classify(batch: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        payload = {"instances": [{
            "instance_id": key,
            "need": value.get("need"),
            "object": value.get("object"),
            "expected_outcome": value.get("expected_outcome"),
            "candidate_capability": value.get("candidate_capability"),
            "constraints": value.get("constraints") or [],
        } for key, value in batch]}
        expected = {key for key, _ in batch}
        result, _ = client.complete_json(system, json.dumps(payload, ensure_ascii=False))
        return _validate(result, expected)

    completed_batches = failed_batches = 0
    if batches:
        mode = "a" if decisions_path.exists() else "w"
        with decisions_path.open(mode, encoding="utf-8") as out, errors_path.open("a", encoding="utf-8") as err:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(classify, batch): batch for batch in batches}
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        decisions = future.result()
                        for row in decisions:
                            enriched = {**row, "scope_version": AUDIT_VERSION if audit_included else SCOPE_VERSION,
                                        "generated_by": {"method": "llm", "provider": provider,
                                                         "model": client.model}}
                            out.write(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")) + "\n")
                            existing[str(row["instance_id"])] = enriched
                        out.flush()
                        os.fsync(out.fileno())
                        completed_batches += 1
                    except Exception as exc:
                        err.write(json.dumps({"instance_ids": [key for key, _ in batch], "error": str(exc)},
                                             ensure_ascii=False) + "\n")
                        err.flush()
                        failed_batches += 1
                    done = completed_batches + failed_batches
                    if done % 20 == 0 or done == len(batches):
                        print(json.dumps({"stage": "ml_llm_scope", "completed_batches": completed_batches,
                                          "failed_batches": failed_batches, "total_batches": len(batches),
                                          "classified_instances": len(existing)}, ensure_ascii=False),
                              file=sys.stderr, flush=True)

    # Compact append-only checkpoints into one latest decision per instance after a completed pass.
    with decisions_path.open("w", encoding="utf-8") as out:
        for instance_id in instances:
            if instance_id in existing:
                out.write(json.dumps(existing[instance_id], ensure_ascii=False, separators=(",", ":")) + "\n")

    counts = {"included": 0, "excluded": 0, "uncertain": 0, "unclassified": 0}
    for instance_id in instances:
        row = existing.get(instance_id)
        counts[row["scope"] if row else "unclassified"] += 1
    unaudited_included = sum(
        row.get("scope") == "included" and row.get("scope_version") != AUDIT_VERSION
        for row in existing.values()
    )
    # Re-read each source independently to retain batch boundaries and record metadata.
    filtered_paths = []
    for index, source in enumerate(feature_paths):
        target = filtered_dir / f"{index:02d}_{source.stem}.jsonl"
        filtered_paths.append(str(target.resolve()))
        with source.open(encoding="utf-8") as inp, target.open("w", encoding="utf-8") as out:
            for line in inp:
                if not line.strip():
                    continue
                record = json.loads(line)
                kept = []
                for instance in record.get("values", {}).get("instances") or []:
                    decision = existing.get(str(instance.get("instance_id") or ""))
                    if decision and decision.get("scope") == "included":
                        kept.append({**instance, "ml_llm_coding_scope": decision})
                if kept:
                    updated = dict(record)
                    updated["values"] = {**record.get("values", {}), "instances": kept}
                    out.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "scope_version": SCOPE_VERSION,
        "status": ("completed" if not counts["unclassified"]
                   and (not audit_included or not unaudited_included) else "incomplete"),
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "source_feature_files": [str(path.resolve()) for path in feature_paths],
        "total_instances": len(instances), **counts,
        "decision_path": str(decisions_path.resolve()), "filtered_feature_files": filtered_paths,
        "failed_batches": failed_batches,
        "strict_audit": audit_included,
        "unaudited_included": unaudited_included,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
