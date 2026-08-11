from __future__ import annotations

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


GROUNDING_VERSION = "demand_user_grounding_v1"


def audit_demand_grounding(feature_paths: list[Path], output_dir: Path, client: Any,
                           provider: str, workers: int = 16, batch_size: int = 25) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "demand_grounding.jsonl"
    errors_path = output_dir / "errors.jsonl"
    filtered_dir = output_dir / "filtered_features"
    filtered_dir.mkdir(exist_ok=True)
    instances = {}
    for path in feature_paths:
        for line in path.open(encoding="utf-8"):
            if not line.strip(): continue
            record = json.loads(line)
            for instance in record.get("values", {}).get("instances") or []:
                instances.setdefault(str(instance["instance_id"]), instance)
    existing = {}
    if decisions_path.exists():
        for line in decisions_path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line); existing[str(row["instance_id"])] = row
    pending = [(key, value) for key, value in instances.items() if key not in existing]
    batches = [pending[i:i + max(1, batch_size)] for i in range(0, len(pending), max(1, batch_size))]
    system = (
        "你是用户需求语义落地审计员。输入包含模型归纳的need、expected_outcome及用户原文quotes。"
        "必须仅根据quotes判断，禁止利用任何Agent回复或外部上下文。返回JSON对象decisions，每项包含"
        "instance_id、grounding、reason；grounding只能是grounded、ungrounded、uncertain。"
        "grounded表示need及其关键具体对象/动作可由用户原文直接陈述或必然推出；允许忠实压缩和同义改写。"
        "若need加入用户未说出的具体算法、文件、实验、下一步、实现方式、指标、约束或目标，必须ungrounded。"
        "模糊请求不能支撑具体技术任务；例如用户只说担心创新性，不能归纳为验证某个具体模型假设。"
        "证据含义不足但无法确定时标uncertain。所有instance_id必须全部且仅出现一次。"
    )

    def classify(batch: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        payload = {"instances": [{"instance_id": key, "need": value.get("need"),
                                  "expected_outcome": value.get("expected_outcome"),
                                  "candidate_capability": value.get("candidate_capability"),
                                  "user_quotes": [row.get("quote") for row in value.get("requirement_evidence") or []]}
                                 for key, value in batch]}
        result, _ = client.complete_json(system, json.dumps(payload, ensure_ascii=False))
        rows = result.get("decisions")
        expected = {key for key, _ in batch}
        ids = [str(row.get("instance_id")) for row in rows or [] if isinstance(row, dict)]
        if not isinstance(rows, list) or len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("Every input instance_id must occur exactly once")
        if any(row.get("grounding") not in {"grounded", "ungrounded", "uncertain"} for row in rows):
            raise ValueError("Invalid grounding value")
        return rows

    completed = failed = 0
    if batches:
        with decisions_path.open("a", encoding="utf-8") as out, errors_path.open("a", encoding="utf-8") as err:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(classify, batch): batch for batch in batches}
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        for row in future.result():
                            enriched = {**row, "grounding_version": GROUNDING_VERSION,
                                        "generated_by": {"method": "llm", "provider": provider,
                                                         "model": client.model}}
                            existing[str(row["instance_id"])] = enriched
                            out.write(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")) + "\n")
                        out.flush(); os.fsync(out.fileno()); completed += 1
                    except Exception as exc:
                        err.write(json.dumps({"instance_ids": [key for key, _ in batch], "error": str(exc)},
                                             ensure_ascii=False) + "\n")
                        err.flush(); failed += 1
                    done = completed + failed
                    if done % 20 == 0 or done == len(batches):
                        print(json.dumps({"stage": "demand_grounding", "completed_batches": completed,
                                          "failed_batches": failed, "total_batches": len(batches),
                                          "decisions": len(existing)}, ensure_ascii=False), file=sys.stderr, flush=True)
    with decisions_path.open("w", encoding="utf-8") as out:
        for instance_id in instances:
            if instance_id in existing:
                out.write(json.dumps(existing[instance_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    filtered_paths = []
    for index, source in enumerate(feature_paths):
        target = filtered_dir / f"{index:02d}_{source.stem}.jsonl"; filtered_paths.append(str(target.resolve()))
        with source.open(encoding="utf-8") as inp, target.open("w", encoding="utf-8") as out:
            for line in inp:
                if not line.strip(): continue
                record = json.loads(line); kept=[]
                for instance in record.get("values", {}).get("instances") or []:
                    decision = existing.get(str(instance.get("instance_id")))
                    if decision and decision.get("grounding") == "grounded":
                        kept.append({**instance, "user_grounding": decision})
                if kept:
                    updated=dict(record); updated["values"]={**record.get("values",{}), "instances":kept}
                    out.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = {"grounded": 0, "ungrounded": 0, "uncertain": 0, "unclassified": 0}
    for key in instances:
        row=existing.get(key); counts[row["grounding"] if row else "unclassified"] += 1
    summary={"grounding_version":GROUNDING_VERSION,
             "status":"completed" if not counts["unclassified"] else "incomplete",
             "generated_at":dt.datetime.now().astimezone().isoformat(), "total_instances":len(instances), **counts,
             "decision_path":str(decisions_path.resolve()), "filtered_feature_files":filtered_paths,
             "failed_batches":failed}
    (output_dir/"grounding_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summary
