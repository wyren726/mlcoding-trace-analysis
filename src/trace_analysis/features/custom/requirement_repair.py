from __future__ import annotations

import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .semantic import (TASKS, add_instance_ids, compact_turn, normalize_model_value,
                       searchable_event_text, validate)


def repair_requirement_sources(batch_dir: Path, feature_file: Path, output_dir: Path,
                               client: Any, provider: str, workers: int = 16,
                               max_input_chars: int = 24000) -> dict[str, Any]:
    """Re-extract Turns whose demand evidence has no user_message source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "repaired_records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    final_path = output_dir / f"{batch_dir.name}.jsonl"
    records = []
    needed_event_ids = set()
    with feature_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            for instance in record.get("values", {}).get("instances") or []:
                needed_event_ids.update(str(row.get("event_id")) for row in instance.get("requirement_evidence") or [])
    event_types = {}
    turn_path = batch_dir / "unified_turns.jsonl"
    with turn_path.open(encoding="utf-8") as handle:
        for line in handle:
            turn = json.loads(line)
            for event in turn.get("events") or []:
                event_id = str(event.get("event_id"))
                if event_id in needed_event_ids:
                    event_types[event_id] = str(event.get("type"))
    contaminated_turn_ids = set()
    for record in records:
        for instance in record.get("values", {}).get("instances") or []:
            evidence = instance.get("requirement_evidence") or []
            if (not evidence
                    or any(event_types.get(str(row.get("event_id"))) != "user_message" for row in evidence)):
                contaminated_turn_ids.add(str(record.get("turn_id")))
                break
    turns = {}
    with turn_path.open(encoding="utf-8") as handle:
        for line in handle:
            turn = json.loads(line)
            turn_id = str(turn.get("turn_id"))
            if turn_id in contaminated_turn_ids:
                turns[turn_id] = turn
    repaired = {}
    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    repaired[str(row["turn_id"])] = row
    pending = [turns[key] for key in sorted(contaminated_turn_ids) if key not in repaired and key in turns]
    task = TASKS["demand_capability_instances"]
    system = (
        "你是Coding Agent用户需求证据修复器。必须返回一个JSON对象，不要输出Markdown。"
        f"任务：{task['instruction']} 输出结构：{json.dumps(task['schema'], ensure_ascii=False)} "
        "本次必须重新抽取当前Turn的全部真实用户需求。requirement_evidence.event_id只能引用user_message；"
        "quote必须从该用户消息逐字连续复制。Agent自己提出的计划、建议、下一步和扩展任务不是用户需求。"
        "若没有可由用户原文支持的需求，返回空instances。"
    )

    def process(turn: dict[str, Any]) -> dict[str, Any]:
        values, api_meta = client.complete_json(system, compact_turn(turn, max_input_chars))
        values = normalize_model_value("demand_capability_instances", values)
        texts = {str(e.get("event_id")): searchable_event_text(e.get("data")) for e in turn.get("events") or []}
        types = {str(e.get("event_id")): str(e.get("type")) for e in turn.get("events") or []}
        valid_instances = []
        for instance in values.get("instances") or []:
            for field in ("requirement_evidence", "fulfillment_evidence"):
                rows = instance.get(field) if isinstance(instance.get(field), list) else []
                instance[field] = [row for row in rows if isinstance(row, dict)
                                   and str(row.get("event_id")) in texts
                                   and isinstance(row.get("quote"), str) and row["quote"].strip()
                                   and row["quote"].strip() in texts[str(row.get("event_id"))]
                                   and (field != "requirement_evidence"
                                        or types.get(str(row.get("event_id"))) == "user_message")]
            if not instance["requirement_evidence"]:
                continue
            if instance.get("fulfillment") != "unclear" and not instance["fulfillment_evidence"]:
                instance["fulfillment"] = "unclear"
                instance["agent_gap"] = "精确证据不足，无法判断本次是否满足"
            valid_instances.append(instance)
        values["instances"] = valid_instances
        validate("demand_capability_instances", values, set(texts), texts)
        values = add_instance_ids(str(turn.get("turn_id")), values)
        return {"trace_id": turn.get("trace_id"), "session_id": turn.get("session_id"),
                "turn_id": turn.get("turn_id"), "feature_set": "demand_capability_instances",
                "feature_version": "v4", "feature_run_id": "requirement_source_repair_v1",
                "generated_by": {"method": "llm", "provider": provider, "model": client.model,
                                 "prompt_version": "v4-requirement-source-repair-v1"},
                "values": values, "api_meta": api_meta}

    completed = failed = 0
    if pending:
        with checkpoint.open("a", encoding="utf-8") as out, errors_path.open("a", encoding="utf-8") as err:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(process, turn): turn for turn in pending}
                for future in as_completed(futures):
                    turn = futures[future]
                    try:
                        row = future.result()
                        repaired[str(row["turn_id"])] = row
                        out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        out.flush(); os.fsync(out.fileno()); completed += 1
                    except Exception as exc:
                        err.write(json.dumps({"turn_id": turn.get("turn_id"), "error": str(exc)},
                                             ensure_ascii=False) + "\n")
                        err.flush(); failed += 1
                    done = completed + failed
                    if done % 25 == 0 or done == len(pending):
                        print(json.dumps({"stage": "requirement_source_repair", "completed": completed,
                                          "failed": failed, "total": len(pending)}, ensure_ascii=False),
                              file=sys.stderr, flush=True)
    with final_path.open("w", encoding="utf-8") as out:
        for record in records:
            turn_id = str(record.get("turn_id"))
            if turn_id in contaminated_turn_ids:
                if turn_id not in repaired:
                    continue
                record = repaired[turn_id]
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"status": "completed" if not failed and len(repaired) == len(contaminated_turn_ids) else "incomplete",
               "batch_id": batch_dir.name, "source_record_count": len(records),
               "contaminated_turn_count": len(contaminated_turn_ids), "repaired_turn_count": len(repaired),
               "pending_turn_count": len(contaminated_turn_ids) - len(repaired), "failed_count": failed,
               "output_path": str(final_path.resolve()),
               "generated_at": dt.datetime.now().astimezone().isoformat()}
    (output_dir / "repair_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    return summary
