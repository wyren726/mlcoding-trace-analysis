#!/usr/bin/env python3
"""Select a reproducible, stratified pilot sample from remote delivery JSONL files."""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_HOST = "renwanying.renwanying-p+root.ailab-llmagent.ws@h.pjlab.org.cn"
DEFAULT_ROOT = "/mnt/shared-storage-user/songdemin/user/renwanying/data4sft/opus"
DEFAULT_QUOTAS = {
    "mlcoding_delivery_20260805_205233.jsonl": 14,
    "mlcoding_delivery_claude-4.6-opus-20260205_20260807_155721.jsonl": 2,
    "mlcoding_delivery_claude-4.8-opus-20260528_20260807_155721.jsonl": 10,
    "mlcoding_delivery_claude-opus-4.8_20260807_155721.jsonl": 24,
}
ERROR_MARKERS = ("error", "failed", "exception", "traceback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("quality_filter/pilot_50"))
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def length_bucket(message_count: int) -> str:
    if message_count <= 30:
        return "short"
    if message_count <= 100:
        return "medium"
    if message_count <= 200:
        return "long"
    return "extra_long"


def describe(sample: dict[str, Any], source_file: str, source_line: int) -> dict[str, Any]:
    messages = sample.get("messages") or []
    user_count = sum(m.get("role") == "user" for m in messages if isinstance(m, dict))
    tool_calls = sum(
        len(m.get("tool_calls") or [])
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    )
    tool_error_count = sum(
        any(marker in str(m.get("content") or "").lower() for marker in ERROR_MARKERS)
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )
    bucket = length_bucket(len(messages))
    return {
        "sample": sample,
        "sample_id": sample.get("id"),
        "source_session_id": (sample.get("metadata") or {}).get("source_session_id"),
        "source_file": source_file,
        "source_line": source_line,
        "message_count": len(messages),
        "user_turn_count": user_count,
        "tool_call_count": tool_calls,
        "tool_error_count": tool_error_count,
        "length_bucket": bucket,
        "multiturn": user_count > 1,
        "has_tool_error": tool_error_count > 0,
        "mixed_model": bool((sample.get("metadata") or {}).get("mixed_model_session")),
        "stratum": f"{bucket}|multi={user_count > 1}|error={tool_error_count > 0}",
    }


def load_remote(host: str, remote_root: str) -> dict[str, list[dict[str, Any]]]:
    remote_code = (
        "import glob,json,os; root=" + repr(remote_root) + "; "
        "fs=sorted(glob.glob(root+'/*.jsonl')); "
        "[(print(json.dumps({'source_file':os.path.basename(f),'source_line':i,'sample':json.loads(z)},"
        "ensure_ascii=False,separators=(',',':')))) "
        "for f in fs for i,z in enumerate(open(f,encoding='utf-8'),1)]"
    )
    command = ["ssh", "-CAXY", host, "python3", "-c", shlex.quote(remote_code)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert process.stdout is not None
    for output_line in process.stdout:
        wrapper = json.loads(output_line)
        grouped[wrapper["source_file"]].append(
            describe(wrapper["sample"], wrapper["source_file"], wrapper["source_line"])
        )
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"remote reader exited with status {return_code}")
    return grouped


def select_balanced(candidates: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        bins[candidate["stratum"]].append(candidate)
    for values in bins.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(bins)
    while len(selected) < quota and ordered_keys:
        next_keys: list[str] = []
        for key in ordered_keys:
            if bins[key] and len(selected) < quota:
                selected.append(bins[key].pop())
            if bins[key]:
                next_keys.append(key)
        ordered_keys = next_keys
    if len(selected) != quota:
        raise ValueError(f"requested {quota} samples, selected {len(selected)}")
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    grouped = load_remote(args.host, args.remote_root)
    selected: list[dict[str, Any]] = []
    for source_file, quota in DEFAULT_QUOTAS.items():
        candidates = grouped.get(source_file, [])
        if len(candidates) < quota:
            raise ValueError(f"{source_file}: only {len(candidates)} candidates for quota {quota}")
        selected.extend(select_balanced(candidates, quota, rng))
    selected.sort(key=lambda item: (item["source_file"], item["source_line"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot_input = [{"pilot_index": index, **item["sample"]} for index, item in enumerate(selected, 1)]
    manifest = [
        {key: value for key, value in item.items() if key != "sample"}
        | {"pilot_index": index, "seed": args.seed}
        for index, item in enumerate(selected, 1)
    ]
    write_jsonl(args.output_dir / "pilot_input.jsonl", pilot_input)
    write_jsonl(args.output_dir / "sample_manifest.jsonl", manifest)
    summary = {
        "seed": args.seed,
        "sample_count": len(selected),
        "source_quotas": DEFAULT_QUOTAS,
        "source_counts": {
            name: sum(item["source_file"] == name for item in selected) for name in DEFAULT_QUOTAS
        },
        "length_bucket_counts": {
            bucket: sum(item["length_bucket"] == bucket for item in selected)
            for bucket in ("short", "medium", "long", "extra_long")
        },
        "multiturn_count": sum(item["multiturn"] for item in selected),
        "tool_error_trace_count": sum(item["has_tool_error"] for item in selected),
        "mixed_model_count": sum(item["mixed_model"] for item in selected),
    }
    (args.output_dir / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
