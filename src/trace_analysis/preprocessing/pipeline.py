from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .adapters import choose_adapter
from .adapters.common import redact_text
from ..registries import registry_root_for_output, write_immutable_record


REGISTRY_FIELDS = [
    "batch_id", "source", "processed_at", "adapter", "raw_size_bytes",
    "output_size_bytes", "session_count", "event_count", "model_distribution",
    "harness_distribution", "output_path", "status",
]

PREPROCESSOR_VERSION = "v1.6.2-collector-v020"


def _append_registry(output_root: Path, row: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    registry = output_root / "data_registry.csv"
    write_header = not registry.exists()
    with registry.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in REGISTRY_FIELDS})


def _input_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path).encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def _read_ingestion_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_ingestion_index(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _native_event_type(event: dict[str, Any]) -> str:
    raw = event.get("raw") if isinstance(event.get("raw"), dict) else {}
    wrapper = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    inner = wrapper.get("payload") if isinstance(wrapper.get("payload"), dict) else {}
    return str(inner.get("type") or wrapper.get("type") or "")


def _turn_status(events: list[dict[str, Any]]) -> str:
    native_types = {_native_event_type(event) for event in events}
    if "turn_aborted" in native_types:
        return "aborted"
    if any(event.get("type") == "error" for event in events):
        return "aborted"
    if "task_complete" in native_types:
        return "completed"
    calls = set()
    results = set()
    for event in events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event.get("type") == "tool_call" and (call_id := data.get("call_id") or data.get("id")):
            calls.add(str(call_id))
        elif event.get("type") == "tool_result" and (call_id := data.get("call_id") or data.get("tool_use_id")):
            results.add(str(call_id))
    substantive = [event for event in events if event.get("type") not in {
        "system_event", "model_call", "model_result", "token_usage"
    }]
    if substantive and substantive[-1].get("type") == "assistant_message" and not (calls - results):
        return "completed"
    return "incomplete"


def _trace_id(session_id: str, events: list[dict[str, Any]]) -> str:
    namespaces: set[str] = set()
    native_trace_ids: set[str] = set()
    project_ids: set[str] = set()
    for event in events:
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        if source.get("adapter"):
            namespaces.add(str(source["adapter"]))
        raw = event.get("raw") if isinstance(event.get("raw"), dict) else {}
        if raw.get("trace_id"):
            native_trace_ids.add(str(raw["trace_id"]))
        project = raw.get("project_hmac")
        if project:
            project_ids.add(json.dumps(project, ensure_ascii=False, sort_keys=True))
    identity = ("native_trace", sorted(native_trace_ids)) if native_trace_ids else (
        "native_session", sorted(namespaces), sorted(project_ids), session_id)
    return f"trace_{hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]}"


def source_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file() and item.suffix in {".json", ".jsonl"}))
        else:
            raise FileNotFoundError(value)
    unique = list(dict.fromkeys(files))
    if not unique:
        raise ValueError("No JSON/JSONL input files found")
    return unique


def first_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"First record is not an object: {path}")
                return value
    raise ValueError(f"Empty input: {path}")


def get_model(event: dict[str, Any]) -> str | None:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    raw = event.get("raw") if isinstance(event.get("raw"), dict) else {}
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    return data.get("model_name") or data.get("model") or payload.get("model")


def get_harness(event: dict[str, Any]) -> str | None:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    if data.get("harness") and data["harness"] != "unknown":
        return str(data["harness"])
    raw = event.get("raw") if isinstance(event.get("raw"), dict) else {}
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    adapter = str(source.get("adapter") or "")
    if "codex" in adapter:
        return "codex"
    if "claude" in adapter:
        return "claude_code"
    return None


def _run_preprocess_impl(inputs: list[str], output_root: Path, requested_adapter: str | None,
                         limit: int | None) -> dict[str, Any]:
    files = source_files(inputs)
    output_root.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().astimezone()
    input_fingerprint = _input_fingerprint(files)
    ingestion_key = hashlib.sha256(
        f"{PREPROCESSOR_VERSION}|{input_fingerprint}|adapter={requested_adapter or 'auto'}|limit={limit}".encode()
    ).hexdigest()
    index_path = output_root / ".ingestion_index.json"
    ingestion_index = _read_ingestion_index(index_path)
    prior = ingestion_index.get(ingestion_key)
    if isinstance(prior, dict) and Path(str(prior.get("output_path", ""))).is_dir():
        return {**prior, "deduplicated": True, "duplicate_of_batch_id": prior.get("batch_id")}
    signature = input_fingerprint[:8]
    batch_id = f"batch_{now.strftime('%Y%m%d_%H%M%S')}_{signature}"
    batch_dir = output_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    trace_path = batch_dir / "unified_traces.jsonl"
    turn_path = batch_dir / "unified_turns.jsonl"
    staging_path = batch_dir / "_staging.sqlite"
    rejected_path = batch_dir / "rejected_records.jsonl"
    database = sqlite3.connect(staging_path)
    # WAL commonly fails on NFS/shared-storage mounts with "disk I/O error".
    # DELETE mode is portable and sufficient for this single-process staging DB.
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute("CREATE TABLE events (session_id TEXT NOT NULL, sequence INTEGER, timestamp TEXT, payload TEXT NOT NULL)")
    database.execute("CREATE INDEX events_session_order ON events(session_id, sequence, timestamp)")
    adapters: Counter[str] = Counter()
    adapter_cache: dict[str, Any] = {}
    rejected = 0
    seen = 0
    staged = 0

    rejected_out = rejected_path.open("w", encoding="utf-8")
    for path in files:
        detected = choose_adapter(first_record(path), requested_adapter)
        adapter = adapter_cache.setdefault(detected.name, detected)
        adapters[adapter.name] += 1
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                if limit is not None and seen >= limit:
                    break
                seen += 1
                try:
                    record = json.loads(line)
                    converted = adapter.convert(record, batch_id=batch_id, source_file=str(path), record_index=index)
                    for event in converted if isinstance(converted, list) else [converted]:
                        database.execute(
                            "INSERT INTO events VALUES (?, ?, ?, ?)",
                            (event["session_id"], event.get("sequence"), event.get("timestamp"),
                             json.dumps(event, ensure_ascii=False, separators=(",", ":"))),
                        )
                        staged += 1
                        if staged % 1000 == 0:
                            database.commit()
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    rejected += 1
                    rejected_out.write(json.dumps({
                        "source_file": str(path), "record_index": index,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "raw_text": redact_text(line.rstrip("\n")[:10000]),
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
            if limit is not None and seen >= limit:
                break

    rejected_out.close()
    database.commit()
    model_call_counts: Counter[str] = Counter()
    turn_model_counts: Counter[str] = Counter()
    harness_counts: Counter[str] = Counter()
    event_count = 0
    with trace_path.open("w", encoding="utf-8") as trace_out, turn_path.open("w", encoding="utf-8") as turn_out:
        session_ids = [row[0] for row in database.execute("SELECT DISTINCT session_id FROM events ORDER BY session_id")]
        for session_id in session_ids:
            events = [json.loads(row[0]) for row in database.execute(
                "SELECT payload FROM events WHERE session_id = ? ORDER BY sequence, timestamp", (session_id,)
            )]
            active_turn: str | None = None
            for event in events:
                raw = event.get("raw") if isinstance(event.get("raw"), dict) else {}
                wrapper = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
                inner = wrapper.get("payload") if isinstance(wrapper.get("payload"), dict) else {}
                native_type = inner.get("type")
                if event.get("turn_id"):
                    active_turn = str(event["turn_id"])
                elif active_turn:
                    event["turn_id"] = active_turn
                if native_type in {"task_complete", "turn_aborted"}:
                    active_turn = None
            # Batch-independent: ingesting the same native Session again keeps the
            # same Trace ID, enabling cross-batch deduplication.
            trace_id = _trace_id(session_id, events)
            harnesses = {value for event in events if (value := get_harness(event))}
            harness = next(iter(harnesses)) if len(harnesses) == 1 else ("mixed" if harnesses else "unknown")
            session_models = {str(value) for event in events if (value := get_model(event))}
            harness_counts[harness] += 1
            turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
            unassigned: list[dict[str, Any]] = []
            for event in events:
                event["trace_id"] = trace_id
                event_count += 1
                model = get_model(event)
                if model and event.get("type") == "model_call":
                    model_call_counts[str(model)] += 1
                if event.get("turn_id"):
                    turns[str(event["turn_id"])].append(event)
                else:
                    unassigned.append(event)
            turn_records = []
            for ordinal, (turn_id, turn_events) in enumerate(turns.items(), 1):
                turn_models = {str(value) for event in turn_events if (value := get_model(event))}
                if not turn_models and len(session_models) == 1:
                    turn_models = set(session_models)
                if not any(event.get("type") == "model_call" for event in turn_events):
                    for model_name in turn_models:
                        turn_model_counts[model_name] += 1
                status = _turn_status(turn_events)
                if status == "incomplete" and ordinal < len(turns):
                    status = "interrupted"
                record = {"schema_version": "v1.0", "batch_id": batch_id, "trace_id": trace_id,
                          "session_id": session_id, "turn_id": turn_id, "turn_index": ordinal,
                          "status": status, "harness": harness,
                          "agent_models": sorted(turn_models), "events": turn_events}
                turn_records.append(record)
                turn_view = {**record, "events": [
                    {**{key: value for key, value in event.items() if key != "raw"},
                     "raw_ref": {"trace_id": trace_id, "event_id": event["event_id"]}}
                    for event in turn_events
                ]}
                turn_out.write(json.dumps(turn_view, ensure_ascii=False, separators=(",", ":")) + "\n")
            trace = {"schema_version": "v1.0", "batch_id": batch_id, "trace_id": trace_id,
                     "sessions": [{"session_id": session_id, "harness": harness, "turns": turn_records,
                                   "unassigned_events": unassigned}]}
            trace_out.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")

    output_size = trace_path.stat().st_size + turn_path.stat().st_size
    status = "partial" if rejected else "success"
    row = {
        "batch_id": batch_id,
        "source": json.dumps([str(Path(value).expanduser().resolve()) for value in inputs], ensure_ascii=False),
        "processed_at": now.isoformat(), "adapter": "+".join(sorted(adapters)),
        "raw_size_bytes": sum(path.stat().st_size for path in files), "output_size_bytes": output_size,
        "session_count": len(session_ids), "event_count": event_count,
        "model_distribution": json.dumps({
            "unit": "model_call" if model_call_counts else "turn_fallback",
            "counts": model_call_counts if model_call_counts else turn_model_counts,
        }, ensure_ascii=False),
        "harness_distribution": json.dumps({"unit": "session", "counts": harness_counts}, ensure_ascii=False),
        "output_path": str(batch_dir.resolve()), "status": status,
    }
    _append_registry(output_root, row)
    registry_record = write_immutable_record(
        "datasets", batch_id, row,
        artifacts=(trace_path, turn_path, rejected_path),
        root=registry_root_for_output(output_root),
    )
    database.close()
    for suffix_path in (staging_path, Path(str(staging_path) + "-wal"), Path(str(staging_path) + "-shm")):
        if suffix_path.exists():
            suffix_path.unlink()
    result = {**row, "preprocessor_version": PREPROCESSOR_VERSION,
              "input_fingerprint": input_fingerprint, "deduplicated": False,
              "rejected_record_count": rejected, "rejected_records_path": str(rejected_path.resolve())}
    result["registry_record_path"] = str(registry_record.resolve())
    ingestion_index[ingestion_key] = result
    _write_ingestion_index(index_path, ingestion_index)
    return result


def run_preprocess(inputs: list[str], output_root: Path, requested_adapter: str | None,
                   limit: int | None) -> dict[str, Any]:
    """Run preprocessing and register both successful and failed attempts."""
    try:
        return _run_preprocess_impl(inputs, output_root, requested_adapter, limit)
    except Exception as exc:
        now = dt.datetime.now().astimezone()
        output_root.mkdir(parents=True, exist_ok=True)
        failure_id = f"failed_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        diagnostics = output_root / f"{failure_id}.json"
        diagnostics.write_text(json.dumps({
            "status": "failed", "processed_at": now.isoformat(), "sources": inputs,
            "requested_adapter": requested_adapter or "auto", "limit": limit,
            "error_type": type(exc).__name__, "error": redact_text(str(exc)),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        resolved = [Path(value).expanduser() for value in inputs]
        _append_registry(output_root, {
            "batch_id": failure_id, "source": json.dumps(inputs, ensure_ascii=False),
            "processed_at": now.isoformat(), "adapter": requested_adapter or "auto",
            "raw_size_bytes": sum(path.stat().st_size for path in resolved if path.is_file()),
            "output_size_bytes": diagnostics.stat().st_size, "session_count": 0, "event_count": 0,
            "model_distribution": json.dumps({"unit": "none", "counts": {}}),
            "harness_distribution": json.dumps({"unit": "none", "counts": {}}),
            "output_path": str(diagnostics.resolve()), "status": "failed",
        })
        raise
