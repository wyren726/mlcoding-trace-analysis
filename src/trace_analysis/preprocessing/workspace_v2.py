from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .adapters import choose_adapter
from .adapters.common import redact, redact_text, stable_id
from .pipeline import (
    _zip_lines,
    _zip_source_metadata,
    run_preprocess,
    run_preprocess_collector_zip,
    source_files,
)
from ..workspace import (
    WorkspaceLayout,
    WorkspaceRegistry,
    default_dataset_id,
    registry_record_id,
    validate_workspace_id,
)


WORKSPACE_PREPROCESSOR_VERSION = "v2.0.0-dual-write"
DEFAULT_SHARD_SIZE_BYTES = 256 * 1024 * 1024


class JsonlShardWriter:
    def __init__(self, directory: Path, shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES) -> None:
        if shard_size_bytes <= 0:
            raise ValueError("shard_size_bytes must be positive")
        self.directory = directory
        self.shard_size_bytes = shard_size_bytes
        self.directory.mkdir(parents=True, exist_ok=False)
        self.paths: list[Path] = []
        self.record_count = 0
        self.size_bytes = 0
        self._handle: Any = None
        self._current_size = 0

    def _open_next(self) -> None:
        path = self.directory / f"part-{len(self.paths):05d}.jsonl"
        self.paths.append(path)
        self._handle = path.open("wb")
        self._current_size = 0

    def write(self, record: dict[str, Any]) -> None:
        encoded = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if self._handle is None:
            self._open_next()
        elif self._current_size and self._current_size + len(encoded) > self.shard_size_bytes:
            self._handle.close()
            self._open_next()
        self._handle.write(encoded)
        self._current_size += len(encoded)
        self.size_bytes += len(encoded)
        self.record_count += 1

    def close(self) -> None:
        if self._handle is None:
            self._open_next()
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "JsonlShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _source_streams(inputs: list[str], zip_source: Path | None
                    ) -> tuple[list[tuple[str, str, Any]], dict[str, Any]]:
    if zip_source:
        metadata = _zip_source_metadata(zip_source)
        member = str(metadata["source_member"])
        return [(
            str(zip_source), str(metadata["source_uri"]),
            lambda: _zip_lines(zip_source, member),
        )], metadata
    files = source_files(inputs)
    return [
        (str(path), str(path),
         lambda path=path: path.open("r", encoding="utf-8", errors="replace"))
        for path in files
    ], {}


def _projection(event: dict[str, Any]) -> dict[str, Any]:
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    source_metadata = {
        key: value for key, value in source.items()
        if key not in {"input_file", "record_index", "adapter"}
    }
    return {
        "projection_schema_version": "v2.0",
        "event_id": str(event["event_id"]),
        "event_type": str(event.get("type") or "unknown"),
        "data": event.get("data"),
        "observed_identity": {
            "session_id": event.get("session_id"),
            "turn_id": event.get("turn_id"),
            "sequence": event.get("sequence"),
            "timestamp": event.get("timestamp"),
        },
        "source_metadata": source_metadata,
        "raw_json_pointer": "/payload",
    }


def _insert_event_reference(database: sqlite3.Connection, event_id: str,
                            source_record_id: str, json_pointer: str) -> None:
    previous = database.execute(
        "SELECT source_record_id, json_pointer FROM event_refs WHERE event_id = ?", (event_id,)
    ).fetchone()
    if previous is not None:
        if previous != (source_record_id, json_pointer):
            raise ValueError(f"Conflicting source references for event_id {event_id}")
        return
    database.execute(
        "INSERT INTO event_refs VALUES (?, ?, ?)",
        (event_id, source_record_id, json_pointer),
    )


def _write_source_products(inputs: list[str], zip_source: Path | None, batch_id: str,
                           dataset_id: str, requested_adapter: str | None,
                           limit: int | None, source_writer: JsonlShardWriter,
                           lineage_writer: JsonlShardWriter,
                           database: sqlite3.Connection) -> dict[str, Any]:
    streams, zip_metadata = _source_streams(inputs, zip_source)
    adapter_cache: dict[str, Any] = {}
    adapters: Counter[str] = Counter()
    seen = 0
    rejected = 0
    for source_label, source_uri, open_lines in streams:
        lines = iter(open_lines())
        first_line = next((line for line in lines if line.strip()), None)
        if first_line is None:
            raise ValueError(f"Empty input: {source_label}")
        first = json.loads(first_line)
        if not isinstance(first, dict):
            raise ValueError(f"First record is not an object: {source_label}")
        detected = choose_adapter(first, requested_adapter)
        adapter = adapter_cache.setdefault(detected.name, detected)
        adapters[adapter.name] += 1

        def records() -> Iterable[str]:
            yield first_line
            yield from lines

        try:
            for record_index, line in enumerate(records()):
                if not line.strip():
                    continue
                if limit is not None and seen >= limit:
                    break
                seen += 1
                parsed: Any = None
                converted_events: list[dict[str, Any]] = []
                error: str | None = None
                try:
                    parsed = json.loads(line)
                    converted = adapter.convert(
                        parsed, batch_id=batch_id, source_file=source_label,
                        record_index=record_index,
                    )
                    converted_events = converted if isinstance(converted, list) else [converted]
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    rejected += 1
                    error = f"{type(exc).__name__}: {redact_text(str(exc))}"
                identity_value = parsed if parsed is not None else line.rstrip("\n")
                source_record_id = stable_id(
                    "src", source_label, record_index, identity_value
                )
                projections = [_projection(event) for event in converted_events]
                source_record = {
                    "schema_version": "v2.0",
                    "dataset_id": dataset_id,
                    "batch_id": batch_id,
                    "source_record_id": source_record_id,
                    "source_format": adapter.name,
                    "source_locator": {
                        "source_uri": source_uri,
                        "source_file": source_label,
                        "source_member": zip_metadata.get("source_member"),
                        "record_index": record_index,
                    },
                    "status": "rejected" if error else "normalized",
                    "payload": redact(parsed) if parsed is not None else None,
                    "raw_text": redact_text(line.rstrip("\n")) if parsed is None else None,
                    "normalized_events": projections,
                    "error": error,
                }
                source_writer.write(source_record)
                for projection_index, event in enumerate(converted_events):
                    event_id = str(event["event_id"])
                    pointer = f"/normalized_events/{projection_index}"
                    _insert_event_reference(
                        database, event_id, source_record_id, pointer
                    )
                    lineage_writer.write({
                        "schema_version": "v2.0",
                        "lineage_edge_id": stable_id(
                            "lin", source_record_id, pointer, event_id
                        ),
                        "edge_type": "normalized_from",
                        "from_ref": {
                            "source_record_id": source_record_id,
                            "json_pointer": pointer,
                        },
                        "to_ref": {"record_type": "event", "event_id": event_id},
                    })
                if seen % 1000 == 0:
                    database.commit()
        finally:
            close = getattr(lines, "close", None)
            if close:
                close()
        if limit is not None and seen >= limit:
            break
    database.commit()
    return {
        "source_record_count": seen,
        "rejected_record_count": rejected,
        "adapters": sorted(adapters),
    }


def _source_ref(database: sqlite3.Connection, event_id: str) -> dict[str, str]:
    row = database.execute(
        "SELECT source_record_id, json_pointer FROM event_refs WHERE event_id = ?", (event_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No v2 source reference for legacy event_id {event_id}")
    return {"source_record_id": str(row[0]), "json_pointer": str(row[1])}


def _v2_event(event: dict[str, Any], dataset_id: str, batch_id: str,
              database: sqlite3.Connection) -> dict[str, Any]:
    return {
        "schema_version": "v2.0",
        "dataset_id": dataset_id,
        "batch_id": batch_id,
        "event_id": str(event["event_id"]),
        "session_id": event.get("session_id"),
        "trace_id": event.get("trace_id"),
        "turn_id": event.get("turn_id"),
        "sequence": event.get("sequence"),
        "timestamp": event.get("timestamp"),
        "type": event.get("type"),
        "source_ref": _source_ref(database, str(event["event_id"])),
    }


def _execution_context(turn: dict[str, Any]) -> dict[str, Any]:
    events = turn.get("events") if isinstance(turn.get("events"), list) else []
    model_calls = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "model_call":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        model_calls.append({
            "event_id": event.get("event_id"),
            "provider": data.get("provider") or "unknown",
            "api_model": data.get("model") or data.get("model_name") or "unknown",
            "model_snapshot": data.get("model_snapshot"),
        })
    if not model_calls:
        model_calls = [{
            "event_id": None,
            "provider": "unknown",
            "api_model": model,
            "model_snapshot": None,
        } for model in turn.get("agent_models") or []]
    tools = sorted({
        str(data.get("tool_name") or "unknown")
        for event in events if isinstance(event, dict) and event.get("type") == "tool_call"
        for data in [event.get("data") if isinstance(event.get("data"), dict) else {}]
    })
    harness = str(turn.get("harness") or "unknown")
    providers = {str(call["provider"]) for call in model_calls}
    return {
        "harness": {"name": harness, "version": None},
        "model_calls": model_calls,
        "tools": tools,
        "permissions": None,
        "context_policy": None,
        "metadata_coverage": {
            "harness": "observed" if harness != "unknown" else "unknown",
            "provider": "observed" if providers and providers != {"unknown"} else "unknown",
            "permissions": "unknown",
            "context_policy": "unknown",
        },
    }


def _write_v2_turns(legacy_turn_path: Path, dataset_id: str, batch_id: str,
                    writer: JsonlShardWriter, database: sqlite3.Connection) -> int:
    event_count = 0
    with legacy_turn_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            turn = json.loads(line)
            events = turn.get("events") if isinstance(turn.get("events"), list) else []
            v2_events = [
                _v2_event(event, dataset_id, batch_id, database)
                for event in events if isinstance(event, dict)
            ]
            source_record_ids = sorted({
                event["source_ref"]["source_record_id"] for event in v2_events
            })
            writer.write({
                "schema_version": "v2.0",
                "dataset_id": dataset_id,
                "batch_id": batch_id,
                "trace_id": turn.get("trace_id"),
                "session_id": turn.get("session_id"),
                "turn_id": turn.get("turn_id"),
                "turn_index": turn.get("turn_index"),
                "status": turn.get("status"),
                "execution_context": _execution_context(turn),
                "events": v2_events,
                "lineage": {
                    "source_record_ids": source_record_ids,
                    "legacy_schema_version": turn.get("schema_version"),
                },
            })
            event_count += len(v2_events)
    return event_count


def _write_v2_traces(legacy_trace_path: Path, dataset_id: str, batch_id: str,
                     writer: JsonlShardWriter, database: sqlite3.Connection) -> int:
    trace_count = 0
    with legacy_trace_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            sessions = []
            for session in trace.get("sessions") or []:
                if not isinstance(session, dict):
                    continue
                turn_refs = [
                    {
                        "turn_id": turn.get("turn_id"),
                        "turn_index": turn.get("turn_index"),
                        "status": turn.get("status"),
                    }
                    for turn in session.get("turns") or [] if isinstance(turn, dict)
                ]
                unassigned = [
                    _v2_event(event, dataset_id, batch_id, database)
                    for event in session.get("unassigned_events") or []
                    if isinstance(event, dict)
                ]
                sessions.append({
                    "session_id": session.get("session_id"),
                    "harness": session.get("harness"),
                    "turn_refs": turn_refs,
                    "unassigned_events": unassigned,
                })
            writer.write({
                "schema_version": "v2.0",
                "dataset_id": dataset_id,
                "batch_id": batch_id,
                "trace_id": trace.get("trace_id"),
                "sessions": sessions,
                "lineage": {"legacy_schema_version": trace.get("schema_version")},
            })
            trace_count += 1
    return trace_count


def _parts(directory: Path) -> list[Path]:
    return sorted(directory.glob("part-*.jsonl"))


def _register_materialization(layout: WorkspaceLayout, manifest: dict[str, Any],
                              batch_dir: Path) -> dict[str, str]:
    registry = WorkspaceRegistry(layout)
    dataset_id = str(manifest["dataset_id"])
    batch_id = str(manifest["batch_id"])
    run_id = str(manifest["run_id"])
    preprocess_dir = batch_dir / "01_preprocess"
    product_specs = {
        "source_records": _parts(preprocess_dir / "source_records"),
        "unified_turns": _parts(preprocess_dir / "unified_turns"),
        "unified_traces": _parts(preprocess_dir / "unified_traces"),
        "lineage_edges": _parts(preprocess_dir / "lineage_edges"),
    }
    registry_paths: dict[str, str] = {}
    dataset_registry = registry.write("datasets", dataset_id, {
        "dataset_id": dataset_id,
        "status": "active",
        "format": "trace_analysis_workspace_v2",
    })
    registry_paths["dataset"] = str(dataset_registry.resolve())
    product_record_ids = []
    for product_id, artifacts in product_specs.items():
        product_record_id = registry_record_id(
            "prod", dataset_id, batch_id, product_id, "v2"
        )
        product_record_ids.append(product_record_id)
        product_registry = registry.write("products", product_record_id, {
            "product_record_id": product_record_id,
            "product_id": product_id,
            "product_version": "v2",
            "stage_id": "01_preprocess",
            "dataset_id": dataset_id,
            "batch_id": batch_id,
            "run_id": run_id,
            "status": manifest["status"],
            "record_count": manifest["products"][product_id]["record_count"],
            "artifact_paths": [str(path.resolve()) for path in artifacts],
            "source_fingerprint": manifest["source_fingerprint"],
        }, artifacts)
        registry_paths[f"product:{product_id}"] = str(product_registry.resolve())
    all_artifacts = [path for paths in product_specs.values() for path in paths]
    all_artifacts.append(batch_dir / "batch_manifest.json")
    batch_record_id = registry_record_id("batchreg", dataset_id, batch_id)
    batch_registry = registry.write("batches", batch_record_id, {
        "batch_record_id": batch_record_id,
        "dataset_id": dataset_id,
        "batch_id": batch_id,
        "status": manifest["status"],
        "created_at": manifest["created_at"],
        "batch_path": str(batch_dir.resolve()),
        "source_fingerprint": manifest["source_fingerprint"],
        "legacy_batch_path": manifest["legacy_batch_path"],
        "event_count": manifest["event_count"],
        "turn_count": manifest["products"]["unified_turns"]["record_count"],
        "trace_count": manifest["trace_count"],
        "run_id": run_id,
        "product_record_ids": product_record_ids,
    }, all_artifacts)
    registry_paths["batch"] = str(batch_registry.resolve())
    run_registry = registry.write("runs", run_id, {
        "run_id": run_id,
        "run_type": "preprocess_v2_materialization",
        "stage_id": "01_preprocess",
        "dataset_id": dataset_id,
        "batch_id": batch_id,
        "status": manifest["status"],
        "runner_version": manifest["preprocessor_version"],
        "source_fingerprint": manifest["source_fingerprint"],
        "input_product_ids": ["legacy_unified_turns.v1", "legacy_unified_traces.v1"],
        "output_product_record_ids": product_record_ids,
    }, (batch_dir / "batch_manifest.json",))
    registry_paths["run"] = str(run_registry.resolve())
    dataset_manifest = registry.rebuild_dataset_manifest(dataset_id)
    registry_paths["dataset_manifest"] = str(dataset_manifest.resolve())
    return registry_paths


def materialize_workspace_v2(inputs: list[str], legacy_result: dict[str, Any],
                             workspace_root: Path, dataset_id: str,
                             requested_adapter: str | None, limit: int | None,
                             zip_source: Path | None = None,
                             shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES) -> dict[str, Any]:
    dataset_id = validate_workspace_id(dataset_id, "dataset_id")
    batch_id = validate_workspace_id(str(legacy_result["batch_id"]), "batch_id")
    layout = WorkspaceLayout(workspace_root)
    destination = layout.batch_dir(dataset_id, batch_id)
    if destination.exists():
        manifest_path = destination / "batch_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Existing workspace batch has no manifest: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(legacy_result.get("input_fingerprint") or
                       legacy_result.get("source_fingerprint") or "")
        if (manifest.get("source_fingerprint") != expected
                or manifest.get("preprocessor_version") != WORKSPACE_PREPROCESSOR_VERSION):
            raise ValueError(f"Existing immutable workspace batch is incompatible: {destination}")
        registry_paths = _register_materialization(layout, manifest, destination)
        return {
            **manifest,
            "output_path": str(destination.resolve()),
            "deduplicated": True,
            "registry_paths": registry_paths,
        }

    legacy_batch = Path(str(legacy_result["output_path"])).resolve()
    legacy_turn_path = legacy_batch / "unified_turns.jsonl"
    legacy_trace_path = legacy_batch / "unified_traces.jsonl"
    if not legacy_turn_path.is_file() or not legacy_trace_path.is_file():
        raise FileNotFoundError(f"Legacy batch is incomplete: {legacy_batch}")
    layout.temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f"{dataset_id}.{batch_id}.", dir=layout.temporary_root
    ))
    preprocess_dir = temporary / "01_preprocess"
    database_path = temporary / "event_refs.sqlite"
    database = sqlite3.connect(database_path)
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute(
        "CREATE TABLE event_refs (event_id TEXT PRIMARY KEY, source_record_id TEXT NOT NULL, "
        "json_pointer TEXT NOT NULL)"
    )
    created_at = dt.datetime.now().astimezone().isoformat()
    try:
        with (
            JsonlShardWriter(preprocess_dir / "source_records", shard_size_bytes) as source_writer,
            JsonlShardWriter(preprocess_dir / "lineage_edges", shard_size_bytes) as lineage_writer,
            JsonlShardWriter(preprocess_dir / "unified_turns", shard_size_bytes) as turn_writer,
            JsonlShardWriter(preprocess_dir / "unified_traces", shard_size_bytes) as trace_writer,
        ):
            source_summary = _write_source_products(
                inputs, zip_source, batch_id, dataset_id, requested_adapter, limit,
                source_writer, lineage_writer, database,
            )
            turn_event_count = _write_v2_turns(
                legacy_turn_path, dataset_id, batch_id, turn_writer, database
            )
            trace_count = _write_v2_traces(
                legacy_trace_path, dataset_id, batch_id, trace_writer, database
            )
            source_writer.close()
            lineage_writer.close()
            turn_writer.close()
            trace_writer.close()
        mapped_event_count = int(database.execute("SELECT COUNT(*) FROM event_refs").fetchone()[0])
        expected_event_count = int(legacy_result.get("event_count") or 0)
        if mapped_event_count != expected_event_count:
            raise ValueError(
                f"v2 event mapping coverage mismatch: expected {expected_event_count}, "
                f"mapped {mapped_event_count}"
            )
        if turn_event_count > mapped_event_count:
            raise ValueError("v2 Turn event count exceeds mapped source events")
        database.close()
        database_path.unlink()
        status = "partial" if source_summary["rejected_record_count"] else "success"
        source_fingerprint = str(
            legacy_result.get("input_fingerprint")
            or legacy_result.get("source_fingerprint") or ""
        )
        run_id = registry_record_id(
            "run", dataset_id, batch_id, "01_preprocess", WORKSPACE_PREPROCESSOR_VERSION
        )
        products = {
            "source_records": {
                "version": "v2", "record_count": source_writer.record_count,
                "parts": [str(path.relative_to(temporary)) for path in source_writer.paths],
                "size_bytes": source_writer.size_bytes,
            },
            "unified_turns": {
                "version": "v2", "record_count": turn_writer.record_count,
                "parts": [str(path.relative_to(temporary)) for path in turn_writer.paths],
                "size_bytes": turn_writer.size_bytes,
            },
            "unified_traces": {
                "version": "v2", "record_count": trace_writer.record_count,
                "parts": [str(path.relative_to(temporary)) for path in trace_writer.paths],
                "size_bytes": trace_writer.size_bytes,
            },
            "lineage_edges": {
                "version": "v2", "record_count": lineage_writer.record_count,
                "parts": [str(path.relative_to(temporary)) for path in lineage_writer.paths],
                "size_bytes": lineage_writer.size_bytes,
            },
        }
        manifest = {
            "manifest_schema_version": "2.0",
            "preprocessor_version": WORKSPACE_PREPROCESSOR_VERSION,
            "created_at": created_at,
            "dataset_id": dataset_id,
            "batch_id": batch_id,
            "run_id": run_id,
            "status": status,
            "source_fingerprint": source_fingerprint,
            "sources": inputs,
            "adapter": "+".join(source_summary["adapters"]),
            "limit": limit,
            "legacy_batch_path": str(legacy_batch),
            "source_record_count": source_summary["source_record_count"],
            "rejected_record_count": source_summary["rejected_record_count"],
            "event_count": mapped_event_count,
            "turn_event_count": turn_event_count,
            "trace_count": trace_count,
            "products": products,
        }
        (temporary / "batch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
    except Exception:
        database.close()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    registry_paths = _register_materialization(layout, manifest, destination)
    return {
        **manifest,
        "output_path": str(destination.resolve()),
        "deduplicated": False,
        "registry_paths": registry_paths,
    }


def run_workspace_preprocess(inputs: list[str], workspace_root: Path,
                             legacy_output_root: Path, requested_adapter: str | None,
                             limit: int | None, dataset_id: str | None = None,
                             shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one source is required")
    resolved_dataset_id = dataset_id or default_dataset_id(inputs)
    zip_source: Path | None = None
    if len(inputs) == 1 and Path(inputs[0]).expanduser().suffix.lower() == ".zip":
        zip_source = Path(inputs[0]).expanduser().resolve()
        legacy = run_preprocess_collector_zip(str(zip_source), legacy_output_root, limit)
        adapter = "collector_events"
    else:
        legacy = run_preprocess(inputs, legacy_output_root, requested_adapter, limit)
        adapter = requested_adapter
    v2 = materialize_workspace_v2(
        inputs, legacy, workspace_root, resolved_dataset_id, adapter, limit,
        zip_source=zip_source, shard_size_bytes=shard_size_bytes,
    )
    return {
        "status": v2["status"],
        "dataset_id": resolved_dataset_id,
        "batch_id": v2["batch_id"],
        "legacy": legacy,
        "workspace_v2": v2,
    }


__all__ = [
    "DEFAULT_SHARD_SIZE_BYTES", "JsonlShardWriter", "WORKSPACE_PREPROCESSOR_VERSION",
    "materialize_workspace_v2", "run_workspace_preprocess",
]
