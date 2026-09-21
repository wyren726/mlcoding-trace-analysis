from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REGISTRY_SCHEMA_VERSION = "1.0"
KINDS = {"datasets", "feature-runs", "analysis-runs", "pipeline-runs"}


def project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def registry_root_for_output(output_root: Path) -> Path:
    project = project_root()
    try:
        output_root.resolve().relative_to(project)
        return project
    except ValueError:
        return output_root.resolve().parent


def artifact_metadata(paths: Iterable[Path], root: Path | None = None) -> list[dict[str, Any]]:
    base = (root or project_root()).resolve()
    result = []
    for path in paths:
        resolved = path.resolve()
        try:
            logical_path = str(resolved.relative_to(base))
        except ValueError:
            logical_path = str(resolved)
        result.append({
            "logical_path": logical_path,
            "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
            "sha256": None,
            "hash_status": "deferred",
        })
    return result


def write_immutable_record(kind: str, record_id: str, record: dict[str, Any],
                           artifacts: Iterable[Path] = (), root: Path | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"Unknown registry kind: {kind}")
    base = (root or project_root()).resolve()
    destination = base / "registries" / kind / f"{record_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "kind": kind,
        "record_id": record_id,
        "registered_at": dt.datetime.now().astimezone().isoformat(),
        "record": record,
        "artifacts": artifact_metadata(artifacts, base),
    }
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        # Registration time is not semantic; an identical record is idempotent.
        comparable_previous = {k: v for k, v in previous.items() if k != "registered_at"}
        comparable_payload = {k: v for k, v in payload.items() if k != "registered_at"}
        if comparable_previous != comparable_payload:
            raise ValueError(f"Immutable registry record already exists with different content: {destination}")
        return destination
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def registry_records(kind: str, root: Path | None = None) -> list[dict[str, Any]]:
    if kind not in KINDS:
        raise ValueError(f"Unknown registry kind: {kind}")
    directory = (root or project_root()) / "registries" / kind
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))]


def rebuild_registry_csv(kind: str, output: Path, root: Path | None = None) -> dict[str, Any]:
    records = registry_records(kind, root)
    rows = [item.get("record") or {} for item in records]
    fields = sorted({key for row in rows for key in row})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    return {"status": "success", "kind": kind, "record_count": len(rows),
            "output_path": str(output.resolve())}


def build_artifact_manifest(output: Path, root: Path | None = None) -> dict[str, Any]:
    base = (root or project_root()).resolve()
    indexed: dict[str, dict[str, Any]] = {}
    for kind in sorted(KINDS):
        for registry in registry_records(kind, base):
            for artifact in registry.get("artifacts") or []:
                logical = str(artifact.get("logical_path") or "")
                if not logical:
                    continue
                path = Path(logical)
                resolved = path if path.is_absolute() else base / path
                if not resolved.is_file():
                    indexed.setdefault(logical, {
                        "logical_path": logical, "size_bytes": artifact.get("size_bytes"),
                        "sha256": None, "status": "missing",
                        "source_record_id": registry.get("record_id"),
                    })
                    continue
                digest = hashlib.sha256()
                with resolved.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                indexed[logical] = {
                    "logical_path": logical, "size_bytes": resolved.stat().st_size,
                    "sha256": digest.hexdigest(), "status": "available",
                    "source_record_id": registry.get("record_id"),
                }
    manifest = {"manifest_version": "1.0", "generated_at": dt.datetime.now().astimezone().isoformat(),
                "artifacts": [indexed[key] for key in sorted(indexed)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "success", "artifact_count": len(indexed),
            "available_count": sum(item["status"] == "available" for item in indexed.values()),
            "missing_count": sum(item["status"] == "missing" for item in indexed.values()),
            "output_path": str(output.resolve())}
