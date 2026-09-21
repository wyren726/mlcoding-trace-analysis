from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_REGISTRY_SCHEMA_VERSION = "2.0"
WORKSPACE_REGISTRY_KINDS = {"datasets", "batches", "runs", "products"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_workspace_id(value: str, label: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{label} must match {_SAFE_ID.pattern!r}; received {value!r}"
        )
    return value


def _logical_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _artifact_metadata(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        resolved = path.resolve()
        result.append({
            "logical_path": _logical_path(resolved, root),
            "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
            "sha256": None,
            "hash_status": "deferred",
        })
    return result


@dataclass(frozen=True)
class WorkspaceLayout:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    @property
    def registry_root(self) -> Path:
        return self.root / "registry"

    @property
    def temporary_root(self) -> Path:
        return self.root / ".tmp"

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / "datasets" / validate_workspace_id(dataset_id, "dataset_id")

    def batch_dir(self, dataset_id: str, batch_id: str) -> Path:
        validate_workspace_id(batch_id, "batch_id")
        return self.dataset_dir(dataset_id) / "batches" / batch_id


class WorkspaceRegistry:
    """Immutable v2 registry records stored under one workspace root."""

    def __init__(self, layout: WorkspaceLayout) -> None:
        self.layout = layout

    def write(self, kind: str, record_id: str, record: dict[str, Any],
              artifacts: Iterable[Path] = ()) -> Path:
        if kind not in WORKSPACE_REGISTRY_KINDS:
            raise ValueError(f"Unknown workspace registry kind: {kind}")
        validate_workspace_id(record_id, "record_id")
        destination = self.layout.registry_root / kind / f"{record_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registry_schema_version": WORKSPACE_REGISTRY_SCHEMA_VERSION,
            "kind": kind,
            "record_id": record_id,
            "registered_at": dt.datetime.now().astimezone().isoformat(),
            "record": record,
            "artifacts": _artifact_metadata(artifacts, self.layout.root),
        }
        if destination.exists():
            previous = json.loads(destination.read_text(encoding="utf-8"))
            comparable_previous = {key: value for key, value in previous.items()
                                   if key != "registered_at"}
            comparable_payload = {key: value for key, value in payload.items()
                                  if key != "registered_at"}
            if comparable_previous != comparable_payload:
                raise ValueError(
                    f"Immutable workspace registry record exists with different content: "
                    f"{destination}"
                )
            return destination
        temporary = destination.with_suffix(f".json.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return destination

    def records(self, kind: str) -> list[dict[str, Any]]:
        if kind not in WORKSPACE_REGISTRY_KINDS:
            raise ValueError(f"Unknown workspace registry kind: {kind}")
        directory = self.layout.registry_root / kind
        if not directory.exists():
            return []
        result = []
        for path in sorted(directory.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def rebuild_dataset_manifest(self, dataset_id: str) -> Path:
        validate_workspace_id(dataset_id, "dataset_id")
        batches = []
        for wrapper in self.records("batches"):
            record = wrapper.get("record") if isinstance(wrapper.get("record"), dict) else {}
            if record.get("dataset_id") == dataset_id:
                batches.append({
                    "batch_id": record.get("batch_id"),
                    "batch_record_id": wrapper.get("record_id"),
                    "status": record.get("status"),
                    "created_at": record.get("created_at"),
                    "batch_path": record.get("batch_path"),
                    "source_fingerprint": record.get("source_fingerprint"),
                })
        batches.sort(key=lambda row: (str(row.get("created_at") or ""),
                                      str(row.get("batch_id") or "")))
        destination = self.layout.dataset_dir(dataset_id) / "dataset_manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_schema_version": "2.0",
            "generated_at": dt.datetime.now().astimezone().isoformat(),
            "dataset_id": dataset_id,
            "batch_count": len(batches),
            "batches": batches,
            "canonical_registry": _logical_path(
                self.layout.registry_root / "batches", self.layout.root
            ),
        }
        temporary = destination.with_suffix(f".json.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return destination


def default_dataset_id(inputs: Iterable[str]) -> str:
    paths = [Path(value).expanduser().resolve() for value in inputs]
    if not paths:
        raise ValueError("At least one source is required to derive dataset_id")
    anchors = [path if path.is_dir() else path.parent for path in paths]
    common = Path(os.path.commonpath([str(path) for path in anchors]))
    if common == Path(common.anchor):
        identity = "|".join(sorted(str(path) for path in anchors))
        label = "multi_source"
    else:
        identity = str(common)
        label = common.name or "data"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or "data"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
    return f"dataset_{slug[:40]}_{digest}"


def registry_record_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"{prefix}_{digest}"


__all__ = [
    "WORKSPACE_REGISTRY_KINDS", "WORKSPACE_REGISTRY_SCHEMA_VERSION",
    "WorkspaceLayout", "WorkspaceRegistry", "default_dataset_id", "registry_record_id",
    "validate_workspace_id",
]
