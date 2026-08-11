from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_sync_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle).get("artifacts") or {}
    remote = str(value.get("remote") or "").rstrip("/")
    roots = [str(item).strip("/") for item in value.get("roots") or [] if str(item).strip("/")]
    if not remote or ":" not in remote:
        raise ValueError("artifacts.remote must be an rsync SSH target such as user@host:/absolute/path")
    if not roots:
        raise ValueError("artifacts.roots must contain at least one relative directory")
    return {"remote": remote, "roots": roots}


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def sync_artifacts(action: str, config_path: Path, project: Path,
                   dry_run: bool = False) -> dict[str, Any]:
    if action not in {"status", "push", "pull"}:
        raise ValueError(f"Unsupported sync action: {action}")
    config = load_sync_config(config_path)
    project = project.resolve()
    operations = []
    for root in config["roots"]:
        local = project / root
        remote = f"{config['remote']}/{root}"
        if action in {"status", "push"} and not local.exists():
            operations.append({"root": root, "direction": "push", "status": "local_missing"})
            continue
        directions = ("push", "pull") if action == "status" else (action,)
        for direction in directions:
            source, destination = ((str(local) + "/", remote + "/") if direction == "push"
                                   else (remote + "/", str(local) + "/"))
            command = ["rsync", "-az", "--itemize-changes", "--checksum"]
            if action == "status" or dry_run:
                command.append("--dry-run")
            else:
                # Artifact paths are immutable. A differing existing file is a
                # conflict to inspect, never something to overwrite silently.
                command.append("--ignore-existing")
            command.extend([source, destination])
            result = _run(command)
            operations.append({"root": root, "direction": direction,
                               "status": "success" if result["returncode"] == 0 else "failed",
                               **result})
    failed = sum(item.get("status") == "failed" for item in operations)
    return {"status": "success" if not failed else "failed", "action": action,
            "dry_run": action == "status" or dry_run, "operation_count": len(operations),
            "failed_operation_count": failed, "operations": operations}
