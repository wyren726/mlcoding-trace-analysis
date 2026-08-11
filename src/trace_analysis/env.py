from __future__ import annotations

import os
from pathlib import Path


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.split(" #", 1)[0].rstrip()


def load_project_env(path: Path | None = None) -> Path | None:
    """Load a local .env without overriding variables already in the process."""
    candidates = [path] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    env_path = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
    if env_path is None:
        return None
    with env_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, raw_value = line.partition("=")
            key = key.strip()
            if not separator or not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
                continue
            os.environ.setdefault(key, _parse_value(raw_value))
    return env_path.resolve()
