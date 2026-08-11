from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from ...env import load_project_env


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    allowed_models: tuple[str, ...]
    timeout_seconds: int
    max_retries: int
    requests_per_minute: int


def load_provider(path: Path, name: str) -> ProviderConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle).get("providers", {}).get(name)
    if not isinstance(raw, dict):
        raise ValueError(f"Unknown provider: {name}")
    if raw.get("api_type") != "openai_compatible":
        raise ValueError(f"Unsupported API type for provider {name}")
    return ProviderConfig(
        name=name, base_url=str(raw["base_url"]).rstrip("/"), api_key_env=str(raw["api_key_env"]),
        default_model=str(raw["default_model"]), allowed_models=tuple(raw.get("allowed_models") or ()),
        timeout_seconds=int(raw.get("timeout_seconds", 120)), max_retries=int(raw.get("max_retries", 4)),
        requests_per_minute=int(raw.get("requests_per_minute", 90)),
    )


class OpenAICompatibleClient:
    def __init__(self, config: ProviderConfig, model: str | None = None, cache_dir: Path | None = None,
                 timeout_seconds: int | None = None, max_retries: int | None = None) -> None:
        load_project_env()
        self.config = config
        self.model = model or config.default_model
        if config.allowed_models and self.model not in config.allowed_models:
            raise ValueError(f"Model {self.model!r} is not allowed for provider {config.name}")
        self.api_key = os.environ.get(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key environment variable: {config.api_key_env}")
        self.cache_dir = cache_dir
        self.max_retries = max_retries if max_retries is not None else config.max_retries
        self._request_interval = 60.0 / config.requests_per_minute
        self._last_request_at = 0.0
        self._rate_lock = threading.Lock()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=config.base_url,
            timeout=timeout_seconds if timeout_seconds is not None else config.timeout_seconds,
            # Retries are handled below so every retry is also rate limited.
            max_retries=0,
        )

    def _wait_for_request_slot(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            wait_seconds = self._request_interval - (now - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        request_body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        cache_key = hashlib.sha256(json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            response = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._parse(response), {"cache_hit": True, "cache_key": cache_key}
        for attempt in range(self.max_retries + 1):
            self._wait_for_request_slot()
            try:
                response = self._client.chat.completions.create(**request_body)
                break
            except Exception:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raw = response.model_dump()
        if cache_path:
            cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return self._parse(raw), {"cache_hit": False, "cache_key": cache_key, "usage": raw.get("usage")}

    @staticmethod
    def _parse(response: dict[str, Any]) -> dict[str, Any]:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model response content is not text")
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        field = re.search(r'"[A-Za-z_][A-Za-z0-9_]*"\s*:', text)
        bases = [text]
        if field and field.start() > 1:
            bases.append("{" + text[field.start():])
        candidates = [candidate + suffix for candidate in bases for suffix in ("", "}", "]", "}}", "]}")]
        last_error: json.JSONDecodeError | None = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            assert last_error is not None
            raise last_error
        if not isinstance(value, dict):
            raise ValueError("Model response JSON is not an object")
        return value
