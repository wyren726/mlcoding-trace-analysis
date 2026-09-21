from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import httpx
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..env import load_project_env


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
    reasoning_effort: str | None
    max_completion_tokens: int | None
    use_env_proxy: bool


class ThinkingModeMismatchError(RuntimeError):
    """The provider returned reasoning although No Thinking was requested."""


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
        reasoning_effort=(str(raw["reasoning_effort"]) if raw.get("reasoning_effort") else None),
        max_completion_tokens=(
            int(raw["max_completion_tokens"]) if raw.get("max_completion_tokens") else None
        ),
        use_env_proxy=bool(raw.get("use_env_proxy", True)),
    )


class OpenAICompatibleClient:
    def __init__(self, config: ProviderConfig, model: str | None = None, cache_dir: Path | None = None,
                 timeout_seconds: int | None = None, max_retries: int | None = None,
                 thinking_type: str | None = None,
                 max_completion_tokens: int | None = None,
                 api_key: str | None = None) -> None:
        load_project_env()
        self.config = config
        self.model = model or config.default_model
        if config.allowed_models and self.model not in config.allowed_models:
            raise ValueError(f"Model {self.model!r} is not allowed for provider {config.name}")
        self.api_key = api_key or os.environ.get(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key environment variable: {config.api_key_env}")
        self.cache_dir = cache_dir
        self.max_retries = max_retries if max_retries is not None else config.max_retries
        if max_completion_tokens is not None and max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        self.max_completion_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else config.max_completion_tokens
        )
        if thinking_type not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled, disabled, or None")
        self.thinking_type = thinking_type
        self._request_interval = 60.0 / config.requests_per_minute
        self._last_request_at = 0.0
        self._rate_lock = threading.Lock()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.service_backoff = None
        if os.environ.get('TRACE_ANALYSIS_503_BACKOFF') == '1' and cache_dir:
            from .service_backoff import ServiceBackoff
            self.service_backoff = ServiceBackoff(cache_dir / 'service-retry-v1')
        client_options: dict[str, Any] = {}
        if not config.use_env_proxy:
            # Cluster-local ``*.svc`` traffic must not be sent through the
            # workstation HTTP proxy; large request bodies otherwise fail with
            # EOF/closed-connection errors before reaching the model server.
            client_options["http_client"] = httpx.Client(
                trust_env=False,
                timeout=timeout_seconds if timeout_seconds is not None else config.timeout_seconds,
            )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=config.base_url,
            timeout=timeout_seconds if timeout_seconds is not None else config.timeout_seconds,
            # Retries are handled below so every retry is also rate limited.
            max_retries=0,
            **client_options,
        )

    def _wait_for_request_slot(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            wait_seconds = self._request_interval - (now - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._complete_json_with_response_format(
            system, user, {"type": "json_object"}
        )

    def complete_json_schema(
        self, system: str, user: str, *, name: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._complete_json_with_response_format(
            system, user,
            {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema,
            }},
        )

    def _complete_json_with_response_format(
        self, system: str, user: str, response_format: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_body = {"model": self.model, "temperature": 0, "response_format": response_format,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        config = getattr(self, "config", None)
        reasoning_effort = getattr(config, "reasoning_effort", None)
        # Some lightweight test and compatibility clients are constructed via
        # ``__new__``; fall back to the provider value when no instance override
        # has been installed by ``__init__``.
        max_completion_tokens = getattr(
            self, "max_completion_tokens",
            getattr(config, "max_completion_tokens", None),
        )
        thinking_type = getattr(self, "thinking_type", None)
        if thinking_type == "disabled":
            # AILab's OpenAI-compatible GLM endpoint currently ignores
            # ``thinking.type=disabled`` on its own.  Both controls below were
            # verified against glm-5.2-1m: each produces zero reasoning tokens.
            request_body["extra_body"] = {
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        elif thinking_type:
            # The OpenAI SDK merges provider-specific extra_body fields into
            # the JSON request body.
            request_body["extra_body"] = {"thinking": {"type": thinking_type}}
        if reasoning_effort and thinking_type not in {"disabled"}:
            request_body["reasoning_effort"] = reasoning_effort
        if max_completion_tokens:
            request_body["max_completion_tokens"] = max_completion_tokens
        cache_key = hashlib.sha256(json.dumps(request_body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        parse_errors = (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError)
        invalid_cache_paths: list[Path] = []

        def invalid_path() -> Path:
            assert cache_path is not None
            return cache_path.with_name(f"{cache_path.stem}.invalid-{time.time_ns()}.json")

        if cache_path and cache_path.exists():
            try:
                response = json.loads(cache_path.read_text(encoding="utf-8"))
                self._validate_thinking_mode(response)
                return self._parse(response), {
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "usage": response.get("usage"),
                    "thinking_mode_verified": thinking_type == "disabled",
                }
            except parse_errors:
                quarantined = invalid_path()
                cache_path.replace(quarantined)
                invalid_cache_paths.append(quarantined)
        transport_failures = 0
        invalid_response_failures = 0
        # Even when transport retries are disabled, allow one repair request for
        # a syntactically invalid JSON response.  This is distinct from retrying
        # a network failure and prevents a bad response from poisoning the cache.
        invalid_response_retries = max(1, self.max_retries)
        while True:
            backoff = getattr(self, 'service_backoff', None)
            if backoff:
                backoff.before(cache_key)
            self._wait_for_request_slot()
            try:
                response = self._client.chat.completions.create(**request_body)
            except Exception as exc:
                if backoff:
                    unavailable = getattr(exc, 'status_code', None) == 503
                    backoff.result(cache_key, unavailable)
                    if unavailable:
                        continue
                if transport_failures >= self.max_retries:
                    raise
                transport_failures += 1
                time.sleep(min(2 ** (transport_failures - 1), 30))
                continue
            if backoff:
                backoff.result(cache_key, False)
            raw = response.model_dump()
            self._validate_thinking_mode(raw)
            try:
                value = self._parse(raw)
            except parse_errors:
                if cache_path:
                    quarantined = invalid_path()
                    quarantined.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
                    invalid_cache_paths.append(quarantined)
                if invalid_response_failures >= invalid_response_retries:
                    raise
                invalid_response_failures += 1
                time.sleep(min(2 ** (invalid_response_failures - 1), 30))
                continue
            if cache_path:
                cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            metadata = {
                "cache_hit": False,
                "cache_key": cache_key,
                "usage": raw.get("usage"),
                "thinking_mode_verified": thinking_type == "disabled",
            }
            if invalid_cache_paths:
                resolved = [str(path.resolve()) for path in invalid_cache_paths]
                metadata["invalid_cache_path"] = resolved[0]
                metadata["invalid_cache_paths"] = resolved
            return value, metadata

    def _validate_thinking_mode(self, response: dict[str, Any]) -> None:
        if getattr(self, "thinking_type", None) != "disabled":
            return
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        details = ((response.get("usage") or {}).get("completion_tokens_details") or {})
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning or (isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0):
            raise ThinkingModeMismatchError(
                "Provider returned reasoning although No Thinking was requested: "
                f"reasoning_tokens={reasoning_tokens!r}"
            )

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


class RoundRobinJSONClient:
    """Distribute concurrent JSON completions across independent clients.

    Stage runners still own scheduling and use a single output writer.  This
    object only chooses which independently rate-limited API client serves each
    request, so multiple credentials never create competing JSONL writers.
    """

    def __init__(self, clients: list[OpenAICompatibleClient]) -> None:
        if not clients:
            raise ValueError("RoundRobinJSONClient requires at least one client")
        models = {client.model for client in clients}
        if len(models) != 1:
            raise ValueError("All pooled clients must use the same model")
        self.clients = tuple(clients)
        self.model = clients[0].model
        self.config = clients[0].config
        self.key_count = len(clients)
        self._next_client = 0
        self._pool_lock = threading.Lock()

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        client_index, client = self._take_client()
        value, metadata = client.complete_json(system, user)
        return value, {**metadata, "client_pool_index": client_index}

    def complete_json_schema(
        self, system: str, user: str, *, name: str, schema: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        client_index, client = self._take_client()
        value, metadata = client.complete_json_schema(
            system, user, name=name, schema=schema
        )
        return value, {**metadata, "client_pool_index": client_index}

    def _take_client(self) -> tuple[int, OpenAICompatibleClient]:
        with self._pool_lock:
            client_index = self._next_client
            self._next_client = (self._next_client + 1) % len(self.clients)
        return client_index, self.clients[client_index]


def build_multi_key_client(
    config: ProviderConfig,
    *,
    api_keys_env: str = "TRACE_ANALYSIS_API_KEYS",
    include_primary_key: bool = True,
    model: str | None = None,
    cache_dir: Path | None = None,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    thinking_type: str | None = None,
    max_completion_tokens: int | None = None,
) -> RoundRobinJSONClient:
    """Build a secret-safe pool from a comma/newline-separated environment value."""
    load_project_env()
    values: list[str] = []
    if include_primary_key:
        primary = os.environ.get(config.api_key_env, "").strip()
        if primary:
            values.append(primary)
    raw = os.environ.get(api_keys_env, "")
    for value in re.split(r"[,\n]", raw):
        key = value.strip()
        if key and key not in values:
            values.append(key)
    if not values:
        raise ValueError(
            f"No API keys found in {config.api_key_env} or {api_keys_env}"
        )
    clients = [
        OpenAICompatibleClient(
            config,
            model=model,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            thinking_type=thinking_type,
            max_completion_tokens=max_completion_tokens,
            api_key=key,
        )
        for key in values
    ]
    return RoundRobinJSONClient(clients)
