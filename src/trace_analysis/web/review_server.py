from __future__ import annotations

import json
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .review_store import ReviewConflictError, ReviewValidationError, SQLiteReviewStore
from .review_sync import sync_reviews_to_jsonl


API_PATH = "/api/pain-reviews"
MAX_REQUEST_BYTES = 64 * 1024


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_published_case_catalog(site_dir: Path) -> dict[str, dict[str, str]]:
    """Load canonical case identities from the public pain-report indexes."""
    root = Path(site_dir) / "data" / "pain-report"
    catalog_path = root / "catalog.json"
    sources: list[tuple[Path, str, str]] = []
    if catalog_path.is_file():
        catalog = _read_json(catalog_path)
        for batch in catalog.get("batches") or []:
            sources.append((
                root / str(batch["case_index"]),
                str(batch.get("batch_id") or ""),
                str(batch.get("active_run_id") or ""),
            ))
    else:
        metadata = _read_json(root / "metadata.json")
        sources.append((
            root / "case-index.json",
            str(metadata.get("batch_id") or ""),
            str(metadata.get("analysis_run_id") or ""),
        ))

    cases: dict[str, dict[str, str]] = {}
    for index_path, fallback_batch_id, fallback_run_id in sources:
        for item in _read_json(index_path):
            case_id = str(item.get("case_id") or "").strip()
            if not case_id:
                continue
            canonical = {
                "case_id": case_id,
                "batch_id": str(item.get("batch_id") or fallback_batch_id),
                "analysis_run_id": str(item.get("analysis_run_id") or fallback_run_id),
                "trace_id": str(item.get("trace_id") or ""),
                "episode_id": str(item.get("episode_id") or ""),
            }
            if not all(canonical.values()):
                raise ValueError(f"Published case has incomplete identity: {case_id}")
            if case_id in cases and cases[case_id] != canonical:
                raise ValueError(f"Published case_id is not globally unique: {case_id}")
            cases[case_id] = canonical
    return cases


def _browser_review(record: dict[str, Any]) -> dict[str, Any]:
    """Return review state needed by the UI without exposing reviewer identity."""
    return {
        key: record.get(key)
        for key in (
            "review_id",
            "case_id",
            "decision",
            "attribution_override",
            "note",
            "revision",
            "supersedes_review_id",
            "created_at",
        )
    }


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    server: "ReviewHTTPServer"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _request_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ReviewValidationError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ReviewValidationError("Invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ReviewValidationError("Review request has an invalid size")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewValidationError("Request body must be a JSON object") from exc
        if not isinstance(value, dict):
            raise ReviewValidationError("Request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != API_PATH:
            super().do_GET()
            return
        reviews = [_browser_review(item) for item in self.server.review_store.latest_reviews()]
        self._send_json(HTTPStatus.OK, {
            "schema_version": "human-review-api-v1",
            "reviews": reviews,
            "can_review": True,
            "sign_in_url": None,
        })

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != API_PATH:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            request = self._request_json()
            case_id = str(request.get("case_id") or "").strip()
            canonical = self.server.case_catalog.get(case_id)
            if canonical is None:
                self._send_json(HTTPStatus.NOT_FOUND, {
                    "error": "unknown_case",
                    "message": "The case is not present in the published report.",
                })
                return
            expected_revision = request.get("expected_revision")
            if expected_revision is not None and not isinstance(expected_revision, int):
                raise ReviewValidationError("expected_revision must be an integer")
            with self.server.review_lock:
                record = self.server.review_store.save_review(
                    **canonical,
                    decision=str(request.get("decision") or ""),
                    note=str(request.get("note") or ""),
                    attribution_override=(
                        str(request["attribution_override"]).strip()
                        if request.get("attribution_override") else None
                    ),
                    reviewer_user_id=self.server.reviewer_user_id,
                    reviewer_email=self.server.reviewer_email,
                    expected_revision=expected_revision,
                )
                sync = sync_reviews_to_jsonl(
                    self.server.review_store,
                    self.server.export_path,
                    self.server.manifest_path,
                )
            self._send_json(HTTPStatus.CREATED, {
                "schema_version": "human-review-api-v1",
                "review": _browser_review(record),
                "sync": sync,
            })
        except ReviewValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {
                "error": "invalid_review",
                "message": str(exc),
            })
        except ReviewConflictError as exc:
            self._send_json(HTTPStatus.CONFLICT, {
                "error": "review_conflict",
                "message": str(exc),
            })

    def log_message(self, format: str, *args: Any) -> None:
        if not self.server.quiet:
            super().log_message(format, *args)


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        site_dir: Path,
        database_path: Path,
        export_path: Path,
        manifest_path: Path,
        reviewer_user_id: str,
        reviewer_email: str,
        quiet: bool = False,
    ):
        resolved_site = Path(site_dir).expanduser().resolve()
        if not (resolved_site / "index.html").is_file():
            raise FileNotFoundError(resolved_site / "index.html")
        self.review_store = SQLiteReviewStore(database_path)
        self.case_catalog = load_published_case_catalog(resolved_site)
        self.export_path = Path(export_path).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.reviewer_user_id = reviewer_user_id
        self.reviewer_email = reviewer_email
        self.review_lock = threading.Lock()
        self.quiet = quiet
        handler = partial(ReviewRequestHandler, directory=str(resolved_site))
        super().__init__(server_address, handler)


def create_review_server(
    *,
    host: str,
    port: int,
    site_dir: Path,
    database_path: Path,
    export_path: Path,
    manifest_path: Path,
    reviewer_user_id: str,
    reviewer_email: str,
    quiet: bool = False,
) -> ReviewHTTPServer:
    return ReviewHTTPServer(
        (host, port),
        site_dir=site_dir,
        database_path=database_path,
        export_path=export_path,
        manifest_path=manifest_path,
        reviewer_user_id=reviewer_user_id,
        reviewer_email=reviewer_email,
        quiet=quiet,
    )
