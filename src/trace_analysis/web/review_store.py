from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_DECISIONS = {"confirmed", "excluded", "review"}


class ReviewValidationError(ValueError):
    pass


class ReviewConflictError(RuntimeError):
    pass


class SQLiteReviewStore:
    """Append-only local equivalent of the planned Sites D1 review store."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pain_review_events (
                    event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    analysis_run_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('confirmed', 'excluded', 'review')),
                    attribution_override TEXT,
                    note TEXT NOT NULL,
                    reviewer_user_id TEXT NOT NULL,
                    reviewer_email TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    supersedes_review_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_id, revision)
                );
                CREATE INDEX IF NOT EXISTS pain_review_events_case_revision
                    ON pain_review_events(case_id, revision DESC);
                """
            )

    def save_review(
        self,
        *,
        case_id: str,
        batch_id: str,
        analysis_run_id: str,
        trace_id: str,
        episode_id: str,
        decision: str,
        note: str,
        reviewer_user_id: str,
        reviewer_email: str,
        attribution_override: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        required = {
            "case_id": case_id,
            "batch_id": batch_id,
            "analysis_run_id": analysis_run_id,
            "trace_id": trace_id,
            "episode_id": episode_id,
            "note": note,
            "reviewer_user_id": reviewer_user_id,
            "reviewer_email": reviewer_email,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ReviewValidationError(f"Missing required review fields: {', '.join(missing)}")
        if decision not in ALLOWED_DECISIONS:
            raise ReviewValidationError(f"Unsupported review decision: {decision}")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                """
                SELECT review_id, revision FROM pain_review_events
                WHERE case_id = ? ORDER BY revision DESC LIMIT 1
                """,
                (case_id,),
            ).fetchone()
            current_revision = int(previous["revision"]) if previous else 0
            if expected_revision is not None and expected_revision != current_revision:
                raise ReviewConflictError(
                    f"Review changed concurrently: expected revision {expected_revision}, "
                    f"current revision is {current_revision}"
                )
            record = {
                "review_id": f"review_{uuid.uuid4().hex}",
                "case_id": case_id,
                "batch_id": batch_id,
                "analysis_run_id": analysis_run_id,
                "trace_id": trace_id,
                "episode_id": episode_id,
                "decision": decision,
                "attribution_override": attribution_override or None,
                "note": note.strip(),
                "reviewer_user_id": reviewer_user_id,
                "reviewer_email": reviewer_email,
                "revision": current_revision + 1,
                "supersedes_review_id": previous["review_id"] if previous else None,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            cursor = connection.execute(
                """
                INSERT INTO pain_review_events (
                    review_id, case_id, batch_id, analysis_run_id, trace_id,
                    episode_id, decision, attribution_override, note,
                    reviewer_user_id, reviewer_email, revision,
                    supersedes_review_id, created_at
                ) VALUES (
                    :review_id, :case_id, :batch_id, :analysis_run_id, :trace_id,
                    :episode_id, :decision, :attribution_override, :note,
                    :reviewer_user_id, :reviewer_email, :revision,
                    :supersedes_review_id, :created_at
                )
                """,
                record,
            )
            record["event_sequence"] = cursor.lastrowid
            connection.commit()
            return record

    def latest_reviews(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT pain_review_events.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY case_id ORDER BY revision DESC
                           ) AS latest_rank
                    FROM pain_review_events
                )
                WHERE latest_rank = 1
                ORDER BY batch_id, analysis_run_id, trace_id, episode_id
                """
            ).fetchall()
        return [
            {key: row[key] for key in row.keys() if key != "latest_rank"}
            for row in rows
        ]

    def latest_review(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pain_review_events
                WHERE case_id = ? ORDER BY revision DESC LIMIT 1
                """,
                (case_id,),
            ).fetchone()
        return dict(row) if row else None

    def event_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM pain_review_events").fetchone()
        return int(row["count"])

    def max_event_sequence(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(event_sequence), 0) AS value FROM pain_review_events"
            ).fetchone()
        return int(row["value"])
