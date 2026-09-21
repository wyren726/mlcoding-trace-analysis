from __future__ import annotations

import json

from trace_analysis.web.site_review_import import import_site_review_export


def _event(sequence: int, revision: int, decision: str) -> dict:
    return {
        "schema_version": "site-human-review-event-v1",
        "event_sequence": sequence,
        "review_id": f"review_{sequence}",
        "case_id": "case_a",
        "batch_id": "batch_a",
        "analysis_run_id": "run_a",
        "trace_id": "trace_a",
        "episode_id": "episode_a",
        "decision": decision,
        "attribution_override": None,
        "note": f"note {sequence}",
        "reviewer_name": "reviewer",
        "revision": revision,
        "supersedes_review_id": f"review_{sequence - 1}" if sequence > 1 else None,
        "created_at": f"2026-08-24T00:00:0{sequence}Z",
    }


def test_import_site_review_export_keeps_history_and_latest_decision(tmp_path):
    source = tmp_path / "export.jsonl"
    source.write_text(
        "\n".join(json.dumps(value) for value in (
            _event(1, 1, "review"), _event(2, 2, "excluded"),
        )) + "\n",
        encoding="utf-8",
    )
    result = import_site_review_export(source, tmp_path / "mirror")
    assert result["event_count"] == 2
    assert result["latest_decision_count"] == 1
    events = (tmp_path / "mirror" / "site_review_events.jsonl").read_text(encoding="utf-8").splitlines()
    decisions = (tmp_path / "mirror" / "human_review_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 2
    assert json.loads(decisions[0])["decision"] == "excluded"
    assert json.loads(decisions[0])["revision"] == 2
