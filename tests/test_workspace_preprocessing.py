from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from trace_analysis.preprocessing.workspace_v2 import run_workspace_preprocess


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_parts(directory: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(directory.glob("part-*.jsonl")):
        records.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


class WorkspacePreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.legacy = self.root / "preprocessed"
        self.dataset_id = "dataset_fixture"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _source(self, name: str, prompt: str) -> Path:
        path = self.root / "sources" / name
        _write_jsonl(path, [
            {
                "type": "user", "promptId": f"prompt-{name}", "uuid": f"u-{name}",
                "timestamp": "2026-01-01T00:00:01Z", "sessionId": f"session-{name}",
                "message": {"role": "user", "content": prompt},
            },
            {
                "type": "assistant", "uuid": f"a-{name}",
                "timestamp": "2026-01-01T00:00:02Z", "sessionId": f"session-{name}",
                "message": {
                    "id": f"message-{name}", "role": "assistant", "model": "model-a",
                    "provider": "provider-a", "content": [{"type": "text", "text": "完成"}],
                    "stop_reason": "end_turn",
                },
            },
        ])
        return path

    def test_dual_write_source_references_registry_and_idempotence(self) -> None:
        source = self._source("one.jsonl", "请分析这个问题")

        first = run_workspace_preprocess(
            [str(source)], self.workspace, self.legacy, None, None,
            dataset_id=self.dataset_id, shard_size_bytes=1,
        )
        batch_id = first["batch_id"]
        batch_dir = Path(first["workspace_v2"]["output_path"])
        preprocess = batch_dir / "01_preprocess"
        source_records = _read_parts(preprocess / "source_records")
        turns = _read_parts(preprocess / "unified_turns")
        lineage = _read_parts(preprocess / "lineage_edges")

        self.assertEqual(len(source_records), 2)
        self.assertEqual(len(turns), 1)
        self.assertEqual(first["workspace_v2"]["event_count"], len(lineage))
        self.assertGreater(len(list((preprocess / "source_records").glob("part-*.jsonl"))), 1)
        source_index = {str(row["source_record_id"]): row for row in source_records}
        for event in turns[0]["events"]:
            self.assertNotIn("data", event)
            self.assertNotIn("raw", event)
            source_ref = event["source_ref"]
            source_record = source_index[str(source_ref["source_record_id"])]
            projection_index = int(str(source_ref["json_pointer"]).rsplit("/", 1)[1])
            projection = source_record["normalized_events"][projection_index]
            self.assertEqual(projection["event_id"], event["event_id"])
        context = turns[0]["execution_context"]
        self.assertEqual(context["harness"]["name"], "claude_code")
        self.assertEqual(context["model_calls"][0]["provider"], "provider-a")

        self.assertEqual(len(list((self.workspace / "registry" / "datasets").glob("*.json"))), 1)
        self.assertEqual(len(list((self.workspace / "registry" / "batches").glob("*.json"))), 1)
        self.assertEqual(len(list((self.workspace / "registry" / "runs").glob("*.json"))), 1)
        self.assertEqual(len(list((self.workspace / "registry" / "products").glob("*.json"))), 4)
        dataset_manifest = json.loads(
            (self.workspace / "datasets" / self.dataset_id / "dataset_manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(dataset_manifest["batch_count"], 1)

        second = run_workspace_preprocess(
            [str(source)], self.workspace, self.legacy, None, None,
            dataset_id=self.dataset_id, shard_size_bytes=1,
        )
        self.assertTrue(second["legacy"]["deduplicated"])
        self.assertTrue(second["workspace_v2"]["deduplicated"])
        self.assertEqual(second["batch_id"], batch_id)

    def test_incremental_batches_and_target_profile_status(self) -> None:
        first_source = self._source("first.jsonl", "第一个需求")
        second_source = self._source("second.jsonl", "第二个需求")
        first = run_workspace_preprocess(
            [str(first_source)], self.workspace, self.legacy, None, None,
            dataset_id=self.dataset_id,
        )
        second = run_workspace_preprocess(
            [str(second_source)], self.workspace, self.legacy, None, None,
            dataset_id=self.dataset_id,
        )
        self.assertNotEqual(first["batch_id"], second["batch_id"])
        dataset_manifest = json.loads(
            (self.workspace / "datasets" / self.dataset_id / "dataset_manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(dataset_manifest["batch_count"], 2)

        self.assertEqual(dataset_manifest["batch_count"], 2)

    def test_collector_zip_is_streamed_into_workspace_v2(self) -> None:
        source = self.root / "collector.zip"
        base = {
            "trace_schema_version": "v0.2.0", "session_id": "session-zip",
            "trace_id": None, "timestamp": "2026-01-01T00:00:00Z",
            "project_hmac": {}, "project_identity_source": "cwd",
            "source": {"adapter": "codex_rollout_jsonl"},
        }
        records = [
            {**base, "event_id": "event-1", "turn_id": "turn-1", "sequence": 1,
             "event_type": "event_msg", "payload": {"type": "event_msg", "payload": {
                 "type": "task_started", "turn_id": "turn-1"}}},
            {**base, "event_id": "event-2", "turn_id": "turn-1", "sequence": 2,
             "event_type": "event_msg", "payload": {"type": "event_msg", "payload": {
                 "type": "task_complete", "turn_id": "turn-1"}}},
        ]
        member = "collection/raw_trace_events.jsonl"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member, "".join(json.dumps(row) + "\n" for row in records))

        result = run_workspace_preprocess(
            [str(source)], self.workspace, self.legacy, None, None,
            dataset_id=self.dataset_id,
        )
        preprocess = Path(result["workspace_v2"]["output_path"]) / "01_preprocess"
        source_records = _read_parts(preprocess / "source_records")

        self.assertEqual(len(source_records), 2)
        self.assertEqual(source_records[0]["source_locator"]["source_member"], member)
        self.assertTrue(str(source_records[0]["source_locator"]["source_uri"]).startswith("zip://"))
        self.assertEqual(result["workspace_v2"]["event_count"], 2)


if __name__ == "__main__":
    unittest.main()
