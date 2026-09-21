#!/usr/bin/env python3
"""Synchronize latest human review decisions into an auditable JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_analysis.web import SQLiteReviewStore, sync_reviews_to_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = sync_reviews_to_jsonl(
        SQLiteReviewStore(args.database), args.output, args.manifest
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
