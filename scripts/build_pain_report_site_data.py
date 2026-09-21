#!/usr/bin/env python3
"""Build the public-safe pain analysis data used by the static research site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_analysis.web import publish_pain_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--user-turns", required=True, type=Path)
    parser.add_argument("--unified-turns", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--site-dir", required=True, type=Path)
    args = parser.parse_args()
    result = publish_pain_report(
        args.analysis,
        args.user_turns,
        args.unified_turns,
        args.manifest,
        args.site_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
