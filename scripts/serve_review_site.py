#!/usr/bin/env python3
"""Serve the local report with persistent human-review API endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_analysis.web import create_review_server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reviewer-user-id", default="local-reviewer")
    parser.add_argument("--reviewer-email", default="local-reviewer@localhost")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    server = create_review_server(
        host=args.host,
        port=args.port,
        site_dir=args.site_dir,
        database_path=args.database,
        export_path=args.output,
        manifest_path=args.manifest,
        reviewer_user_id=args.reviewer_user_id,
        reviewer_email=args.reviewer_email,
    )
    host, port = server.server_address[:2]
    print(f"Review site: http://{host}:{port}/")
    print(f"Review JSONL: {server.export_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
