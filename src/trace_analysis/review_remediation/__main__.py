from __future__ import annotations

import argparse
import json
from pathlib import Path

from .planner import plan_review_remediation
from .workflow import (
    approve_queue_items, build_validated_candidate_runs,
    execute_prepared_reanalysis, prepare_approved_reanalysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review-driven remediation")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="classify reviews and scan impact")
    plan.add_argument("--reviews", type=Path, required=True)
    plan.add_argument("--analysis-root", type=Path, required=True)
    plan.add_argument("--preprocessed-root", type=Path, required=True)
    plan.add_argument("--publication-root", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--cycle-id")
    approve = commands.add_parser("approve", help="approve selected queue items")
    approve.add_argument("--cycle-dir", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--reason", required=True)
    approve.add_argument("--queue-id", action="append")
    approve.add_argument("--issue-code", action="append")
    approve.add_argument("--limit", type=int)
    approve.add_argument("--all", action="store_true")
    prepare = commands.add_parser(
        "prepare", help="materialize approved Stage-03 repair jobs"
    )
    prepare.add_argument("--cycle-dir", type=Path, required=True)
    prepare.add_argument("--analysis-root", type=Path, required=True)
    prepare.add_argument("--preprocessed-root", type=Path, required=True)
    execute = commands.add_parser("execute", help="run approved Stage-03 jobs")
    execute.add_argument("--cycle-dir", type=Path, required=True)
    execute.add_argument("--provider", required=True)
    execute.add_argument("--provider-config", type=Path, required=True)
    execute.add_argument("--cache-dir", type=Path, required=True)
    execute.add_argument("--model")
    execute.add_argument("--workers", type=int, default=4)
    execute.add_argument("--direct-max-chars", type=int, default=120_000)
    execute.add_argument("--request-max-chars", type=int, default=800_000)
    execute.add_argument("--timeout-seconds", type=int)
    execute.add_argument("--max-retries", type=int)
    execute.add_argument(
        "--thinking-type", choices=("enabled", "disabled"), default="disabled"
    )
    execute.add_argument("--max-completion-tokens", type=int)
    validate = commands.add_parser(
        "validate", help="merge completed repairs into non-published candidate Runs"
    )
    validate.add_argument("--cycle-dir", type=Path, required=True)
    validate.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan_review_remediation(
            reviews_path=args.reviews,
            analysis_root=args.analysis_root,
            preprocessed_root=args.preprocessed_root,
            publication_root=args.publication_root,
            output_root=args.output_root,
            cycle_id=args.cycle_id,
        )
    elif args.command == "approve":
        if not args.all and not args.queue_id and not args.issue_code:
            parser.error("approve requires --queue-id, --issue-code, or explicit --all")
        result = approve_queue_items(
            cycle_dir=args.cycle_dir,
            reviewer_name=args.reviewer,
            reason=args.reason,
            queue_ids=set(args.queue_id) if args.queue_id else None,
            issue_codes=set(args.issue_code) if args.issue_code else None,
            limit=args.limit,
        )
    elif args.command == "prepare":
        result = prepare_approved_reanalysis(
            cycle_dir=args.cycle_dir,
            analysis_root=args.analysis_root,
            preprocessed_root=args.preprocessed_root,
        )
    elif args.command == "execute":
        result = execute_prepared_reanalysis(
            cycle_dir=args.cycle_dir,
            provider=args.provider,
            provider_config=args.provider_config,
            cache_dir=args.cache_dir,
            model=args.model,
            workers=args.workers,
            direct_max_chars=args.direct_max_chars,
            request_max_chars=args.request_max_chars,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            thinking_type=args.thinking_type,
            max_completion_tokens=args.max_completion_tokens,
        )
    else:
        result = build_validated_candidate_runs(
            cycle_dir=args.cycle_dir, analysis_root=args.analysis_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
