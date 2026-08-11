from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .preprocessing.pipeline import run_preprocess
from .features.basic import run_basic_features
from .features.incremental import consolidate_feature_set
from .features.custom import (TASKS, audit_demand_grounding, configured_client, repair_requirement_sources,
                              run_semantic_features, write_dry_run)
from .features.model_api import load_provider
from .orchestration import run_update
from .analysis.extensions import EXTENSIONS, run_extension
from .analysis.core import (build_candidate_core, publish_reviewed_core, review_suggestions_with_llm,
                            organize_hierarchy, suggest_reviews, induce_taxonomy,
                            induce_taxonomy_scalable, refine_taxonomy_labels,
                            filter_ml_llm_coding_scope, build_dynamic_taxonomy)
from .env import load_project_env
from .reporting import generate_capability_report
from .artifact_sync import sync_artifacts
from .registries import KINDS, build_artifact_manifest, rebuild_registry_csv


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="trace_analysis")
    commands = root.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser("preprocess", help="Normalize raw trace files")
    preprocess.add_argument("--source", nargs="+", required=True, help="Files and/or directories")
    preprocess.add_argument("--output-root", default="preprocessed")
    preprocess.add_argument("--adapter", choices=("collector_events", "sls_proxy", "session_jsonl"))
    preprocess.add_argument("--limit", type=int, help="Smoke-test record limit")
    features = commands.add_parser("features", help="Extract reusable features")
    features.add_argument("--batch", nargs="+", required=True,
                          help="One or more preprocessed batch directories")
    features.add_argument("--feature-set", default="basic_turn_features", choices=("basic_turn_features", *TASKS))
    features.add_argument("--output-root", default="features")
    features.add_argument("--provider", default="pjlab_proxy")
    features.add_argument("--model")
    features.add_argument("--provider-config", default="configs/providers.toml")
    features.add_argument("--cache-dir", default=".cache/llm")
    features.add_argument("--limit", type=int)
    features.add_argument("--max-input-chars", type=int, default=24000,
                          help="Maximum redacted Turn characters sent to the model")
    features.add_argument("--timeout-seconds", type=int, help="Override provider request timeout")
    features.add_argument("--max-retries", type=int, help="Override provider retry count")
    features.add_argument("--workers", type=int, default=8,
                          help="Concurrent semantic API requests sharing the provider rate limiter")
    features.add_argument("--evidence-repair", action="store_true",
                          help="Retry missing records with strict verbatim-evidence instructions")
    features.add_argument("--dry-run", action="store_true", help="Write model requests without calling the API")
    consolidate = commands.add_parser("consolidate-features", help="Consolidate incremental feature runs by input batch")
    consolidate.add_argument("--batch", nargs="+", required=True, help="Input batch IDs")
    consolidate.add_argument("--feature-set", required=True)
    consolidate.add_argument("--feature-version", required=True)
    consolidate.add_argument("--output-root", default="features")
    consolidate.add_argument("--provider", default="")
    consolidate.add_argument("--model", default="")
    analyze_core = commands.add_parser("analyze-core", help="Build a reviewable candidate capability snapshot")
    analyze_core.add_argument("--feature-file", nargs="+", required=True)
    analyze_core.add_argument("--output-root", default="analysis")
    analyze_core.add_argument("--taxonomy-version")
    analyze_core.add_argument("--existing-taxonomy", help="Previous capability_taxonomy.json used for matching")
    induce = commands.add_parser("induce-taxonomy", help="Induce a validated bottom-up capability hierarchy")
    induce.add_argument("--draft-core", required=True)
    induce.add_argument("--output-root", default="analysis")
    induce.add_argument("--taxonomy-version")
    induce.add_argument("--provider", default="pjlab_proxy")
    induce.add_argument("--model")
    induce.add_argument("--provider-config", default="configs/providers.toml")
    induce.add_argument("--cache-dir", default=".cache/llm")
    induce.add_argument("--timeout-seconds", type=int)
    induce.add_argument("--max-retries", type=int)
    scalable = commands.add_parser("induce-taxonomy-scalable", help="Induce a large taxonomy in resumable chunks")
    scalable.add_argument("--draft-core", required=True)
    scalable.add_argument("--output-root", default="analysis")
    scalable.add_argument("--taxonomy-version")
    scalable.add_argument("--provider", default="pjlab_proxy")
    scalable.add_argument("--model")
    scalable.add_argument("--provider-config", default="configs/providers.toml")
    scalable.add_argument("--cache-dir", default=".cache/llm")
    scalable.add_argument("--timeout-seconds", type=int)
    scalable.add_argument("--max-retries", type=int)
    scalable.add_argument("--workers", type=int, default=16)
    scalable.add_argument("--target-unit-size", type=int, default=30)
    scalable.add_argument("--max-unit-size", type=int, default=40)
    refine = commands.add_parser("refine-taxonomy-labels", help="Refine middle labels and top-level capability domains")
    refine.add_argument("--core", required=True)
    refine.add_argument("--output-root", default="analysis")
    refine.add_argument("--taxonomy-version", required=True)
    refine.add_argument("--provider", default="pjlab_proxy")
    refine.add_argument("--model")
    refine.add_argument("--provider-config", default="configs/providers.toml")
    refine.add_argument("--cache-dir", default=".cache/llm")
    refine.add_argument("--timeout-seconds", type=int)
    refine.add_argument("--max-retries", type=int)
    refine.add_argument("--workers", type=int, default=16)
    dynamic = commands.add_parser("build-dynamic-taxonomy",
                                  help="Globally reassign leaves and recursively build a variable-depth taxonomy")
    dynamic.add_argument("--core", required=True)
    dynamic.add_argument("--output-root", default="analysis")
    dynamic.add_argument("--taxonomy-version", required=True)
    dynamic.add_argument("--provider", default="pjlab_proxy")
    dynamic.add_argument("--model")
    dynamic.add_argument("--provider-config", default="configs/providers.toml")
    dynamic.add_argument("--cache-dir", default=".cache/llm")
    dynamic.add_argument("--timeout-seconds", type=int)
    dynamic.add_argument("--max-retries", type=int)
    dynamic.add_argument("--max-children", type=int, default=8)
    dynamic.add_argument("--workers", type=int, default=16)
    dynamic.add_argument("--batch-size", type=int, default=10)
    dynamic.add_argument("--negotiation-feature-file", nargs="+",
                         help="Optional requirement_negotiation feature JSONL files")
    scope = commands.add_parser("filter-ml-llm-scope", help="Keep only scientific ML/LLM coding demand instances")
    scope.add_argument("--feature-file", nargs="+", required=True)
    scope.add_argument("--output-dir", required=True)
    scope.add_argument("--provider", default="pjlab_proxy")
    scope.add_argument("--model")
    scope.add_argument("--provider-config", default="configs/providers.toml")
    scope.add_argument("--cache-dir", default=".cache/llm")
    scope.add_argument("--timeout-seconds", type=int)
    scope.add_argument("--max-retries", type=int)
    scope.add_argument("--workers", type=int, default=16)
    scope.add_argument("--batch-size", type=int, default=25)
    scope.add_argument("--audit-included", action="store_true",
                       help="Strictly recheck existing included decisions and remove false positives")
    scope.add_argument("--seed-decisions", help="Reuse decisions for unchanged instance IDs")
    repair = commands.add_parser("repair-requirement-sources",
                                 help="Re-extract demand Turns whose evidence is not from user messages")
    repair.add_argument("--batch", required=True)
    repair.add_argument("--feature-file", required=True)
    repair.add_argument("--output-dir", required=True)
    repair.add_argument("--provider", default="pjlab_proxy")
    repair.add_argument("--model")
    repair.add_argument("--provider-config", default="configs/providers.toml")
    repair.add_argument("--cache-dir", default=".cache/llm")
    repair.add_argument("--timeout-seconds", type=int)
    repair.add_argument("--max-retries", type=int)
    repair.add_argument("--workers", type=int, default=16)
    repair.add_argument("--max-input-chars", type=int, default=24000)
    grounding = commands.add_parser("audit-demand-grounding",
                                    help="Verify that normalized demands are entailed by user quotes")
    grounding.add_argument("--feature-file", nargs="+", required=True)
    grounding.add_argument("--output-dir", required=True)
    grounding.add_argument("--provider", default="pjlab_proxy")
    grounding.add_argument("--model")
    grounding.add_argument("--provider-config", default="configs/providers.toml")
    grounding.add_argument("--cache-dir", default=".cache/llm")
    grounding.add_argument("--timeout-seconds", type=int)
    grounding.add_argument("--max-retries", type=int)
    grounding.add_argument("--workers", type=int, default=16)
    grounding.add_argument("--batch-size", type=int, default=25)
    report = commands.add_parser("report-capabilities", help="Generate a complete Markdown capability report")
    report.add_argument("--core", required=True)
    report.add_argument("--batch", nargs="+", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--scope-summary", help="Optional ML/LLM coding scope_summary.json")
    report.add_argument("--grounding-summary", help="Optional demand grounding_summary.json")
    review_core = commands.add_parser("review-core", help="Apply human decisions and publish a taxonomy version")
    review_core.add_argument("--draft-core", required=True)
    review_core.add_argument("--decisions", required=True)
    review_core.add_argument("--taxonomy-version", required=True)
    review_core.add_argument("--output-root", default="analysis")
    suggest_review = commands.add_parser("suggest-review", help="Suggest possible merges and splits for review")
    suggest_review.add_argument("--draft-core", required=True)
    suggest_review.add_argument("--output", required=True)
    suggest_review.add_argument("--merge-threshold", type=float, default=0.65)
    suggest_review.add_argument("--split-similarity-threshold", type=float, default=0.32)
    suggest_review.add_argument("--split-min-cases", type=int, default=4)
    suggest_review.add_argument("--split-min-cluster-size", type=int, default=2)
    llm_review = commands.add_parser("review-suggestions-llm", help="LLM review of deterministic candidates")
    llm_review.add_argument("--draft-core", required=True)
    llm_review.add_argument("--suggestions", required=True)
    llm_review.add_argument("--output-root", default="analysis")
    llm_review.add_argument("--provider", default="pjlab_proxy")
    llm_review.add_argument("--model")
    llm_review.add_argument("--provider-config", default="configs/providers.toml")
    llm_review.add_argument("--cache-dir", default=".cache/llm")
    llm_review.add_argument("--limit", type=int)
    llm_review.add_argument("--max-input-chars", type=int, default=24000)
    llm_review.add_argument("--timeout-seconds", type=int)
    llm_review.add_argument("--max-retries", type=int)
    llm_review.add_argument("--dry-run", action="store_true")
    hierarchy = commands.add_parser("organize-hierarchy", help="Apply reviewed hierarchy operations")
    hierarchy.add_argument("--core", required=True)
    hierarchy.add_argument("--operations", required=True)
    hierarchy.add_argument("--taxonomy-version", required=True)
    hierarchy.add_argument("--output-root", default="analysis")
    update = commands.add_parser("update", help="Run or resume the incremental pipeline")
    update.add_argument("--config", default="configs/update.json")
    update.add_argument("--source", nargs="+")
    update.add_argument("--resume")
    update.add_argument("--review-decisions")
    update.add_argument("--hierarchy-operations")
    extension = commands.add_parser("analyze-extension", help="Run an independent analysis extension")
    extension.add_argument("--name", required=True, choices=tuple(EXTENSIONS))
    extension.add_argument("--core", required=True)
    extension.add_argument("--feature-file", nargs="*", default=[])
    extension.add_argument("--output-dir", required=True)
    sync = commands.add_parser("sync", help="Inspect or synchronize immutable analysis artifacts over SSH")
    sync.add_argument("action", choices=("status", "push", "pull"))
    sync.add_argument("--config", default="configs/sync.toml")
    sync.add_argument("--project-root", default=".")
    sync.add_argument("--dry-run", action="store_true")
    registry = commands.add_parser("registry", help="Rebuild CSV views from immutable registry records")
    registry.add_argument("action", choices=("rebuild", "manifest"))
    registry.add_argument("--kind", choices=tuple(sorted(KINDS)))
    registry.add_argument("--output", required=True)
    registry.add_argument("--project-root", default=".")
    return root


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    args = parser().parse_args(argv)
    if args.command == "preprocess":
        result = run_preprocess(args.source, Path(args.output_root), args.adapter, args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "consolidate-features":
        result = consolidate_feature_set(
            Path(args.output_root), args.feature_set, args.feature_version,
            args.batch, args.provider, args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze-core":
        result = build_candidate_core([Path(path) for path in args.feature_file], Path(args.output_root),
                                      args.taxonomy_version,
                                      Path(args.existing_taxonomy) if args.existing_taxonomy else None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "induce-taxonomy":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = induce_taxonomy(Path(args.draft_core), Path(args.output_root), client, args.provider,
                                 args.taxonomy_version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "induce-taxonomy-scalable":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = induce_taxonomy_scalable(
            Path(args.draft_core), Path(args.output_root), client, args.provider,
            args.taxonomy_version, args.workers, args.target_unit_size, args.max_unit_size,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "refine-taxonomy-labels":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = refine_taxonomy_labels(Path(args.core), Path(args.output_root), client, args.provider,
                                        args.taxonomy_version, args.workers)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-dynamic-taxonomy":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = build_dynamic_taxonomy(Path(args.core), Path(args.output_root), client, args.provider,
                                        args.taxonomy_version, args.max_children, args.workers, args.batch_size,
                                        [Path(path) for path in args.negotiation_feature_file]
                                        if args.negotiation_feature_file else None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "filter-ml-llm-scope":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = filter_ml_llm_coding_scope(
            [Path(path) for path in args.feature_file], Path(args.output_dir), client, args.provider,
            args.workers, args.batch_size, args.audit_included,
            Path(args.seed_decisions) if args.seed_decisions else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "repair-requirement-sources":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = repair_requirement_sources(Path(args.batch), Path(args.feature_file), Path(args.output_dir),
                                            client, args.provider, args.workers, args.max_input_chars)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-demand-grounding":
        client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                   args.timeout_seconds, args.max_retries)
        result = audit_demand_grounding([Path(path) for path in args.feature_file], Path(args.output_dir),
                                        client, args.provider, args.workers, args.batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report-capabilities":
        result = generate_capability_report(
            Path(args.core), [Path(path) for path in args.batch], Path(args.output),
            Path(args.scope_summary) if args.scope_summary else None,
            Path(args.grounding_summary) if args.grounding_summary else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "review-core":
        result = publish_reviewed_core(Path(args.draft_core), Path(args.decisions), Path(args.output_root),
                                       args.taxonomy_version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "suggest-review":
        result = suggest_reviews(Path(args.draft_core), Path(args.output), args.merge_threshold,
                                 args.split_similarity_threshold, args.split_min_cases,
                                 args.split_min_cluster_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "review-suggestions-llm":
        if args.dry_run:
            provider_config = load_provider(Path(args.provider_config), args.provider)
            client = SimpleNamespace(model=args.model or provider_config.default_model)
        else:
            client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                       args.timeout_seconds, args.max_retries)
        result = review_suggestions_with_llm(Path(args.draft_core), Path(args.suggestions),
                                             Path(args.output_root), client, args.provider, args.dry_run,
                                             args.limit, args.max_input_chars)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "organize-hierarchy":
        result = organize_hierarchy(Path(args.core), Path(args.operations), Path(args.output_root),
                                    args.taxonomy_version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "update":
        result = run_update(Path(args.config), args.source, args.resume,
                            Path(args.review_decisions) if args.review_decisions else None,
                            Path(args.hierarchy_operations) if args.hierarchy_operations else None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "analyze-extension":
        result = run_extension(args.name, Path(args.core), [Path(path) for path in args.feature_file],
                               Path(args.output_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "sync":
        result = sync_artifacts(args.action, Path(args.config), Path(args.project_root), args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "success" else 1
    if args.command == "registry":
        if args.action == "rebuild":
            if not args.kind:
                raise SystemExit("registry rebuild requires --kind")
            result = rebuild_registry_csv(args.kind, Path(args.output), Path(args.project_root))
        else:
            result = build_artifact_manifest(Path(args.output), Path(args.project_root))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "features":
        batch_paths = [Path(path) for path in args.batch]
        client = None
        if args.feature_set != "basic_turn_features" and not args.dry_run:
            client = configured_client(Path(args.provider_config), args.provider, args.model, Path(args.cache_dir),
                                       args.timeout_seconds, args.max_retries)
        runs = []
        for batch_path in batch_paths:
            if args.feature_set == "basic_turn_features":
                runs.append(run_basic_features(batch_path, Path(args.output_root)))
            elif args.dry_run:
                runs.append(write_dry_run(batch_path, Path(args.output_root), args.feature_set, args.limit,
                                          args.max_input_chars))
            else:
                runs.append(run_semantic_features(batch_path, Path(args.output_root), args.feature_set,
                                                  client, args.provider, args.limit, args.max_input_chars,
                                                  args.workers, args.evidence_repair))
        result = runs[0] if len(runs) == 1 else {
            "feature_set": args.feature_set,
            "batch_count": len(batch_paths),
            "runs": runs,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"{args.command} is scaffolded but not implemented yet")
