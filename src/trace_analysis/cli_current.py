"""Command line interface for the current 01--05 pain-analysis pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocessing.pipeline import run_preprocess, run_preprocess_collector_zip
from .preprocessing.workspace_v2 import DEFAULT_SHARD_SIZE_BYTES, run_workspace_preprocess
from .pipeline import (
    build_pilot_sample, export_trace_analysis, pipeline_status,
    repair_trace_analysis, run_pipeline,
    write_pipeline_status,
)
from .registries import KINDS, build_artifact_manifest, rebuild_registry_csv
from .web.site_review_import import import_site_review_export
from .artifact_sync import sync_artifacts
from .pipeline.stage_06_trace_quality_labeling import run_labeling


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="trace_analysis")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preprocess")
    p.add_argument("--source", nargs="+", required=True); p.add_argument("--output-root", default="preprocessed")
    p.add_argument("--adapter", choices=("collector_events", "sls_proxy", "session_jsonl")); p.add_argument("--limit", type=int)
    p = commands.add_parser("preprocess-workspace")
    p.add_argument("--source", nargs="+", required=True); p.add_argument("--workspace-root", default="workspace")
    p.add_argument("--legacy-output-root", default="preprocessed"); p.add_argument("--dataset-id"); p.add_argument("--adapter", choices=("collector_events", "sls_proxy", "session_jsonl")); p.add_argument("--limit", type=int)
    p.add_argument("--shard-size-mb", type=int, default=DEFAULT_SHARD_SIZE_BYTES // (1024 * 1024))
    p = commands.add_parser("ingest-collector-zip"); p.add_argument("--source", required=True); p.add_argument("--output-root", default="preprocessed"); p.add_argument("--limit", type=int)
    p = commands.add_parser("pipeline"); actions = p.add_subparsers(dest="pipeline_action", required=True)
    p = actions.add_parser("run"); inp = p.add_mutually_exclusive_group(required=True); inp.add_argument("--source", nargs="+"); inp.add_argument("--batch")
    p.add_argument("--preprocessed-root", default="preprocessed"); p.add_argument("--analysis-root", default="workspace/analysis-runs"); p.add_argument("--publication-root", default="workspace/publications/pain-report"); p.add_argument("--adapter", choices=("collector_events", "sls_proxy", "session_jsonl")); p.add_argument("--mode", choices=("preview", "execute"), default="preview"); p.add_argument("--provider", default="pjlab_proxy"); p.add_argument("--model"); p.add_argument("--provider-config", default="configs/providers.toml"); p.add_argument("--cache-dir", default=".cache/trace-analysis"); p.add_argument("--workers", type=int, default=4); p.add_argument("--limit", type=int); p.add_argument("--preview-limit", type=int, default=5); p.add_argument("--trace-id-file"); p.add_argument("--direct-max-chars", type=int, default=120000); p.add_argument("--request-max-chars", type=int, default=800000); p.add_argument("--timeout-seconds", type=int); p.add_argument("--max-retries", type=int); p.add_argument("--max-completion-tokens", type=int); p.add_argument("--thinking", choices=("enabled", "disabled")); p.add_argument("--resume")
    p = actions.add_parser("repair"); p.add_argument("--batch", required=True); p.add_argument("--run-dir", required=True); p.add_argument("--provider", default="pjlab_proxy"); p.add_argument("--model"); p.add_argument("--provider-config", default="configs/providers.toml"); p.add_argument("--cache-dir", default=".cache/trace-analysis"); p.add_argument("--workers", type=int, default=2); p.add_argument("--trace-id-file"); p.add_argument("--direct-max-chars", type=int, default=120000); p.add_argument("--request-max-chars", type=int, default=800000); p.add_argument("--timeout-seconds", type=int); p.add_argument("--max-retries", type=int); p.add_argument("--max-completion-tokens", type=int); p.add_argument("--thinking", choices=("enabled", "disabled")); p.add_argument("--no-export", action="store_true")
    p = actions.add_parser("export"); p.add_argument("--run-dir", required=True); p.add_argument("--batch", required=True)
    p = actions.add_parser("status"); p.add_argument("--preprocessed-root", default="preprocessed"); p.add_argument("--analysis-root", default="workspace/analysis-runs"); p.add_argument("--only-incomplete", action="store_true"); p.add_argument("--output")
    p = actions.add_parser("sample"); p.add_argument("--batch", required=True); p.add_argument("--output", required=True); p.add_argument("--size", type=int, default=16); p.add_argument("--direct-max-chars", type=int, default=120000)
    p = actions.add_parser("label"); p.add_argument("--input", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--no-resume", action="store_true"); p.add_argument("--taxonomy", required=True, help="User-owned requirement/subrequirement CSV; atoms are derived locally"); p.add_argument("--provider", default="pjlab_proxy"); p.add_argument("--model"); p.add_argument("--provider-config", default="configs/providers.toml"); p.add_argument("--cache-dir", default=".cache/trace-analysis-stage06"); p.add_argument("--workers", type=int, default=1); p.add_argument("--limit", type=int); p.add_argument("--prepare-only", action="store_true"); p.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled"); p.add_argument("--max-completion-tokens", type=int, default=2048)
    p.add_argument("--source-lines", nargs="+", type=int, help="Only retry these original input line numbers")
    p = actions.add_parser("select", help="Select Stage06 traces using existing labels and evidence-linked audits")
    p.add_argument("--labels", required=True, help="New Stage06 labels JSONL; legacy v3/v5 are not silently converted")
    p.add_argument("--output-dir", required=True, help="New directory; existing runs are never overwritten")
    p.add_argument("--evidence", help="Evidence bundles JSONL, keyed by case_id")
    p.add_argument("--sources", help="Source/session metadata JSONL; defaults to sources.jsonl next to labels")
    p.add_argument("--audits", help="S1–S4 selection audit records JSONL")
    p.add_argument("--reviews", help="Existing per-claim review records JSONL")
    p.add_argument("--policy", help="Selection policy JSON; defaults to Stage06 config")
    p.add_argument("--mode", choices=("episode", "case_requirement"))
    p.add_argument("--rules-only", action="store_true", help="Skip model review; missing audits remain pending")
    p.add_argument("--provider", default="pjlab_proxy")
    p.add_argument("--model")
    p.add_argument("--provider-config", default="configs/providers.toml")
    p.add_argument("--cache-dir", default=".cache/trace-analysis-stage06-selection")
    p.add_argument("--max-input-chars", type=int, default=180000)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--workers", type=int, default=4)

    p = commands.add_parser("sync"); p.add_argument("action", choices=("status", "push", "pull")); p.add_argument("--config", default="configs/sync.toml"); p.add_argument("--project-root", default="."); p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("registry"); p.add_argument("action", choices=("rebuild", "manifest")); p.add_argument("--kind", choices=tuple(sorted(KINDS))); p.add_argument("--output", required=True); p.add_argument("--project-root", default=".")
    p = commands.add_parser("import-site-reviews"); p.add_argument("--input", required=True); p.add_argument("--output-dir", default="workspace/human-review/site-mirror")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "preprocess": result = run_preprocess(args.source, Path(args.output_root), args.adapter, args.limit)
    elif args.command == "preprocess-workspace": result = run_workspace_preprocess(args.source, Path(args.workspace_root), Path(args.legacy_output_root), args.adapter, args.limit, dataset_id=args.dataset_id, shard_size_bytes=args.shard_size_mb * 1024 * 1024)
    elif args.command == "ingest-collector-zip": result = run_preprocess_collector_zip(args.source, Path(args.output_root), args.limit)
    elif args.command == "import-site-reviews": result = import_site_review_export(Path(args.input), Path(args.output_dir))
    elif args.command == "sync": result = sync_artifacts(args.action, Path(args.config), Path(args.project_root), args.dry_run)
    elif args.command == "registry": result = rebuild_registry_csv(args.kind, Path(args.output), Path(args.project_root)) if args.action == "rebuild" else build_artifact_manifest(Path(args.output), Path(args.project_root))
    elif args.command == "pipeline":
        if args.pipeline_action == "run":
            ids = None
            if args.trace_id_file:
                ids = {str(json.loads(line).get("trace_id") if line.lstrip().startswith("{") else line.strip()) for line in Path(args.trace_id_file).read_text().splitlines() if line.strip()}
            result = run_pipeline(batch_dir=Path(args.batch) if args.batch else None, sources=args.source, preprocessed_root=Path(args.preprocessed_root), analysis_root=Path(args.analysis_root), adapter=args.adapter, mode=args.mode, provider=args.provider, model=args.model, provider_config=Path(args.provider_config), cache_dir=Path(args.cache_dir), workers=args.workers, limit=args.limit, preview_limit=args.preview_limit, direct_max_chars=args.direct_max_chars, request_max_chars=args.request_max_chars, timeout_seconds=args.timeout_seconds, max_retries=args.max_retries, resume_run_id=args.resume, include_trace_ids=ids, thinking_type=args.thinking, max_completion_tokens=args.max_completion_tokens, publication_root=Path(args.publication_root))
        elif args.pipeline_action == "repair":
            from .model_api import OpenAICompatibleClient, load_provider
            config = load_provider(Path(args.provider_config), args.provider); client = OpenAICompatibleClient(config, model=args.model, cache_dir=Path(args.cache_dir), timeout_seconds=args.timeout_seconds, max_retries=args.max_retries, thinking_type=args.thinking, max_completion_tokens=args.max_completion_tokens)
            result = repair_trace_analysis(Path(args.batch), Path(args.run_dir), client, args.provider, workers=args.workers, direct_max_chars=args.direct_max_chars, request_max_chars=args.request_max_chars, export_after=not args.no_export)
        elif args.pipeline_action == "export": result = export_trace_analysis(Path(args.run_dir), Path(args.batch))
        elif args.pipeline_action == "label":
            client = None
            if not args.prepare_only:
                from .model_api import OpenAICompatibleClient, load_provider
                config = load_provider(Path(args.provider_config), args.provider)
                client = OpenAICompatibleClient(config, model=args.model, cache_dir=Path(args.cache_dir), thinking_type=args.thinking, max_completion_tokens=args.max_completion_tokens)
            result = run_labeling(Path(args.input), Path(args.output_dir), resume=not args.no_resume, taxonomy_path=Path(args.taxonomy), client=client, workers=args.workers, limit=args.limit, include_lines=set(args.source_lines) if args.source_lines else None)
        elif args.pipeline_action == "select":
            from .pipeline.stage_06_trace_quality_labeling.selection import run_selection
            client = None
            if not args.rules_only:
                from .model_api import OpenAICompatibleClient, load_provider
                config = load_provider(Path(args.provider_config), args.provider)
                client = OpenAICompatibleClient(config, model=args.model, cache_dir=Path(args.cache_dir),
                                                thinking_type="disabled", max_completion_tokens=args.max_completion_tokens)
            result = run_selection(Path(args.labels), Path(args.output_dir),
                                   evidence_path=Path(args.evidence) if args.evidence else None,
                                   sources_path=Path(args.sources) if args.sources else None,
                                   audits_path=Path(args.audits) if args.audits else None,
                                   reviews_path=Path(args.reviews) if args.reviews else None,
                                   policy_path=Path(args.policy) if args.policy else None, mode=args.mode,
                                   client=client, max_input_chars=args.max_input_chars, workers=args.workers)
        elif args.pipeline_action == "status":
            result = pipeline_status(Path(args.preprocessed_root), Path(args.analysis_root), only_incomplete=args.only_incomplete)
            if args.output: result["output_path"] = write_pipeline_status(result, Path(args.output))
        else: result = build_pilot_sample(Path(args.batch), Path(args.output), size=args.size, direct_max_chars=args.direct_max_chars)
    else: raise SystemExit("unsupported command")
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
