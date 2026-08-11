from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..analysis.core import (build_candidate_core, organize_hierarchy, publish_reviewed_core,
                             suggest_reviews, validate_core_snapshot)
from ..analysis.extensions import run_extension
from ..features.basic import run_basic_features
from ..features.custom import TASKS, configured_client, run_semantic_features, write_dry_run
from ..features.model_api import load_provider
from ..features.incremental import compatible_feature_files
from ..preprocessing.pipeline import run_preprocess


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifacts_exist(stage: dict[str, Any]) -> bool:
    artifacts = stage.get("artifacts") or []
    return bool(artifacts) and all(Path(path).exists() for path in artifacts)


def _file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execute(manifest_path: Path, manifest: dict[str, Any], name: str,
             action: Callable[[], tuple[dict[str, Any], list[str]]],
             input_fingerprint: str | None = None) -> dict[str, Any]:
    known = manifest["stages"].get(name)
    if (known and known.get("status") == "completed" and _artifacts_exist(known)
            and known.get("input_fingerprint") == input_fingerprint):
        return known["result"]
    started = dt.datetime.now().astimezone()
    manifest["stages"][name] = {"status": "running", "started_at": started.isoformat()}
    _write_manifest(manifest_path, manifest)
    try:
        result, artifacts = action()
        finished = dt.datetime.now().astimezone()
        manifest["stages"][name] = {
            "status": "completed", "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - started).total_seconds(), 3),
            "artifacts": artifacts, "result": result, "input_fingerprint": input_fingerprint,
        }
        _write_manifest(manifest_path, manifest)
        return result
    except Exception as exc:
        finished = dt.datetime.now().astimezone()
        manifest["stages"][name] = {"status": "failed", "started_at": started.isoformat(),
                                    "finished_at": finished.isoformat(), "error": str(exc)}
        manifest["status"] = "failed"
        manifest["next_action"] = f"修复阶段 {name} 的错误后，使用 --resume {manifest['run_id']} 继续。"
        _write_manifest(manifest_path, manifest)
        raise


def run_update(config_path: Path, source_overrides: list[str] | None = None,
               resume_run_id: str | None = None, review_decisions: Path | None = None,
               hierarchy_operations: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent

    def resolve(value: str | Path | None, default: Path) -> Path:
        path = Path(value).expanduser() if value else default
        return (project_root / path).resolve() if not path.is_absolute() else path.resolve()

    runs_root = resolve(config.get("runs_root"), project_root / "pipeline-runs")
    if resume_run_id:
        run_dir = runs_root / resume_run_id
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest["config"]
        sources = manifest["sources"]
    else:
        sources = source_overrides or config.get("sources") or []
        if not sources:
            raise ValueError("update requires sources in config or --source")
        sources = [str(resolve(value, project_root)) for value in sources]
        now = dt.datetime.now().astimezone()
        signature = hashlib.sha256(json.dumps({"sources": sources, "config": config}, sort_keys=True).encode()).hexdigest()[:8]
        run_id = f"update_{now.strftime('%Y%m%d_%H%M%S_%f')}_{signature}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "run_manifest.json"
        manifest = {"run_id": run_id, "status": "running", "created_at": now.isoformat(),
                    "config_path": str(config_path.resolve()), "sources": sources, "config": config,
                    "stages": {}, "next_action": None}
        _write_manifest(manifest_path, manifest)

    roots = config.get("roots") or {}
    preprocessed_root = resolve(roots.get("preprocessed"), project_root / "preprocessed")
    features_root = resolve(roots.get("features"), project_root / "features")
    analysis_root = resolve(roots.get("analysis"), project_root / "analysis")
    for root in (preprocessed_root, features_root, analysis_root):
        root.mkdir(parents=True, exist_ok=True)
    preprocess_result = _execute(manifest_path, manifest, "preprocess", lambda: (
        (result := run_preprocess(sources, preprocessed_root, config.get("adapter"), config.get("limit"))),
        [result["output_path"], str(preprocessed_root / "data_registry.csv")],
    ))
    batch_dir = Path(preprocess_result["output_path"])
    basic_result = _execute(manifest_path, manifest, "basic_features", lambda: (
        (result := run_basic_features(batch_dir, features_root)),
        [result["output_path"], str(features_root / "feature_registry.csv")],
    ))

    semantic_outputs: dict[str, str] = {}
    semantic_results: dict[str, dict[str, Any]] = {}
    llm = config.get("llm") or {}
    for feature_set in config.get("semantic_features") or []:
        def semantic_action(feature_set: str = feature_set) -> tuple[dict[str, Any], list[str]]:
            if llm.get("dry_run", False):
                result = write_dry_run(batch_dir, features_root, feature_set, config.get("semantic_limit"),
                                       llm.get("max_input_chars", 24000))
            else:
                client = configured_client(resolve(llm.get("provider_config"), project_root / "configs/providers.toml"),
                                           llm.get("provider", "pjlab_proxy"), llm.get("model"),
                                           resolve(llm.get("cache_dir"), project_root / ".cache/llm"),
                                           llm.get("timeout_seconds"), llm.get("max_retries"))
                result = run_semantic_features(batch_dir, features_root, feature_set, client,
                                               llm.get("provider", "pjlab_proxy"),
                                               config.get("semantic_limit"), llm.get("max_input_chars", 24000))
                if result.get("status") in {"failed", "partial"}:
                    raise RuntimeError(
                        f"Semantic feature {feature_set} has {result['error_count']} failed Turn(s); "
                        f"resume will reuse completed labels and retry failures; see {result['output_path']}"
                    )
            return result, [result["output_path"]]

        result = _execute(manifest_path, manifest, f"semantic:{feature_set}", semantic_action)
        if not llm.get("dry_run", False):
            semantic_outputs[feature_set] = result["output_path"]
            semantic_results[feature_set] = result

    capability_features = [str(resolve(path, project_root))
                           for path in config.get("historical_capability_features") or []]
    demand_result = semantic_results.get("demand_capability_instances") or {}
    selected_judge_model = str(demand_result.get("model") or llm.get("model") or "")
    capability_features.extend(str(path) for path in compatible_feature_files(
        features_root, "demand_capability_instances", "v3",
        "" if llm.get("dry_run", False) else llm.get("provider", "pjlab_proxy"),
        "" if llm.get("dry_run", False) else selected_judge_model,
    ))
    if semantic_outputs.get("demand_capability_instances"):
        capability_features.append(semantic_outputs["demand_capability_instances"])
    capability_features = list(dict.fromkeys(capability_features))

    def extension_feature_paths() -> list[Path]:
        paths = compatible_feature_files(features_root, "basic_turn_features", "v4")
        paths.append(Path(basic_result["output_path"]))
        paths.extend(Path(path) for path in semantic_outputs.values())
        for feature_set in config.get("semantic_features") or []:
            feature_model = str((semantic_results.get(feature_set) or {}).get("model")
                                or llm.get("model") or "")
            paths.extend(compatible_feature_files(
                features_root, feature_set, TASKS[feature_set]["version"],
                llm.get("provider", "pjlab_proxy"), feature_model))
        paths.extend(resolve(path, project_root) for path in config.get("extension_feature_files") or [])
        return list(dict.fromkeys(paths))

    def run_configured_extensions(core_path: Path) -> None:
        extension_dir = core_path.parent / "extensions"
        feature_paths = extension_feature_paths()
        extension_fingerprint = hashlib.sha256(
            ((core_path / "capability_taxonomy.json").read_bytes()
             + "|".join(str(path.resolve()) for path in feature_paths).encode())
        ).hexdigest()
        for extension_name in config.get("extensions") or []:
            _execute(manifest_path, manifest, f"extension:{extension_name}", lambda extension_name=extension_name: (
                (result := run_extension(extension_name, core_path, feature_paths, extension_dir)),
                [result["output_path"]],
            ), extension_fingerprint)
    if capability_features:
        review_enabled = bool(config.get("enable_human_review", False) or review_decisions)
        version = f"{manifest['run_id']}_snapshot"
        core_result = _execute(manifest_path, manifest, "candidate_core", lambda: (
            (result := build_candidate_core([Path(path) for path in capability_features], analysis_root, version,
                                            resolve(config["existing_taxonomy"], project_root)
                                            if config.get("existing_taxonomy") else None)),
            [result["output_path"]],
        ))
        if review_enabled and config.get("generate_review_suggestions", True):
            suggestion_path = run_dir / "review_suggestions.json"
            _execute(manifest_path, manifest, "review_suggestions", lambda: (
                (result := suggest_reviews(Path(core_result["output_path"]), suggestion_path,
                                           float(config.get("merge_threshold", 0.65)),
                                           float(config.get("split_similarity_threshold", 0.32)),
                                           int(config.get("split_min_cases", 4)),
                                           int(config.get("split_min_cluster_size", 2)))),
                [result["output_path"]],
            ))
        review_value = review_decisions or config.get("review_decisions")
        if not review_value:
            if review_enabled:
                manifest["status"] = "waiting_for_human_review"
                manifest["next_action"] = (f"填写审核决定后运行：trace_analysis update --config {config_path} "
                                           f"--resume {manifest['run_id']} --review-decisions <审核决定.json>")
            else:
                snapshot_core_path = Path(core_result["output_path"])
                run_configured_extensions(snapshot_core_path)
                manifest["status"] = "completed"
                manifest["final_core_path"] = str(snapshot_core_path.resolve())
                manifest["final_taxonomy_version"] = core_result["taxonomy_version"]
                manifest["next_action"] = "三层分析与配置启用的附加分析均已完成。"
        else:
            review_path = resolve(review_value, project_root)
            review_fingerprint = _file_fingerprint(review_path)
            reviewed_version = f"{manifest['run_id']}_reviewed_{review_fingerprint[:8]}"
            reviewed = _execute(manifest_path, manifest, "publish_reviewed_core", lambda: (
                (result := publish_reviewed_core(Path(core_result["output_path"]), review_path,
                                                 analysis_root, reviewed_version)),
                [result["output_path"]],
            ), review_fingerprint)
            final_core_path = Path(reviewed["output_path"])
            hierarchy_value = hierarchy_operations or config.get("hierarchy_operations")
            if hierarchy_value:
                hierarchy_path = resolve(hierarchy_value, project_root)
                hierarchy_fingerprint = _file_fingerprint(hierarchy_path)
                hierarchy_version = f"{manifest['run_id']}_final_{hierarchy_fingerprint[:8]}"
                hierarchy_result = _execute(manifest_path, manifest, "organize_hierarchy", lambda: (
                    (result := organize_hierarchy(final_core_path, hierarchy_path, analysis_root,
                                                  hierarchy_version)),
                    [result["output_path"]],
                ), hashlib.sha256(f"{review_fingerprint}|{hierarchy_fingerprint}".encode()).hexdigest())
                final_core_path = Path(hierarchy_result["output_path"])

            validation_path = run_dir / "core_validation.json"

            def validate_action() -> tuple[dict[str, Any], list[str]]:
                result = validate_core_snapshot(final_core_path)
                validation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
                return result, [str(validation_path.resolve()), str(final_core_path.resolve())]

            final_fingerprint = hashlib.sha256(str(final_core_path.resolve()).encode()).hexdigest()
            validation = _execute(manifest_path, manifest, "validate_core", validate_action,
                                  final_fingerprint)
            manifest["status"] = "completed"
            manifest["final_core_path"] = str(final_core_path.resolve())
            manifest["final_taxonomy_version"] = validation["taxonomy_version"]
            manifest["review_decisions"] = {"path": str(review_path), "sha256": review_fingerprint}
            if hierarchy_value:
                manifest["hierarchy_operations"] = {"path": str(hierarchy_path),
                                                     "sha256": hierarchy_fingerprint}
            run_configured_extensions(final_core_path)
            manifest["next_action"] = ("三层核心 Pipeline 与配置启用的附加分析均已完成。"
                                       if config.get("extensions") else
                                       "核心三层 Pipeline 已完成；可在此核心快照上按需运行附加分析。")
    else:
        manifest["status"] = "completed_without_core_analysis"
        manifest["next_action"] = "启用 demand_capability_instances 或配置历史能力特征文件后再生成核心分析。"
    manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
    _write_manifest(manifest_path, manifest)
    return {"run_id": manifest["run_id"], "status": manifest["status"],
            "manifest_path": str(manifest_path.resolve()), "next_action": manifest["next_action"]}
