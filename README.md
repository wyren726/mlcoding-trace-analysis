# Trace Analysis

当前项目用于从真实用户 Trace 中分析 Coding/科研 Agent 痛点，并筛选需求有依据、任务有挑战、完成条件明确的候选任务。

## 当前主流程

```text
01_preprocess
→ 02_user_screen
→ 03_agent_verify
→ 04_build_results
→ 05_report_publish

04 confirmed/review → 06 增量打标 → S1–S4 筛选 → 候选报告与 session 清单
```

流程及统计口径见[当前 Trace 筛选流程](docs/TRACE_SCREENING_WORKFLOW.md)。优先候选不等于人工核验通过或 workspace 已可复现；取得材料后的任务构造与复现核验是后续步骤。

## 安装与运行

使用 Python 3.10 或更新版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

模型服务由 `configs/providers.toml` 配置，凭据通过其中 `api_key_env` 指定的环境变量提供。执行前替换服务地址和模型为可用配置；预览模式不调用模型。

```bash
python -m trace_analysis pipeline run \
  --batch preprocessed/<batch_id> --mode preview
```

核对预览后使用 `--mode execute` 执行 01–05 流程；使用 `pipeline status` 查看进度，使用 `--resume <run_id>` 续跑已有分析。

## 增量打标与筛选

新版打标从已有 Stage04 候选及可追溯的预处理事件读取证据，独立运行：

```bash
python -m trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel \
  --input /path/to/stage04-candidates.jsonl \
  --output /path/to/new-label-run

python -m trace_analysis pipeline select \
  --labels /path/to/new-label-run/labels.jsonl \
  --evidence /path/to/new-label-run/evidence.jsonl \
  --sources /path/to/new-label-run/sources.jsonl \
  --output-dir /path/to/new-selection-run
```

独立打标入口目前读取 `configs/providers.toml` 中的 `pjlab_proxy`；筛选入口支持 `--provider` 和 `--model`。`pipeline label` 保留历史 v5 格式，不能用其输出直接替代新版标签。`pipeline select` 默认对缺失的 S3、S4 核验调用模型；`--rules-only` 跳过模型调用，缺失的语义核验保持待复核。完整输入格式、版本与输出说明见[筛选使用说明](src/trace_analysis/pipeline/stage_06_trace_quality_labeling/config/SELECTION_USAGE.md)。设置 `TRACE_ANALYSIS_503_BACKOFF=1` 可启用有限次数的 503 退避与持久化重试预算。

## 目录与验证

主实现位于 `src/trace_analysis/pipeline/`，模型客户端位于 `src/trace_analysis/model_api/`。标签、证据和运行结果写入 `workspace/analysis-runs/`；分批报告和会话清单写入 `docs/高质量Trace筛选_20260915/`。原始数据、生成的报告、运行产物、缓存和本地环境文件不随代码提交。预处理、运行登记与制品的 schema 随仓库保留。

`scripts/` 包含发布、证据覆盖校验与子需求诊断工具；批次诊断脚本需要本地已生成的报告及证据文件，不是无数据即可执行的示例。

旧版分析、特征与编排代码已从当前源码中移除；历史版本可从 Git 历史查看，本地 `archive/` 不随仓库发布。

```bash
python -m pytest -q
```
