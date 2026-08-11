# Coding Agent Trace Analysis

本仓库正在按三层结构重建 Coding Agent Trace 数据分析 Pipeline：

```text
数据预处理层 → 特征提取层 → 核心与附加分析层
```

设计基线见 [`docs/NEW_PIPELINE_SPEC.md`](docs/NEW_PIPELINE_SPEC.md)。旧分析代码和旧产物已可逆归档至 `archive/legacy_20260810/`，原始数据保留在 `raw_data/`。独立的数据收集、转换和发布工具已移至同级目录 `../trace_collection_tooling/`。

## 当前状态

已建立新的 `src/trace_analysis/` 代码框架，并实现预处理 CLI 与两个候选 Adapter：

- `collector_events`：数据收集器的 `raw_trace_events.jsonl`；
- `sls_proxy`：SLS Proxy 外层 JSONL 加嵌套 `content` JSON。

`collector_events` 已适配 Codex collector v0.2.0，并在远程真实数据上完成 1,000 事件 Smoke Test；工具调用、结果、模型和 Harness 均可恢复。`sls_proxy` 与 `session_jsonl` Adapter 也已实现并通过测试。预处理使用批次内 SQLite 暂存事件，再按 Session 输出，以避免把整个数据集保存在内存中。

特征层已实现 `basic_turn_features`、多种语义标签和开放式需求能力实例抽取。分析层已实现科研 ML/LLM Coding 范围筛选、自下而上能力归纳、证据索引、模型/Harness 对比、训练候选筛选、未满足需求排名和正式报告生成。

数据批次、特征运行和分析版本同时写入 `registries/` 下的不可变 JSON 登记记录；传统 CSV 是可重建视图。大型数据和分析制品不进入 Git，通过 SSH 制品存储同步：

```bash
python3 -m trace_analysis sync status --config configs/sync.toml
python3 -m trace_analysis sync push --config configs/sync.toml --dry-run
python3 -m trace_analysis sync pull --config configs/sync.toml --dry-run

python3 -m trace_analysis registry rebuild \
  --kind datasets \
  --output preprocessed/data_registry.csv

python3 -m trace_analysis registry manifest \
  --output artifacts/manifest.json
```

## 运行预处理

当前项目采用 `src` 布局。开发环境可直接运行：

```bash
PYTHONPATH=src python3 -m trace_analysis preprocess \
  --source <单个文件、多个文件或目录> \
  --output-root preprocessed
```

Adapter 默认自动识别，也可使用 `--adapter collector_events` 或 `--adapter sls_proxy` 覆盖。Smoke Test 可增加 `--limit 100`。

每次运行生成：

```text
preprocessed/
├── data_registry.csv
└── batch_<时间>_<短哈希>/
    ├── unified_traces.jsonl  # 内嵌完整脱敏 Raw
    └── unified_turns.jsonl   # 通过 trace_id + event_id 回指 Raw
```

## 测试

```bash
PYTHONPATH=src python3 -m pytest tests/test_new_preprocessing.py -q
```

## 提取基础特征

```bash
PYTHONPATH=src python3 -m trace_analysis features \
  --batch preprocessed/<batch-id> \
  --feature-set basic_turn_features \
  --output-root features
```

输出写入 `features/basic_turn_features/run_<时间>_<短哈希>/features.jsonl`，并更新 `features/feature_registry.csv`。

语义特征示例：

```bash
export TRACE_ANALYSIS_API_KEY='...'
PYTHONPATH=src python3 -m trace_analysis features \
  --batch preprocessed/<batch-id> \
  --feature-set user_feedback \
  --provider pjlab_proxy \
  --model glm-5.2 \
  --limit 10
```

Provider 和允许使用的模型配置在 `configs/providers.toml`。API Key 只从环境变量读取。

调用 API 前可以先检查将要发送的内容：

```bash
PYTHONPATH=src python3 -m trace_analysis features \
  --batch preprocessed/<batch-id> \
  --feature-set user_feedback \
  --dry-run \
  --limit 3
```

开放式需求与能力实例抽取：

```bash
PYTHONPATH=src python3 -m trace_analysis features \
  --batch preprocessed/<batch-id> \
  --feature-set demand_capability_instances \
  --provider pjlab_proxy \
  --model glm-5.2 \
  --limit 10
```

输出仍按独立运行目录保存，不覆盖旧标签：

```text
features/demand_capability_instances/run_<时间>_<短哈希>/
├── features.jsonl  # 每个 Turn 的开放式需求、满足情况、能力缺口和全部证据引用
└── errors.jsonl    # 无法解析或证据引用无效的记录
```

每个原子实例带稳定的 `instance_id`；证据字段只能引用当前 Turn 中真实存在的事件。后续核心分析层读取这些实例做能力归并与层级细分，而不会直接改写本层结果。

## 生成待审核的核心分析快照

```bash
PYTHONPATH=src python3 -m trace_analysis analyze-core \
  --feature-file features/demand_capability_instances/<run-id>/features.jsonl \
  --existing-taxonomy analysis/<上一版本>/core/capability_taxonomy.json \
  --output-root analysis
```

`--feature-file` 可一次传入多个历史及新增特征文件。相同 `instance_id` 只计为一个案例，同时在 `source_features` 中保留它出现过的全部特征版本、运行和模型来源。

该命令输出 `capability_taxonomy.json`、保存全部案例和证据的 `capability_cases.jsonl`，以及 `capability_distribution.csv`。对照上一版能力树时，节点分为：

- `matched_existing`：名称完全一致，复用已有 `capability_id`；
- `review_possible_match`：名称相似，只提供可能的旧节点，不自动合并；
- `new_candidate`：没有足够相似的节点，作为新能力候选。

这些结果仍是待审核草稿，不能当作已发布分类；下一阶段需要实现人工接受、改名、合并和拆分操作。

## 审核并发布能力版本

先复制并填写 [`configs/capability_review.example.json`](configs/capability_review.example.json)，然后运行：

```bash
PYTHONPATH=src python3 -m trace_analysis review-core \
  --draft-core analysis/<草稿版本>/core \
  --decisions <审核决定.json> \
  --taxonomy-version <正式版本号> \
  --output-root analysis
```

支持 `accept`、`reject`、`rename`、`merge`、`assign_existing` 和 `split`。每项决定必须填写理由；`split` 必须把原节点的每个案例恰好分配一次。未审核节点保留在 taxonomy 的 `candidates` 中，不进入正式 `root.children`；被拒绝案例仍保留在 `capability_cases.jsonl` 并标记 `assignment_status=rejected`。发布目录额外保存 `review_audit.json`，记录审核人、决定和来源文件。

## 生成合并与拆分审核建议

```bash
PYTHONPATH=src python3 -m trace_analysis suggest-review \
  --draft-core analysis/<草稿版本>/core \
  --output analysis/<草稿版本>/review_suggestions.json
```

建议文件包含：

- `merge_candidates`：节点名称与案例内容相似的能力对，包含两个节点的全部案例 ID、名称分数、案例内容分数和共同文本片段；
- `split_candidates`：同一节点内部形成多个低相似度案例组的宽泛能力，包含全部分组案例 ID和代表性需求；
- `thresholds`：本次使用的阈值，便于复现和调节；
- `llm_review.status=not_run`：明确表示当前只是规则候选发现，不包含模型判断。

可通过 `--merge-threshold`、`--split-similarity-threshold`、`--split-min-cases` 和 `--split-min-cluster-size` 调节敏感度。该命令只生成建议，不修改 taxonomy；人工仍需将确认后的决定写入审核 JSON，再运行 `review-core`。

## 使用模型复核规则建议

先用 dry-run 检查候选和案例内容：

```bash
PYTHONPATH=src python3 -m trace_analysis review-suggestions-llm \
  --draft-core analysis/<草稿版本>/core \
  --suggestions analysis/<草稿版本>/review_suggestions.json \
  --model deepseek-v4-flash-0731 \
  --dry-run
```

确认后移除 `--dry-run` 并通过环境变量提供 API Key。模型仅复核规则阶段已经发现的候选，输出：

```text
analysis/llm_review_suggestions/run_<时间>_<哈希>/
├── reviews.jsonl
├── errors.jsonl
└── manifest.json
```

合并候选的模型决定为 `merge / keep_separate / insufficient_evidence`；拆分候选为 `split / keep_together / insufficient_evidence`。模型引用的案例必须属于当前候选；建议拆分时必须恰好覆盖全部案例且不能重复。结果同时保留原规则候选、模型建议、模型与 Prompt 版本，但不会自动生成审核决定或修改 taxonomy。

## 将叶子能力组织成多层能力树

复制并填写 [`configs/capability_hierarchy.example.json`](configs/capability_hierarchy.example.json)，然后运行：

```bash
PYTHONPATH=src python3 -m trace_analysis organize-hierarchy \
  --core analysis/<已审核版本>/core \
  --operations <层级操作.json> \
  --taxonomy-version <新版本号> \
  --output-root analysis
```

支持的操作为：

- `create_parent`：创建父能力，并将指定节点移动到其下；
- `move`：移动已有节点；
- `rename`：重命名已有节点但保持稳定 ID。

发布前会校验不存在循环、重复节点、未知父节点，以及“父节点直接持有案例”等错误。案例仍只归属叶子能力；父节点的 `case_count`、`trace_count` 和 `evidence_count` 根据全部后代叶子去重汇总。新版本重新生成完整 taxonomy、案例索引和带 `parent_id / level / capability_path` 的分布表，并保存 `hierarchy_audit.json`。

## 一键增量更新

复制 [`configs/update.example.json`](configs/update.example.json) 为 `configs/update.json`，填写新增数据、历史能力特征和上一版 taxonomy，然后运行：

```bash
PYTHONPATH=src python3 -m trace_analysis update \
  --config configs/update.json
```

也可以用 `--source <新增文件或目录>` 临时覆盖配置中的数据源。自动阶段依次为：

```text
preprocess
→ basic_features
→ 配置中启用的 semantic features
→ candidate_core
→ review_suggestions
→ waiting_for_human_review
```

每次运行在 `pipeline-runs/update_<时间>_<哈希>/` 中生成 `run_manifest.json` 和可选的 `review_suggestions.json`。Manifest 记录每个阶段的开始与结束时间、耗时、状态、产物、错误和下一步操作。到达能力审核时会暂停，不会擅自发布 taxonomy。

失败或暂停后恢复：

```bash
PYTHONPATH=src python3 -m trace_analysis update \
  --config configs/update.json \
  --resume <update-run-id>
```

恢复时，只有状态为 `completed` 且登记产物仍然存在的阶段才会跳过；失败阶段及产物丢失阶段会重新执行。API Key 仍只从环境变量读取，不写入配置或运行清单。示例默认设置 `llm.dry_run=true`，正式调用前可以先检查语义特征请求。

完成候选能力审核后，用同一个 run 继续主流程：

```bash
PYTHONPATH=src python3 -m trace_analysis update \
  --config configs/update.json \
  --resume <update-run-id> \
  --review-decisions <审核决定.json>
```

如果还需要把审核后的叶子能力组织成多层树，可以同时增加：

```bash
--hierarchy-operations <层级操作.json>
```

后半段依次执行：

```text
publish_reviewed_core
→ 可选 organize_hierarchy
→ validate_core
→ completed
```

`validate_core` 会同时校验 taxonomy、`capability_cases.jsonl` 和 `capability_distribution.csv`：节点 ID 必须一致、案例只能归属叶子、案例 ID 不得重复、父子统计必须与后代案例一致、路径和层级必须一致。成功后 Manifest 写入 `final_core_path` 和 `final_taxonomy_version`。审核文件或层级文件发生变化时，系统根据 SHA-256 指纹只重跑审核及其下游阶段。

## 附加分析

附加分析只读取正式核心快照和选定的特征文件，不修改核心 taxonomy 或案例归属：

```bash
PYTHONPATH=src python3 -m trace_analysis analyze-extension \
  --name <扩展名称> \
  --core analysis/<版本>/core \
  --feature-file <basic-features.jsonl> <semantic-features.jsonl> ... \
  --output-dir analysis/<版本>/extensions
```

支持四项扩展：

- `model_comparison` → `model_capability_comparison.csv`：按能力和被分析的 Agent 模型统计完成、未满足、负反馈及难度，同时输出 Harness、数据批次和难度分布；Judge 模型不会被当成 Agent 模型。
- `harness_comparison` → `harness_capability_comparison.csv`：按能力和 Harness 统计相同指标，同时输出 Agent 模型、数据批次和难度分布。
- `training_selection` → `training_trace_candidates.jsonl`：逐 Trace 推荐 `positive_sft / repair_trace / evaluation_candidate / exclude`，保留所有判定信号、覆盖率、原因和限制。没有成对轨迹证据时不会虚构 `preference_pair`。
- `unmet_need_priority` → `unmet_need_priority.csv`：按独立案例和 Trace 汇总未满足频率、负反馈、失败、难度、模型与 Harness 覆盖和证据量。样本不足时输出 `insufficient_evidence`，不强行给高优先级。

如果同一 Turn、同一特征类型传入了互相冲突的标签运行，扩展分析会停止并要求明确选择一个版本，避免混合不同定义或 Judge 模型的结果。所有比率均附带对应的 `*_labeled_count` 或覆盖率；缺失标签不会自动当作正面或负面。

`configs/update.json` 中的 `extensions` 可启用任意扩展。核心发布并校验通过后，`update` 会自动把它们写入同一版本的 `extensions/`，并在 Manifest 中分别记录 `extension:<name>` 阶段。

模型调用可使用 `--max-input-chars`、`--timeout-seconds` 和 `--max-retries` 控制输入与失败时间。当前接口实测中 `deepseek-v4-flash-0731` 可以完成需求抽取，而 `glm-5.2` 在同一输入上出现超时。
