# Coding Agent Trace 数据分析 Pipeline 规格

状态：讨论稿，记录截至 2026-08-10 已确认的设计。

本文定义新的数据分析 Pipeline。旧代码、旧数据处理产物和旧分析结果暂不纳入本设计；正式实施前应先完成可恢复备份，再统一归档。

## 1. 目标

Pipeline 长期服务于一组相对固定的分析问题：

1. Coding Agent 用户有哪些真实需求？
2. 哪些用户需求没有被当前 Agent 满足？
3. 为满足这些需求，Coding Agent 缺少哪些细粒度子能力？
4. 每项能力判断有哪些完整、可回溯的真实案例与证据支持？

数据源和数据量会持续增加。新增数据用于补充证据、发现新需求、修正已有结论，并推动能力体系在证据充分的前提下不断细分。

优质训练 Trace 筛选、重要未满足需求排序，以及不同模型和 Harness 的比较，是建立在核心能力分析之上的附加分析，不改变核心分析数据。

## 2. 三层结构

```text
数据预处理层
→ 特征提取层
→ 分析层
```

### 2.1 数据预处理层

接收任意来源和格式的原始日志，尽可能无损地恢复统一的 Trace、Session、Turn 和 Event 结构。

该层只进行格式解析、结构恢复、脱敏、归属和顺序整理，不判断任务难度、用户满意度、任务类型、能力需求或 Agent 表现。

### 2.2 特征提取层

从统一 Trace 和 Turn 中提取可复用的事实特征及语义标签。基础特征通过确定性代码计算；自定义语义特征可通过规则或模型 API 生成。

### 2.3 分析层

围绕固定分析问题持续积累用户需求实例、Agent 缺位实例、能力节点和证据。新增数据进入后更新当前能力体系和结论，而不是每批数据重新建立一套临时分析项目。

## 3. 数据预处理层

### 3.1 输入范围

预处理程序支持：

- 单个文件；
- 多个文件；
- 整个目录；
- 数据收集器输出；
- SLS 等历史日志；
- 未来新增的 Harness 或日志格式。

数据收集器的 `raw_trace_events.jsonl` 只是一个已支持的数据来源。其他来源不会预先经过数据收集器，也不要求提供 `manifest.json` 或 `diagnostics.jsonl`。

### 3.2 Adapter

每种来源通过 Adapter 转换为统一结构。

- 默认根据文件样本自动识别 Adapter；
- 无法识别或存在歧义时停止并提示；
- 允许通过命令参数手工覆盖 Adapter；
- 新来源通过新增 Adapter 接入，不修改上层特征和分析代码。

### 3.3 信息保留原则

预处理不是字段筛选。原始日志中存在的模型、Harness、Token、时间、工具、错误、上下文和其他来源专有信息均应尽量保留。

每个统一事件包括：

- 跨来源统一后的结构化字段；
- 脱敏后的完整原始记录 `raw`。

统一结构用于后续通用处理，`raw` 用于保真和未来回溯。无法识别的内容不得丢弃，应保留为 `unknown` 或无法归属事件。

这里的“原始记录”指脱敏后的输入记录。Pipeline 不恢复或保存收集器已经移除的密钥、身份和本地敏感信息。

### 3.4 层级关系

```text
Batch
→ Trace
→ Session
→ Turn
→ Event
```

- Session 是 Harness 原生定义的一段会话。
- Trace 是预处理层形成的一条完整轨迹，可以包含一个或多个有关联的 Session。
- 没有明确跨 Session 关联证据时，一个 Session 形成一个独立 Trace。
- 只有项目相同、用户相同或时间接近，不足以自动合并 Session。

### 3.5 Turn 定义

一个 Turn 是一次用户触发到对应 Agent 执行结束的完整过程：

```text
一组执行前用户输入
→ 模型调用、Reasoning、工具调用和工具返回
→ Agent 最终回复、中止、打断或日志结束
```

边界规则：

1. 原始日志有明确 `turn_id` 时优先使用。
2. 有明确开始、完成或中止事件时使用原始边界。
3. Agent 尚未开始执行前的连续用户补充消息可归入同一 Turn。
4. Agent 已经开始执行后出现的新用户纠正或打断，结束当前 Turn 并开启新 Turn。
5. System、Developer、环境信息和平台注入不能单独开启 Turn。
6. 无法可靠归属的事件保留在 Trace 的未归属事件中。
7. 每个事件最多归属一个 Turn，避免重复统计。

Turn 结构状态包括：

```text
completed
aborted
interrupted
incomplete
```

这些状态描述日志结构，不评价 Agent 表现。

### 3.6 Turn 内事件

Turn 内只维护一条严格有序的事件流，避免同时保存多份用户输入、步骤和最终回复造成重复统计。

事件的公共结构包含：

```text
event_id
sequence
timestamp
type
data
raw
```

第一版统一事件类型覆盖：

```text
user_message
assistant_message
reasoning
model_call
model_result
tool_call
tool_result
token_usage
error
system_event
unknown
```

`data` 保存跨来源统一后的常用内容，`raw` 保存转换前的脱敏完整记录。

Harness 通常保存在 Session 层；模型信息尽量保存在 Turn 上下文或具体模型调用事件中。原始值始终保留。

### 3.7 预处理输出

```text
preprocessed/
├── data_registry.csv
└── batch_<处理时间>_<短哈希>/
    ├── unified_traces.jsonl
    └── unified_turns.jsonl
```

- `unified_traces.jsonl`：一行一个完整 Trace，是第一层的完整标准数据。
- `unified_turns.jsonl`：一行一个 Turn，由 Trace 确定性展开，便于逐 Turn 提取特征和打标签。为避免重复存储大型原始快照，Turn 事件通过 `trace_id + event_id` 回指 Trace 中内嵌的 `raw`。
- 两个文件中的同一 Turn 使用相同的稳定 `turn_id`。

### 3.8 数据登记表

`data_registry.csv` 每个处理批次一行，历史批次不覆盖。核心字段包括：

```text
batch_id
source
processed_at
adapter
raw_size_bytes
output_size_bytes
session_count
event_count
model_distribution
harness_distribution
output_path
status
```

- `source_id` 和 `batch_id` 默认自动生成，可在高级用法中覆盖。
- `model_distribution` 优先按明确模型调用统计；只有 Turn 级模型时按 Turn 统计，并记录统计口径。
- `harness_distribution` 按 Session 统计。
- 无法识别的模型或 Harness 记为 `unknown`。
- 状态至少包括 `success`、`partial` 和 `failed`。

每次运行创建独立时间批次目录，默认不覆盖历史产物。

## 4. 特征提取层

### 4.1 两类特征

基础特征默认自动提取，包括文本长度、模型调用、Token、耗时、工具使用等确定性事实。

自定义特征按需运行，例如：

- 用户反馈和满意度；
- 任务结果；
- 任务难度；
- Agent 行为质量；
- 用户需求实例；
- 需求协商关系（Agent 计划项及用户接受、修改或拒绝）；
- 能力缺位实例；
- 数据用途和训练适用性。

自定义特征可使用规则或模型 API。

### 4.2 工具特征

工具特征不能只保存总调用数。每个 Turn 应按工具名称记录：

```text
call_count
result_count
argument_length
result_length
```

工具调用与返回优先通过 `call_id` 配对。无法配对的调用或返回仍然保留并标记。

### 4.3 特征记录

不同特征采用统一记录外壳：

```text
trace_id / turn_id
feature_set
feature_version
feature_run_id
generated_by
values
```

模型生成结果的 `generated_by` 至少包含：

```text
method
provider
model
prompt_version
```

规则生成结果记录对应的 Extractor 版本。

### 4.4 特征存储

不同特征任务分开保存，不维护一个不断增加列的巨大特征表：

```text
features/
├── feature_registry.csv
├── basic_turn_features/
│   └── run_<时间>_<短哈希>/features.jsonl
├── difficulty/
│   └── run_<时间>_<短哈希>/features.jsonl
└── <其他特征>/
    └── run_<时间>_<短哈希>/features.jsonl
```

分析时通过 `trace_id` 或 `turn_id` 组合所需特征。

`feature_registry.csv` 记录特征名称、定义版本、输入数据批次、生成方式、模型、Prompt、运行时间、输出位置和状态。

### 4.5 特征增量与版本

需要同时支持数据增量和定义增量：

- 定义不变且新增数据：只处理新增 Trace/Turn。
- 增加新字段：复用未变化字段，只补算新增字段。
- 修改字段含义：创建新版本并重算受影响字段。
- 拆分标签：只重处理原标签及相邻候选数据。
- 更换模型或 Prompt：创建新的特征运行，不覆盖旧结果。

旧标签永远保留。新定义可以读取旧结果作为候选或迁移依据，但必须明确哪些字段复用、哪些字段重算。

## 5. 自下而上的需求与能力发现

用户需求和能力标签不采用固定类别强制分类，而采用以下过程：

```text
抽取用户直接提出的需求
→ 抽取 Agent 计划项与后续用户回应
→ 重建直接需求和已接受的协商需求
→ 证据约束的原子能力提取
→ 案例—能力证据蕴含校验
→ 全局去重、归并和动态层级构建
→ 发布新的能力体系版本
```

开放式抽取首先回答用户目标、处理对象、期望结果、约束、现有 Agent 的未满足点和原始证据，不要求模型从预设能力列表中强制选择。

为了减少叶子能力重叠，一个原子能力实例只归入一个叶子节点；一个需求事件可以拆出多个原子能力实例。
Agent 提议的工作步骤只有在后续用户明确接受、行为性接受、部分接受或修改后接受时，才能成为协商需求；
沉默、转移话题和 Agent 自行执行均不能视为用户接受。

原子能力必须由用户原文直接表达、由原始目标逻辑上不可分割地推出，或由完整的“Agent 提议 + 用户接受”证据链支持。
不得根据常见工作流把宽泛目标展开成未经确认的具体步骤。例如“启动论文复现”不能单独支持下载数据、安装依赖、
实现算法或运行实验。

能力节点可以新增、拆分、合并、重命名或废弃。节点变化保留版本关系，旧能力体系和旧案例归属不被覆盖。

正式叶子节点需要：

- 明确具体的用户需求和完成目标；
- 明确纳入与排除条件；
- 能与相邻节点稳定区分；
- 有可回溯的直接证据或已接受的协商证据链；
- 能说明现有 Agent 的缺位和理想能力。

叶子名称后的数字只表示独立用户 Query 数，不额外标记“候选”；没有有效证据的节点不进入正式能力树。

## 6. 核心分析层

核心分析持续维护：

1. 累计需求和能力缺位实例；
2. 当前能力体系；
3. 每个叶子能力的全部支撑案例和证据；
4. 基于当前全部数据的能力分布和分析结论。

新数据进入后：

```text
提取直接需求和需求协商关系
→ 形成有效需求
→ 进行案例—能力证据蕴含校验
→ 匹配当前能力体系
→ 为已有节点增加证据
→ 形成新增或拆分候选
→ 必要时发布新能力体系版本
→ 更新固定分析问题的当前回答
```

### 6.1 核心输出

```text
analysis/
└── version_<时间>_<版本>/
    └── core/
        ├── capability_taxonomy.json
        ├── capability_cases.jsonl
        └── capability_distribution.csv
```

#### capability_taxonomy.json

保存完整层级能力体系、节点定义、父子关系、纳入和排除条件、节点状态、案例数、Trace 数及能力体系版本。

#### capability_cases.jsonl

一行对应一个支持叶子能力的完整案例，保存：

```text
taxonomy_version
capability_id
case_id
trace_id
session_id
turn_ids
user_need
unmet_need
evidence
```

`evidence` 列出该案例的全部必要证据，包括 `event_id`、证据类型和原文摘录。通过 Trace、Turn 和 Event ID 可回到 `unified_traces.jsonl` 中的完整事件及其 `raw` 原始记录。

该文件保存全部案例，不是报告中的代表案例抽样。

#### capability_distribution.csv

将能力树展开成一行一个节点的分布表，至少包含：

```text
capability_id
parent_id
level
capability_path
name
is_leaf
case_count
trace_count
evidence_count
```

发布前必须校验：

```text
节点 case_count
= capability_cases.jsonl 中该节点的唯一 case_id 数
```

以及：

```text
节点 trace_count
= 该节点全部案例中的唯一 trace_id 数
```

## 7. 附加分析

附加分析独立读取核心分析和特征，不修改核心产物。

```text
analysis/
└── version_<时间>_<版本>/
    └── extensions/
        ├── model_capability_comparison.csv
        ├── harness_capability_comparison.csv
        ├── training_trace_candidates.jsonl
        └── unmet_need_priority.csv
```

目前每项扩展只有一个输出文件，因此不额外增加输出子目录。扩展代码仍按职责独立组织；某项扩展将来确实产生多个文件时，再增加对应输出目录。

### 7.1 模型与 Harness 比较

比较应区分任务分布与 Agent 表现。不同模型或 Harness 接触到更多某类需求，不等于它在该能力上更强或更弱。

比较至少考虑能力节点、任务难度、数据来源和另一项变量（模型或 Harness）。无法控制混杂因素时，结果必须表述为观察性差异，不能解释为因果效果。

### 7.2 训练 Trace 筛选

优质正向训练 Trace 的初步标准：

```text
数据完整
+ 用户目标明确
+ 高质量完成
+ 有验证证据
+ 用户满意
+ 中高难度
+ 行为值得模仿
```

训练用途至少区分：

```text
positive_sft
repair_trace
preference_pair
evaluation_candidate
exclude
```

高满意度和高难度是重要信号，但不能替代完成质量、验证证据和数据完整性。

### 7.3 重要未满足需求

重要程度综合考虑：

```text
独立案例和 Trace 中的出现频率
用户不满意程度
任务失败或受阻程度
影响严重性
跨来源覆盖
能力缺口明确程度
证据可信度
```

频率按独立案例或 Trace 统计，不按消息数量统计。第一版优先保留各维度，不急于压缩成不透明的单一总分。

## 8. 模型 API

所有模型调用通过统一 `LLMClient`，特征或分析代码不直接绑定某个供应商。

第一版支持：

- OpenAI 官方接口；
- OpenAI-compatible 第三方接口。

Provider 配置保存 Base URL、API 类型、默认模型和 API Key 环境变量名。API Key 不写入仓库、日志或分析产物。

相同输入、特征定义、Prompt 和模型允许复用缓存。调用支持重试、错误记录和断点继续。

## 9. 人工审核

> 当前实施状态：暂缓。人工审核的触发条件、证据门槛、交互方式和发布语义尚未经过逐项讨论，
> 因此不作为标准 Pipeline 的阻塞阶段。标准流程直接输出当前数据支持下的分析快照和附加分析；
> 本节仅保留为后续讨论方向，确认前不得据此硬编码审核门槛。

人工介入集中在会改变分析含义的环节：

1. 新 Adapter 首次接入的映射抽检；
2. 新特征或特征定义升级后的标签抽检；
3. 能力节点新增、拆分、合并和边界修改；
4. 正式能力版本发布前的数量与证据完整性检查；
5. 训练候选正式进入训练集前的抽检。

普通新增数据在格式、定义和能力体系没有变化时，可以自动增量处理。

人工修改保留修改前结果、修改后结果、审核理由、时间和状态。

## 10. 代码与数据目录

```text
trace_analysis/
├── src/
│   └── trace_analysis/
│       ├── cli.py
│       ├── preprocessing/
│       │   ├── pipeline.py
│       │   └── adapters/
│       ├── features/
│       │   ├── basic/
│       │   ├── custom/
│       │   └── model_api/
│       └── analysis/
│           ├── core/
│           ├── extensions/
│           │   ├── model_comparison/
│           │   ├── harness_comparison/
│           │   ├── training_selection/
│           │   └── unmet_need_priority/
│           └── shared/
├── raw_data/
├── preprocessed/
├── features/
├── analysis/
├── configs/
├── tests/
└── archive/
```

核心分析和附加分析的代码、输出分别存放。公共读取、查询和校验逻辑可以共享，但扩展分析不能反向修改核心产物。

## 11. 命令入口

统一入口：

```bash
python3 -m trace_analysis <command>
```

计划提供：

```bash
python3 -m trace_analysis preprocess --source <文件、多个文件或目录>
python3 -m trace_analysis features --batch <batch-id> --feature-set <名称>
python3 -m trace_analysis analyze-core
python3 -m trace_analysis analyze-extension --name <扩展名称>
python3 -m trace_analysis update --source <新增数据>
```

`update` 依次执行新增数据预处理、默认特征提取、当前启用的语义特征、核心能力更新和默认附加分析。批次 ID、时间后缀和 Adapter 默认自动生成；高级用法允许覆盖。人工审核在完成专项设计前不阻塞 `update`。

阶段失败后可以从失败阶段继续，不重跑已经成功且依赖未变化的阶段。

## 12. 增量运行

```text
新增原始数据
→ 只生成新预处理批次
→ 只为新增 Trace/Turn 提取现有特征
→ 将新需求实例加入累计案例池
→ 匹配已有能力或形成新增/拆分候选
→ 发布新的完整核心分析快照
→ 按依赖更新附加分析
```

重跑范围：

- 新增数据：只处理新批次。
- 特征定义变化：只重算受影响数据和字段。
- 能力节点拆分：只重分配原节点及相邻候选案例。
- 某个扩展规则变化：只重跑该扩展。
- 正式分析版本：每次保存完整快照，旧版本不覆盖。

新版本应记录新增数据、节点变化、案例重新归类和已更新扩展。

## 13. 待后续确认

以下内容尚未定稿，实施前或对应阶段开始前继续逐项讨论：

1. 第一版支持的具体源格式和 Adapter 检测规则。
2. 统一 Trace、Turn 和 Event 的正式 JSON Schema。
3. 稳定 ID 和跨批次去重算法。
4. 基础特征的完整字段清单与精确统计口径。
5. 用户反馈、任务结果、难度和行为质量的正式定义。
6. 能力节点建立、拆分和合并的证据门槛。
7. 人工审核记录的具体存储形式。
8. 模型与 Harness 比较的匹配和校正方法。
9. 训练候选与重要未满足需求的最终准入、排序标准。
10. 大规模数据下 JSONL 的索引和查询实现。

## 14. 实施顺序

1. 审阅并确认本规格。
2. 对旧代码和旧产物创建可恢复备份。
3. 将旧内容归档到 `archive/legacy_<时间>/`。
4. 创建新的最小代码和目录框架。
5. 实现并验证数据预处理层。
6. 实现基础特征和自定义标签框架。
7. 实现核心能力分析及全量证据回溯。
8. 分别实现附加分析。

在规格确认和备份完成前，不移动或删除旧内容。
