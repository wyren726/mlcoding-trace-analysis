# Coding Agent 能力分析报告结构

## 1. 目的

本结构用于根据以下核心分析产物生成可阅读、可回溯的能力分析报告：

- `capability_taxonomy.json`
- `capability_cases.jsonl`
- `capability_distribution.csv`

报告保持简洁，只包含：

1. 源数据和分析数据说明；
2. 完整能力分布目录树；
3. 各级能力释义；
4. 每个叶子能力下的全部证据。

## 2. 报告结构

```text
数据说明
→ 完整能力目录树
→ 一级能力释义
   → 二级能力释义
      → 叶子能力释义
         → 全部证据表格
```

## 3. 报告模板

# Coding Agent 能力分布分析报告

## 1. 数据说明

本报告基于以下数据生成：

- 源数据批次：`<batch_id>`
- 原始数据来源：`<source>`
- 原始日志记录数：`<raw_record_count>`
- 有效 Session 数：`<session_count>`
- 有效 Turn 数：`<turn_count>`
- 需求与能力案例数：`<case_count>`
- 能力节点数：`<capability_node_count>`
- 叶子能力数：`<leaf_capability_count>`
- 能力体系版本：`<taxonomy_version>`
- Agent 模型分布：`<model_distribution>`
- Harness 分布：`<harness_distribution>`

简要说明当前数据的覆盖范围和限制，例如样本数量、数据主题是否集中，以及单案例叶子仍需新增数据验证等。

## 2. 能力分布

根据 `capability_taxonomy.json` 生成完整目录树：

```text
Coding Agent 能力
├── 一级能力 A
│   ├── 二级能力 A1
│   │   ├── 叶子能力 A1.1
│   │   └── 叶子能力 A1.2
│   └── 二级能力 A2
└── 一级能力 B
    ├── 叶子能力 B1
    └── 叶子能力 B2
```

目录树应展示完整能力层级，不省略叶子节点。

## 3. 能力与证据

### 3.1 `<一级能力名称>`

**释义：**  
`<说明该一级能力覆盖的总体问题和能力范围。>`

#### 3.1.1 `<二级能力名称>`

**释义：**  
`<说明该二级能力负责解决什么问题，以及它与相邻能力的区别。>`

##### `<叶子能力名称>`

**释义：**  
`<说明该叶子能力的输入、Agent 应执行的行为和预期输出。>`

- 能力状态：`supported`
- 支持案例数：`<case_count>`

| 案例 | 用户需求 | 满足情况 | 证据类型 | 证据原文 | Trace / Turn / Event |
|---|---|---|---|---|---|
| `<case_id>` | `<user_need>` | `<fulfillment>` | 用户需求 | `<quote>` | `<trace_id> / <turn_id> / <event_id>` |
| `<case_id>` | `<user_need>` | `<fulfillment>` | Agent执行或结果 | `<quote>` | `<trace_id> / <turn_id> / <event_id>` |

按照相同格式依次列出全部一级能力、二级能力和叶子能力。

## 4. 证据表格规则

1. 每一行只记录一条证据。
2. 同一案例包含多条证据时，必须占用多行。
3. 每个叶子能力下的所有案例和所有证据都必须列出，不能只选择代表案例。
4. `requirement` 类型显示为“用户需求”。
5. `fulfillment` 类型根据事件内容显示为“Agent执行或结果”或“用户反馈”。
6. 证据原文必须来自 `capability_cases.jsonl`，不得改写或概括。
7. 每条证据必须保留 `trace_id + turn_id + event_id`，方便回到统一 Trace 定位完整上下文。
8. `fulfillment` 为 `unclear` 时应如实展示，不能自动解释为已满足或未满足。
9. 单案例叶子与其他叶子采用相同展示方式，仅通过括号内的独立用户 Query 数说明当前证据量。

## 5. 内容控制

报告不额外加入以下内容，除非后续单独提出需求：

- 大量字段说明表；
- 重复的多层统计表；
- 模型或 Harness 对比；
- 训练数据筛选；
- 未满足需求优先级排序；
- 人工审核流程说明；
- 只展示部分代表性案例的摘要。

这些内容属于独立的附加分析，不与基础能力分布报告混合。
