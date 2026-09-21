# Stage 06 配置包

当前状态（2026-09-21）：新版打标由 `../relabel.py` 独立执行，S1–S4 筛选由 `pipeline select` 执行，均已消费本目录对应规则与 schema。历史 `runner.py` / `pipeline label` 仍输出旧版格式，未自动迁移；`run_config.json` 的草案开关不控制这两个独立入口。运行方法见仓库 README 和 [筛选使用说明](SELECTION_USAGE.md)。

阅读顺序：label_taxonomy.json → requirement_taxonomy.csv → label_rubrics.yaml →
label_prompt.md → evidence_policy.yaml → ../schemas/ → quality_policy.json → selection_policy.json。

- requirement_taxonomy.csv 是用户原表所需列的快照，保留原始 taxonomy_row；内部分类 ID 非原始 Trace ID。
- rubric YAML 是逐字段判定依据；prompt 不另写一套等级规则。定义修改时同步规范、Rubric 和版本。
- quality_policy.json 已禁用；旧总体质量方案待清理，不是当前筛选依据。
- 空 scope_filters 表示不限制范围；不能为选够数量修改标签。同一版本同一筛选对象只记录一个最终决定。
- 路径以本 config 目录为基准，schema 的 $ref 相对其文件；JSON Schema 方言为 Draft 2020-12。
- manifest.json 固定本包内容哈希；修改任意规则后需更新清单并新建实验版本。

## 校验约束

1. 调用 JSON Schema 校验，启用日期时间格式检查；拒绝未知字段，不将错误回填为 high。
2. 检查 event/turn/evidence/claim/requirement/assessment/mapping ID 唯一、关联存在且范围一致。
3. 引文连续匹配实际原文；核对角色、边界、原始定位、实际提供事件与已省略正文。
4. 父子分类按本 CSV 查回；supported 必须合法子行，partial 仅合法父级，unresolved 不计覆盖。
5. 语义字段逐项 claim 覆盖；每项需求局部评估齐全，核心目标不被替换，次分类不重复主分类。
6. 核对逐标签支持状态与实际复核；旧 gate/完整性/质量字段接入前清理，不执行旧四门聚合。
7. 正式 label 名称与 mapping_choice 一致；筛选 S1–S4 引用同一对象范围的标签与原文，不能复制整体难度给局部任务。
8. human_verified 必须由实际人工复核覆盖关键字段后产生；不在 06 中生成材料可用性判断。
9. 校验模型输出与程序派生字段职责分离；原始 04 数据单独保留，禁止覆盖历史结果。

JSON Schema 不能验证引文的语义支持关系，不能验证文件是否真实存在，也不能替代上述跨文件检查。
无法识别核心目标、无法建立需求的案例进入输入/边界问题队列，不编造数据凑出 model_output。

## 历史配置设计范围

本配置包最初作为独立草案落地，保留历史 prompt.py/runner.py/mapping.py 的运行行为；随后增加了独立打标与筛选实现。模型标签与筛选效果仍需样本校准，程序校验和离线回放不能代替实际人工参考判断。

## 本轮导读修订

按用户确认，移除科研阶段、任务依赖和材料可用性标签；以需求点/子需求点描述科研活动。不新增阶段派生输出。需求演进、外部信息依赖仍可解释难度，必须关联实际挑战；difficulty_factors 的最终精简方式待讨论，当前未删除。证据质量结构也仍待讨论，本轮未改其规则。

## 已确认：保留标签证据支持状态

用 claims[].support_status 表达初标依据判断，实际逐标签复核存入 review.claim_reviews；取值 supported/partial/unsupported/unknown。它用于验证具体标签，不是整个 Trace 的质量等级。复核者类型与标签版本必须记录。原四 gate/完整性/总体质量聚合是此前提案，仍待单独审查；quality_policy.enabled=false，不能把标签错误自动算作 Trace low。

## 主需求与辅助需求

requirement_matching_policy.json 规定 req_001–req_033 可以构成主需求匹配；req_034–req_036
仅作辅助。至少一个前33类父需求有证据支持才具备匹配资格。原始标签可以只含辅助项，但会进入
no_primary_match，不能导出为有效候选，也不能强行制造主需求。partial 若父级有证据、仅子级未知，
仍可计入父级匹配。程序还须核验 selection.primary_requirement_ids 是真实有效映射的集合。

当前不设父/子需求数量硬上限；大于5个父需求或10个子需求的复核阈值为未启用建议。

requirement_taxonomy.csv 最后一列 source_work_object 已按八类规范化，单元格可用中文分号列出
多个候选对象；它不再表示未经修改的原表值。旧值、逐行理由和需结合上下文的标记保存在
work_object_mapping_audit.json。通用项目管理/风险需求必须结合本案例实际对象选择，不能全选。

## 当前筛选方案

见[当前 Trace 筛选流程](../../../../../docs/TRACE_SCREENING_WORKFLOW.md)。selection_policy.json 使用 stage06-selection-v3：主子需求、难度、任务要求、完成条件四项核验，输出 priority_candidate/review/not_selected。筛选不新增语义标签，不要求 historical_confirmed 或 trace_quality=high。S3–S4 需要实际阅读已有原文证据，不能由 schema 自动判定。审核者身份与复核范围另外记录，候选不自动等于人工核验通过；通过独立筛选入口执行。

## 独立筛选入口已实现

`pipeline select` 执行 S1–S4 分流，详见 [运行方式与核验记录格式](SELECTION_USAGE.md)。旧label runner尚未接入新版打标规范；独立筛选入口可以消费新版已有标签，不要求旧总体质量字段。政策中的 automatic_runner_enabled=false 指不随旧label流程自动运行，不禁用显式select命令。
