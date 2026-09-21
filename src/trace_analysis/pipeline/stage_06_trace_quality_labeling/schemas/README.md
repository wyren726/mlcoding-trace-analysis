# Stage 06 Schema

使用 JSON Schema Draft 2020-12。common.schema.json 提供共用定义，各文件通过相对 $ref 引用。
这些 schema 约束的是新版输出契约，不能直接用来验证旧 v3/v5 文件。

| 文件 | 责任 |
|---|---|
| input.schema.json | 04 来源、原始/内部 ID、Episode 边界与证据包引用 |
| evidence.schema.json | 模型实际可见事件、角色、定位和读取范围 |
| model_output.schema.json | 原始语义判断、需求分类选择、claim 与引文；没有最终质量与人工确认字段 |
| label.schema.json | 已校验模型结果、程序查回两级分类、派生质量、复核与运行来源 |
| review.schema.json | 实际复核意见、审核范围与版本化修正 |
| selection.schema.json | 选集决定、范围、依据及关联标签版本 |
| run_manifest.schema.json | 运行状态、计数与产物索引 |

未通过校验的模型输出保存在原始响应/错误日志，不进入 label.schema.json 所定义的有效结果。
common 中 claim.field_path 相对 model_output；target_id 为目标或具体需求 ID。
review.field_path 指向所审核 label 版本中的字段，修正后必须重新验证并重算派生值。
schema 的 HTTPS $id 仅是命名空间标识，校验时应注册本地文件，不能依赖网络读取。
metadata 中人工身份/时间/hash 由运行器负责，不由模型虚构；详见 config/README.md 的程序检查清单。

review.schema.json 的 claim_reviews 保存逐标签证据核验，与整条复核记录的 accepted/corrected/unresolved 分开。其 claim_id 集合须与 reviewed_claim_ids 一致；这是运行器必须实现的关联校验。claims[].support_status 的初标值不代替实际复核。

selection.schema.json 要求入选决定至少含一个 req_001–req_033 的 primary_requirement_ids；
auxiliary_requirement_ids 只允许 req_034–req_036。标签原始记录允许无主需求以保留真实情况，
准入在 selection 层执行；ID 是否确有证据、是否与映射一致仍须程序关联检查。

selection.schema.json 的 policy_version=stage06-selection-v3，决定为 priority_candidate/review/not_selected；rule_checks 以 S1–S4 为键保存审核状态及原有 claim/evidence 引用。入选要求四项及前置检查全部 pass。evaluation 记录实际筛选者和复核引用；label_sha256 固定被审核版本。schema 只验证结构，执行器仍须验证主子关系、难度、引用支持关系、审核身份和最终决定与规则一致。旧 label/model_output 中质量字段接入前待清理，不作为本次筛选依赖。

执行器新增 selection_input.schema.json（复用现有语义字段，不要求旧质量评估）、selection_audit.schema.json（实际S1–S4筛选核验记录）。selection的evaluation允许program，表示规则执行者；模型/人工核验身份保留在引用的审核记录中。使用说明见 ../config/SELECTION_USAGE.md。

label_view.schema.json 约束 selection_input 中可选的程序派生 review_view；模型不生成，程序验证其内容等于原标签、claim与原文的关联展开。
