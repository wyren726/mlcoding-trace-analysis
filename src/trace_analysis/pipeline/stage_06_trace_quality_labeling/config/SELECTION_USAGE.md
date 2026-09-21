# Stage06 筛选执行器

入口为 `trace_analysis pipeline select`，实现在 `../selection.py`。它消费已有标签和原文证据，默认调用模型完成缺失的S3–S4核验，再由程序分流；不重跑01–04，不新增语义标签。旧 `pipeline label` 仍是历史v5流程，本入口不会自动把其输出升级成新版标签。

## 运行

在项目根目录，安装项目依赖后执行（路径替换为实际输入；输出目录必须尚不存在）：

```bash
.venv/bin/python -m trace_analysis pipeline select \
  --labels /path/to/labels.jsonl \
  --evidence /path/to/evidence.jsonl \
  --audits /path/to/selection_audits.jsonl \
  --reviews /path/to/claim_reviews.jsonl \
  --output-dir /path/to/new-selection-run
```

`--sources` 可指定 session 来源记录；默认自动读取 labels.jsonl 同目录的 sources.jsonl。`--audits`、`--reviews` 可省略。默认对没有外部核验记录的S3–S4调用模型；使用 `--rules-only` 可关闭模型调用，此时缺失项目保持pending。没有原文证据包时输入检查保持pending，不调用模型。正式label若引用了复核文件，需要通过`--reviews`显式提供对应记录，不能忽略已存在的复核。`--policy`默认使用本目录selection_policy.json；`--mode`默认episode，也支持case_requirement逐条具体需求筛选。

## 输入与版本

- `labels.jsonl`：每行一个新版06 `model_output` 对象，或包含它的正式label。筛选所需结构由 `../schemas/selection_input.schema.json` 约束，沿用已有字段定义，不要求旧的gate/完整性/quality_results。旧v3/v5格式不能直接代入；无法识别case_id时明确报错，能识别案例但结构非法时记录review。
- `evidence.jsonl`：每行一个 `evidence.schema.json` 证据包，按case_id关联。执行器检查事件ID、正文提供状态、引文连续匹配、角色、turn、位置和Episode归属。证据包本身的来源真实性需由上游取证流程保证，匹配引文不等于自动验证语义支持。
- `claim_reviews.jsonl`：已有 `review.schema.json` 逐标签复核记录。accepted覆盖初标支持状态；冲突或未应用的修正阻止直接入选。修正需先生成新label，再针对新版本复核，执行器不暗中改标签。
- `selection_audits.jsonl`：`selection_audit.schema.json` 规定的S1–S4核验记录。审核者必须实际阅读对应范围与证据后填写；每个筛选对象每个规则只接受一条，冲突先裁决。S3–S4不能由字段非空、路径出现或supported枚举自动推断。

`label_sha256` 是完整输入对象的规范JSON摘要：`ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False` 后UTF-8编码取SHA256，不是JSONL物理行摘要。可调用 `selection.content_hash(label)`。`policy_sha256` 是政策文件字节摘要。初次运行输出的模板已给出两种摘要，修改 label 或规则语义后要重新核验。仅删除 S5 且 S1–S4 定义保持一致时，可使用下述显式迁移。

下面是核验记录形状，示例中的ID、摘要、身份、时间、证据和理由均须替换为实际值；不能把此例当作已完成审核：

```json
{
  "audit_id": "audit-case1-episode1",
  "case_id": "case1",
  "assessment_id": "episode1",
  "mode": "episode",
  "label_sha256": "<模板中的label摘要>",
  "policy_sha256": "<模板中的policy摘要>",
  "reviewer_type": "human",
  "reviewer_id": "<实际审核者>",
  "review_time": "<实际ISO8601时间>",
  "rule_checks": {
    "S3": {
      "status": "pass",
      "reason": "<目标、要求、约束和范围的核验依据>",
      "claim_ids": ["<已有主张ID>"],
      "evidence_ids": ["<同一案例中已验证且支持本规则的证据ID>"]
    }
  }
}
```

默认模型核验由程序生成 reviewer_type=model 的记录，身份与时间由程序填写；不会伪造人审记录。输出evaluation的reviewer_type为program，review_ids指向实际提供的审核记录；不据此宣称整个案例已经人工确认。

## 执行边界

- S1：有支持证据的前33类父需求与明确子需求可自动通过；父级明确但子级未知为pending。只有辅助标签仍须核验是否确实没有主需求，不能用未找到代替不存在；提供S1 fail实际核验后排除。
- S2：Episode已有medium/high且证据支持可通过。low需实际逐标签复核或S2筛选核验才排除；未知/不支持先修正。局部模式没有独立难度字段，不能继承Episode难度，必须提供引用局部已有要求及原文的S2核验记录。
- S3–S4：默认模型读取已有标签和证据包作语义判断，使用本目录 selection_review_prompt.md；程序验证对象、版本、claim/evidence关联后采用。显式提供的核验记录不会被自动覆盖，包括pending记录。模型请求/格式/引用失败保持pending，错误单独记录。
- scope_filters：不同维度AND，同维度任一命中即可。已支持的正向类别命中可通过；非目标类别要经过复核且覆盖主次分类才排除。未命中taxonomy_rows或局部领域/工作对象信息不足时保守保留pending，不从未检索到推断不存在。
- 不支持容量、配额或总分，配置启用时明确报错。不设需求数量上限。同一case_id的完全相同输入折叠并计数，不同版本报错；不同case即使属于同一session也不自动删除。
- 不执行磁盘材料可用性探测、不实际重跑任务。旧 S5 已后移到获取 workspace 后，不参与当前分流，也不伪记为通过。

## 输出

- `selection.jsonl`：唯一逐条筛选结果，符合selection.schema.json，包括S1–S4、前置检查、决定、原因、引用和版本。
- `selection_summary.json`：三个集合数量、输入文件摘要、重复记录数、模式和时间。
- `selection_audit_templates.jsonl`：待核验规则及固定版本。模板故意没有审核者、时间、audit_id；补全真实核验后才是有效输入。
- `selection_report.md`：汇总入口。
- `06阶段标签质量与分布审查报告.md`：每次筛选结束自动更新到 labels.jsonl 所在目录；统计优先候选在完整 36 个需求、216 个子需求中的案例数、比例与覆盖，含零覆盖项、部分/未确定映射、来源行与 session 索引，以及当前分流和标签属性分布。
- `session_paths__priority_candidate__{batch_id}__{analysis_run_id}.jsonl`：每次筛选结束自动写入 labels.jsonl 所在目录的优先候选 session 清单，每个 session 一行，字段为 `session_id`、`session_jsonl_path`、`workspace_path`、`case_ids`。同一 session 的多个入选 Episode/局部任务合并，保留所有入选 case_id；不再按历史 confirmed/high 过滤。零候选输出空文件。
- `selection_policy.json`：实际执行的政策快照。
- `model_selection_audits.jsonl`：通过校验的模型核验记录（有记录时生成）。
- `model_review_log.jsonl`：请求摘要、缓存索引、响应元数据或失败原因，逐次落盘；不重复保存模型原始响应。
- `selection_checkpoint.jsonl`：逐条落盘的结果；异常中断可保留已完成记录。
- `selection_review_prompt.md`：本次使用的提示词快照。

修改标签或补齐核验后输出到新目录重跑筛选；不会覆盖原标签、原始记录、旧v3/v5结果或既有筛选运行。输入格式错误、未知审核对象、同case多个版本等在创建输出目录前报错。

## 模型参数与失败处理

默认provider为pjlab_proxy，模型采用该provider的default_model；可用 `--provider`、`--model`、`--provider-config` 指定。`--cache-dir` 默认 .cache/trace-analysis-stage06-selection，复用客户端缓存；统计 model_calls 为客户端调用次数（含缓存命中），不等于实际HTTP请求数。默认关闭thinking，输出上限4096 tokens，可用 `--max-completion-tokens` 修改。

`--max-input-chars` 默认180000，计算提示词和完整输入字符数；超限不截断原文，不调用模型，保持pending并记录原因。不存在截断后自动视为完整的路径。API错误不导致本批其他案例停止。输出目录在首次请求前独占创建，重跑使用新目录；相同输入的请求可命中缓存，不覆盖历史运行。

Python API `run_selection(..., client=client)` 启用模型，未传client时仍只执行规则；CLI默认启用模型，`--rules-only`禁用。模型仅核验S3–S4，S1缺失映射、S2未知或局部难度等仍按已有规则保留待复核，不伪造新标签。

筛选CLI的`--workers`默认4；Python API的`workers`默认1。模型核验可并行，审核记录、日志和检查点由主线程逐条写入，最终selection.jsonl统一排序。原文可能包含U+2028/U+2029等Unicode字符，JSONL只按实际LF分隔记录，不能将原文中的这些字符拆成新记录。

返回结构或claim/evidence关联不合法时，程序携带具体错误、原响应及已有主张的证据关联重试一次；关联错误按每个请求规则分别列出，避免第一处错误掩盖其他规则的问题。重试仍失败则保持pending。有效的pending判断和显式传入的核验不会因此重试或覆盖。每次尝试写入同一model_review_log.jsonl，以repair_attempt区分；仅通过校验的核验写入model_selection_audits.jsonl，每个案例只写一个最终检查点。API请求错误沿用客户端重试策略，不重复执行这类格式修复。重试不得修改原标签，也不得为了通过校验把证据不足改为通过。

## 从04补跑新版标签

`python -m trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel --input <04.jsonl> --old <v5标签.jsonl> --output <新目录>` 复用04边界与v5长Episode取证位置，重新生成新版标签；使用独立selection_label_prompt.md，不加载旧gate/原子能力指令。证据locator直接指向原有预处理Turn JSONL文件、实际物理行号及record_index，结合event_id定位事件；不生成raw_episodes快照。长记录保留未提供正文列表及chunk标识，不声称完整召回。成功行支持同签名续跑，错误日志保留失败原因和每次调用缓存索引；不生成response_*.json。

`--workers` 默认4，`--max-completion-tokens` 默认16384，`--timeout-seconds` 默认180；较长案例可增大输出上限和请求超时。`--max-input-chars` 默认650000，超限明确失败且不截断原文，可在模型上下文容许时增大。实际输出/输入上限、并发数、重试反馈版本及runner摘要写入运行manifest的invocations。引文校验失败时，重试反馈列出具体evidence_id、event_id及已提供原文，要求模型同时重核关联claim；程序不自动改写引文或放宽逐字校验。每次纠错请求只携带原始输入与最近一次输出，避免累积全部失败输出。

输出格式提示要求先生成evidence_refs与claims，再引用实际存在的ID，避免模型在claim_ids里无限枚举不存在的编号。该补充不限制需求/子需求映射数量；格式提示的版本与摘要记录在invocations，每次成功/失败尝试也保留格式版本。不同格式版本结果仍使用相同rubrics，已有成功标签无需因此重算。

程序可回填引文的机械定位字段locator，但前提是event_id存在且正文已提供、quote逐字连续匹配、source_role/turn_id/episode_membership均正确；不修改引文、角色、事件ID或语义判断。sources/错误日志中每次attempt的locator_resolutions记录原值与回填值，便于区分模型输出和程序定位。

请求以紧凑JSON编码，仅从请求中去掉body_provided=false的空事件占位对象；scope中的遗漏事件列表、所有已提供正文、原始输出evidence.jsonl保持完整。该变动减少重复位置字段造成的上下文占用，版本记为compact-json-visible-events-v1。

serialization_resolutions记录三种确定的格式修正：删除根对象额外的空键空值；case_id正确但Episode target_id误用了同一case_id时，同步回填原始episode_id；工具content本身为合法JSON且quote仅因JSON字符串转义不匹配时，只有编码后的完整quote逐字存在于content才按该表示存储。不合并行、不删行号、不改空格或标点，不处理语义改写。原始模型响应保留在缓存中；任何修正后仍须通过完整schema与引用校验。

每条最多校验4次；续跑失败案例时，可从同格式版本的上次响应缓存继续校验和纠错，不必重新生成整份标签。attempt元数据中的resumed_failed_output标识此情况。未展示的事件或仅展示chunk时，错误反馈会指出具体引用ID；仍要求模型重新核对证据与claim，不能把未展示正文当作已读取。

## labels.jsonl 内的中文审核展示

每条成功标签的前部新增 `review_view`，按任务范围、具体要求、属性与难度、需求/子需求、用户反馈、失败机制分组。每项就近显示中文取值、已有claim理由、模型初标支持状态、原文短引文、角色及定位，不生成额外案例文件。既有字段和ID保留供程序读取；没有对应claim时明确提示缺少解释，绝不从其他字段挪用supported。展示由label_view.py确定性生成，筛选时重新计算校验一致性；S3–S4模型请求不携带这个重复展示块。该块不是人工审核结论，未改变原有语义判断。

审核展示 `review_view` 使用英文结构字段（label_name/value/claims/reason/support_status/evidence/quote），中文用于标签名称、理由和原文。枚举保留 medium、supported、user 等原值，支持状态说明只在展示开头说明一次。inline-label-review-v2 是展示版本，不改变语义标签或筛选标准。


## v3 规则与 v2 核验迁移（2026-09-18）

当前只执行 S1–S4，先筛 Trace 再凭 session ID 索取 workspace。未展示文件全文或未确认外部材料可取得，不作为当前门槛；环境、工具及依赖难度标签保留。语义核验可以使用同一案例 evidence_refs 中已验证但未挂在所选 claim 下的原文，须在 reason 中解释它对当前规则的支持；不存在的 ID、无效引文、跨案例引用或不支持的 claim 仍拒收。

`selection_migration.migrate_v2_audits(labels_path, evidence_path, audit_paths, source_policy_path, output_path)` 可显式复用通过旧版校验的核验。它验证标签摘要、旧核验、规则定义与新版本，保留 S1–S4 原状态（包括 pending）、理由、引用、原审核者和时间，另记 policy_migration 来源与迁移时间。无效/过期核验或规则含义变化时拒绝迁移。输入和旧运行不覆盖，新运行可以用 `--audits` 提供迁移文件，仅对缺少有效 S3/S4 的案例调用模型。

selection schema 为读取历史结果保留 v2 的 S5；v3 结果禁止 S5。历史 v2 政策仅用于原记录校验和迁移，新模型核验使用 v3。


## 优先候选 session 清单

清单直接放在标签运行目录（例如 `trace-label-selection-v6-20260917/`），与 labels.jsonl 同级。每次筛选结束原子更新这份清单为最新结果；筛选子目录的 summary 保存路径与 SHA256，历史逐条决定仍保留在各自 selection.jsonl 中。

导出来源为 `sources.jsonl` 的 session_ids 和 source_row.source_references；可以用 `--sources` / Python API 的 `sources_path` 显式指定。来源明确属于单一 batch/run 时使用上述完整文件名；混合或缺少 batch/run 时使用 `session_paths__priority_candidate.jsonl`。`selection_summary.json` 的 `session_export` 记录文件位置、去重后 session 数和缺少的来源信息。

session ID 不从案例 ID 或路径名称猜测。缺少 session ID 的案例列入 `missing_session_case_ids`，清单状态标记 incomplete；仅路径缺失或无法确定一对一对应时保留 session ID，路径字段为 null，并列入 `unresolved_path_session_ids`。导出不检查 workspace 可获得性，不影响原筛选决定。


## 自动生成标签质量与需求分布报告

`distribution_report.py` 在筛选完成后根据本轮 labels、selection、sources、分类表和已提供逐标签复核记录生成报告，不调用模型。文件与 session 清单同放在标签运行目录，例如 `trace-label-selection-v6-20260917/`。`selection_summary.json` 的 `distribution_report` 保存文件路径、SHA256 和本轮覆盖指标；报告标明其对应的筛选运行 ID。

主表按 case_id 去重，只统计 mapping_status=supported、父子关系合法且关联 claim 有支持证据、无冲突的映射；有效逐标签复核覆盖初标。部分匹配、未确定或不支持的映射另表展示，不扩充明确子需求覆盖，也不改变原筛选决定。每个父需求的案例数通过集合合并计算，不累加子需求案例数；命中子需求数单独去重。分母包含所有优先候选案例，零候选仍输出完整零覆盖表，百分比显示“—”。局部模式只统计入选具体要求的映射，同一案例的多个入选局部要求按案例去重，并同时标明筛选单元数。
