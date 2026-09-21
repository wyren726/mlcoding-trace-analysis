"""Materialize explicit local semantic review findings with checked citations."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/高质量Trace筛选_20260915'
OUT=DOC/'六项零命中语义核查_20260921'

FINDINGS=[
 ('sub_047','episode_be44d396','evt_91084c39c005f505c4c118c2','随机分配或匹配的具体方法','边界案例','用户要求补充论文中随机分配或匹配的方法；尚不能确定是在设计新随机化方案，还是报告既有方法。不能直接认定漏标。'),
 ('sub_047','episode_713c50','evt_fbad3b524d88020a29d469bf','不要涉及临床随机对照试验','否定需求','用户明确排除RCT，关键词出现不构成随机化设计需求。'),
 ('sub_051','episode_b8157','evt_ac91db64d6136d98be88a4db','4.5 验证驱动参数标定','疑似漏标','用户给定框架包含训练/验证/测试划分、候选区间和标定准则设计，现有映射集中于训练优化与论文方法撰写。需厘清本轮是设计协议还是执行已有协议，不能维持真实零需求结论。'),
 ('sub_051','episode_139875','evt_e8ed0099a43019f8dab10dbf','不要这种调参实验','否定需求','用户明确要求跳过调参实验，不计作正例。'),
 ('sub_054','episode_325946','evt_d0a0fc9b561415e7c4542d77','至少应使用多个独立合成数据生成机制','疑似漏标','用户提出改变数据生成机制并观察结论稳定性，符合分布变化下稳健性条件设计。现有supported映射均为论文写作/格式/复现文档；需核实该轮审稿意见是否作为研究设计修改任务采纳。'),
 ('sub_054','episode_390804','evt_79baa3f177e86b058f9d968b','机械臂负载突变实验','边界案例','用户在机器人模型变化场景要求负载突变实验及研究计划，支持核查实验设计细粒度漏标，但尚未给出完整扰动条件，不能仅凭鲁棒控制背景自动补标。'),
 ('sub_108','episode_1dd7d82','evt_18794da82e67056358548537','完成这个模块的适配','相邻需求','从LASER迁移思想到LEADER代码，未证明在集成已选定的开源组件；可能是新算法实现。已有sub_098/sub_105映射不能据此判错。'),
 ('sub_108','episode_b25023','evt_faeb4d9ed8cc12fe9283487a','其权重开源了吗','相邻需求','询问预训练编码器能否复用和权重是否开源，属于可获取性/方案讨论，尚未明确执行选定组件的接口集成。'),
 ('sub_128','episode_05eb634','evt_b0fb61d6fdaaeb33554435dc','申请四张a800开始跑','疑似粒度漏标','用户要求四卡启动训练，已有sub_073资源获取与sub_129执行实验。需进一步核对四卡同一训练作业还是独立并行实验，不能把四卡资源请求自动等同于分布式训练。'),
 ('sub_128','episode_c2a0f2','evt_74cb5dace3f70bd0c8683113','充分发挥多卡优势加速实验进程','疑似粒度漏标','用户要求多卡加速实验，存在跨设备运行需求；实际并行方式仍需核查。该证据反对仅凭标签零命中直接删除。'),
 ('sub_207','episode_b65e63','evt_23fd9ddc1e5722d51e05033e','然后上传huggingface（带时间戳以区分）','明确漏标建议','用户明确要求代码/数据包上传指定渠道并以时间戳区分版本；现有同一要求只映射sub_206，而sub_206边界明确不执行发布。应保留sub_206并新增sub_207，不以是否上传成功决定需求是否存在。'),
 ('sub_207','episode_b65e63','evt_adfe27639aced77b2cde36fb','可以的，上传','上下文确认','用户再次确认执行上传，支持前述发布需求；这是需求判断，不声称实际上传已成功。'),
 ('sub_207','episode_1facabe','evt_2ce0eb3c8ae8d8b379bd7405','我会下载后自己上传github','反例','上传由用户自己执行，助手任务是整理打包，不能将此归为助手被要求发布。'),
]

DECISIONS=[
 ('sub_047','证据不足，暂保留','未找到足以确认新增随机化设计的正例；当前例子涉及论文报告既有方法，且有明确排除RCT的反例。','可探索将随机化控制作为sub_056实验流程的结构化属性，但需扩写定义；不与sub_050重复实验等同。','需要从实验设计原文中抽取随机化正例，验证取消独立项是否损失分配/顺序控制的验收标准。'),
 ('sub_051','撤出删除候选，先核查漏标','用户框架明确含参数空间、数据划分、选择准则，不能因训练执行/论文写作标签覆盖而忽略协议层。','暂不归入sub_153搜索超参数组合，后者是执行搜索；若以后重组，可考虑sub_120方法评估协议下的调优协议属性，但涉及跨父需求调整。','区分本轮制定或修改调参协议与执行已有协议；确认后再形成补标。'),
 ('sub_054','撤出删除候选，先核查漏标','已有改变合成数据机制、观察稳定性的具体需求线索，存在被论文写作标签掩盖的可能。','不直接并入sub_055敏感性分析或sub_157执行鲁棒性测试；前者关注参数/假设变化，后者是执行阶段。','核查用户采纳审稿意见的上下文，并区分实验条件设计、结果描述和实际执行。'),
 ('sub_108','证据不足，暂保留','模块适配和询问权重开源并不必然是集成已选定的开源组件，尚无强证据证明漏标或等价冗余。','可评估将sub_105扩写为实现或集成算法模块，并保留实现方式属性；当前sub_105定义没有明确覆盖开源接口适配，不能直接迁移。','需补查已选择组件并要求接入项目的案例，判断许可证、接口和依赖适配是否要独立评估。'),
 ('sub_128','撤出删除候选，核查运行粒度','用户有明确四卡训练和多卡加速要求，原标签可能只记资源获取和一般执行。','如只保留两级，可评估归入sub_129执行实验并增加单卡/多卡/多机属性；若考核跨设备启动与通信，应保留独立项。','确定是单作业分布式训练还是多作业并行，再决定补标；不能用资源数量直接判定。'),
 ('sub_207','保留；提出明确补标','用户要求上传指定渠道并带时间戳，只有打包标签属于漏标；未完成发布不影响发布需求成立。','不与sub_206打包研究资产直接合并，两者可独立验收，且有用户自行上传的明确反例。','对该案例新增发布映射及独立证据判断，保留原打包映射；本轮仅形成提案，不覆盖正式标签或改动统计。'),
]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 source=DOC/'子需求精简诊断_20260921/逐案例命中与来源.jsonl'
 cases=[json.loads(l) for l in source.open() if l.strip()]
 taxpath=ROOT/'src/trace_analysis/pipeline/stage_06_trace_quality_labeling/config/requirement_taxonomy.csv'
 with taxpath.open(encoding='utf-8-sig') as f:tax={r['subrequirement_id']:r for r in csv.DictReader(f)}
 ids={next(c['case_id'] for c in cases if c['case_id'].startswith(pref)) for _,pref,*_ in FINDINGS}
 selected={c['case_id']:c for c in cases if c['case_id'] in ids}
 paths={Path(c['label_path']).parent/'evidence.jsonl' for c in selected.values()}
 evidence={}
 for p in paths:
  for line in p.open():
   r=json.loads(line)
   if r['case_id'] in selected and Path(selected[r['case_id']]['label_path']).parent==p.parent:evidence[r['case_id']]=r
 records=[]
 for sid,prefix,eid,needle,status,reason in FINDINGS:
  c=next(c for c in selected.values() if c['case_id'].startswith(prefix))
  e=next(e for e in evidence[c['case_id']]['events'] if e['event_id']==eid)
  assert e['role']=='user' and e['body_provided'] and needle in e['content']
  start=e['content'].index(needle)
  quote=e['content'][max(0,start-100):min(len(e['content']),start+len(needle)+350)]
  records.append(dict(subrequirement_id=sid,name=tax[sid]['subrequirement_name'],
    case_id=c['case_id'],goal=c['goal'],batch=c['batch'],session_ids=c['session_ids'],
    event_id=eid,turn_id=e['turn_id'],locator=e['locator'],source_role='user',quote=quote,
    label_path=c['label_path'],label_line=c['label_line'],existing_subrequirements=c['subrequirements'],
    existing_mapping_requirements={k:[m['requirement_text'] for m in v] for k,v in c['mappings'].items()},
    review_status=status,reason=reason,reviewer='codex-local-semantic-review',
    human_reviewed=False,applied_to_labels=False))
 def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 dump('逐条语义核查记录.json',records)
 decision_rows=[dict(子需求ID=sid,子需求=tax[sid]['subrequirement_name'],核查后建议=decision,
     依据=reason,精简方向及前提=option,下一步=nextstep) for sid,decision,reason,option,nextstep in DECISIONS]
 with (OUT/'六项核查结论.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(decision_rows[0]));w.writeheader();w.writerows(decision_rows)
 stats_path=DOC/'子需求精简统计决策_20260921/子需求统计决策表.csv'
 with stats_path.open(encoding='utf-8-sig') as f:full_rows=list(csv.DictReader(f))
 updates={r['子需求ID']:r for r in decision_rows}
 for r in full_rows:
  u=updates.get(r['子需求ID'])
  r['最新行动建议']=u['核查后建议'] if u else r['统计决策类别']
  r['本轮语义核查依据']=u['依据'] if u else '本轮未新增语义核查，沿用统计候选状态'
  r['本轮语义核查状态']='Codex本地复核，非人工复核' if u else '未开展'
  r['后续操作']=u['下一步'] if u else r['尚缺证据']
 with (OUT/'216项决策表_语义核查更新.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(full_rows[0]));w.writeheader();w.writerows(full_rows)
 proposal=next(r for r in records if r['subrequirement_id']=='sub_207' and r['review_status']=='明确漏标建议')
 dump('发布研究资产补标提案.json',dict(case_id=proposal['case_id'],action='add_supported_mapping_after_formal_validation',
   proposed_taxonomy_row=207,subrequirement_id='sub_207',parent_requirement_id='req_033',
   keep_existing_mapping='sub_206',evidence_events=[r['event_id'] for r in records if r['subrequirement_id']=='sub_207' and r['case_id']==proposal['case_id']],
   reason=proposal['reason'],status='proposal_not_applied',needs='绑定现有具体需求，补充claim/evidence关联，校验标签并重算统计；不重跑模型或改写历史快照'))
 summary=dict(reviewed_labels=6,selected_evidence_records=len(records),selected_cases=len(selected),
   confirmed_missing_label_cases=1,suspected_or_scope_cases=True,approved_deletions=0,
   applied_label_changes=0,reviewer='codex-local-semantic-review',human_reviewed=False)
 dump('核查摘要.json',summary)
 lines=['# 六项零命中子需求：第一轮语义核查','',
   '**结论：本轮不建议直接删除这6项。发布研究资产有明确漏标证据；调优协议、稳健性实验、多卡调度需先核查漏标/粒度；随机化与开源组件目前证据不足。**','',
   '这份结论更新了统计决策版中6项“删除候选”的行动建议。原统计仍准确描述已有标签，但不能据此估计无漏标的真实需求出现率。统计显著低频不能校正系统性漏标。','',
   '## 核查范围','',
   '对1129个去重任务片段的证据包扫描了11009条episode内用户事件，所有这些事件均有正文。六组关键词用于召回，不是需求判定；对其中有代表性的13条证据、11个案例进行了本地模型语义判断，未宣称逐案读完所有检索结果。关键词漏召回、文档附件内需求及上下文省略表达仍可能存在。','',
   '核查不是随机抽样，不能由这些案例估计总体漏标率，不能对未找到正例的项计算“真实零需求”置信上界。引用是用户原文片段；助手自行提出的方案未升级为用户要求。全部判断由Codex本地核查，未经人工复核；没有修改正式标签、候选清单或分类表。','',
   '## 六项结论（原顺序）','',
   '| ID | 子需求 | 核查后建议 | 主要依据 |','|---|---|---|---|']
 for r in decision_rows:lines.append('| '+' | '.join(r[k] for k in ['子需求ID','子需求','核查后建议','依据'])+' |')
 for sid,decision,reason,option,nextstep in DECISIONS:
  t=tax[sid];lines+=['',f'## {sid} {t["subrequirement_name"]}','',f'原定义：{t["definition"]}','',f'原边界：{t["boundary"]}','',f'建议：{decision}。{reason}','',f'精简路径：{option}','',f'下一步：{nextstep}','']
  for r in records:
   if r['subrequirement_id']!=sid:continue
   loc=r['locator'];lines += [f'### {r["review_status"]}：{r["case_id"]}','',
     f'批次{r["batch"]}；事件`{r["event_id"]}`。 [用户原始记录]({loc["source_file"]}:{loc["physical_line"]}) · [已有标签]({r["label_path"]}:{r["label_line"]})','',
     '> '+r['quote'].replace('\n','\n> '),'',r['reason'],'',
     '现有明确子需求：'+', '.join(r['existing_subrequirements'])+'。','']
 lines+=['## 对精简规则的修正','',
   '在任何统计删项门槛之前增加“漏标反证检查”：只要发现定义清晰、证据明确却未命中的案例，该项退出自动删除通道，先修复或核实打标，再重算频率。不能用已观察到的0次模型命中直接推断真实需求罕见。','',
   '对拟合并项还要检查独立验收反例：打包与发布有明确的“由用户自己上传”反例，因此不能仅因同一任务经常打包后上传就合并。设计与执行、组件检索与集成也需分开检验。','',
   '## 交付文件','',
   '- [216项完整决策表（原顺序、含本轮语义更新）](216项决策表_语义核查更新.csv)',
   '- [六项核查结论CSV](六项核查结论.csv)','- [逐条核查证据JSON](逐条语义核查记录.json)',
   '- [发布研究资产补标提案](发布研究资产补标提案.json)','- [关键词召回候选](用户原文检索候选.jsonl)',
   '- [检索覆盖](检索覆盖.json)','- [核查摘要](核查摘要.json)','- [核查来源与校验](核查manifest.json)','']
 # Counts in prose are derived, not manually assumed.
 lines=[x.replace('13条证据、11个案例',f'{len(records)}条证据、{len(selected)}个案例') for x in lines]
 (OUT/'六项零命中语义核查报告.md').write_text('\n'.join(lines))
 paths.update(Path(c['label_path']) for c in selected.values());paths.update([source,taxpath,stats_path])
 dump('核查manifest.json',dict(generated_at=datetime.now(timezone.utc).isoformat(),summary=summary,
   generator_sha256=sha(Path(__file__)),inputs={str(p):sha(p) for p in paths},
   outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='核查manifest.json'}))
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
