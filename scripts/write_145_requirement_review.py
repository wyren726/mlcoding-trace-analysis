"""Reproducible first-round structural review; never mutate source labels/taxonomy."""
from pathlib import Path
from collections import Counter
import csv, json, hashlib, re
from audit_six_zero_hit_requirements import redact
from compare_subrequirement_thresholds import GROUPS
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/高质量Trace筛选_20260915'
OUT=DOC/'145项子需求逐项核查_20260921'
def readcsv(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=DOC/'六项零命中语义核查_20260921/216项决策表_语义核查更新.csv'
 cp=DOC/'子需求精简诊断_20260921/逐案例命中与来源.jsonl'
 tp=ROOT/'src/trace_analysis/pipeline/stage_06_trace_quality_labeling/config/requirement_taxonomy.csv'
 np=Path(__file__).with_name('subrequirement_review_notes_20260921.txt')
 rows=readcsv(rp);tax={r['subrequirement_id']:r for r in readcsv(tp)}
 cases={c['case_id']:c for c in map(json.loads,cp.open()) if c['counted_in_primary']}
 notes={}
 for line in np.read_text().splitlines():
  n,status,note=line.split('|',2);notes[f'sub_{int(n):03}']=(status,note)
 selected=[r for r in rows if int(r['明确命中案例数'])/len(cases)<.01]
 assert len(cases)==1129 and len(selected)==145
 assert set(notes)=={r['子需求ID'] for r in selected}
 recall=list(map(json.loads,(OUT/'31项零命中原文召回.jsonl').open()))
 shortlist=json.loads((OUT/'零命中定位样例.json').read_text())
 old=json.loads((DOC/'六项零命中语义核查_20260921/逐条语义核查记录.json').read_text())
 strong={
 'sub_080':('episode_ad05ba7b26a1b54ea1242333__line_1101','evt_ba5a91894006908ba992616c'),
 'sub_131':('episode_05eb6344387d04a01efc2cbb__line_16','evt_b0fb61d6fdaaeb33554435dc'),
 'sub_164':('episode_95b242ba6285d54038015fab__line_123','evt_081bb8b7e11bbad6ed88b773'),
 'sub_207':('episode_b65e63a75def5948a7e739a3__line_75','evt_23fd9ddc1e5722d51e05033e'),
 'sub_215':('episode_7d5a7696c439c21220573e8c__line_195','evt_dfff2f2a70c5dfcde8ea10f7')}
 events={};evidencepaths={Path(cases[cid]['label_path']).parent/'evidence.jsonl' for cid,eid in strong.values()}
 for p in evidencepaths:
  for pack in map(json.loads,p.open()):
   cid=pack['case_id']
   if cid not in {x[0] for x in strong.values()} or Path(cases[cid]['label_path']).parent!=p.parent:continue
   for e in pack['events']:events[(cid,e['event_id'])]=e
 proposals=[]
 for sid,key in strong.items():
  cid,eid=key;c=cases[cid];e=events[key]
  assert e['role']=='user' and e['episode_membership']=='episode' and sid not in c['subrequirements']
  proposals.append(dict(subrequirement_id=sid,name=tax[sid]['subrequirement_name'],case_id=cid,event_id=eid,turn_id=e['turn_id'],source_role=e['role'],locator=e['locator'],quote=redact(e['content']),label_path=c['label_path'],label_line=c['label_line'],existing_subrequirements=c['subrequirements'],reason=notes[sid][1],proposal='新增需求映射；现有映射分别复核，不自动替换',scope_caveat='131仅支持本例存储组织，不证明完整轮换能力；215需统一GPU使用与单实验状态排除边界。',applied=False,human_reviewed=False))
 groupby={sid:g for g in GROUPS[:8] for sid in g[1]}
 assert set(groupby)<=set(notes)
 evidence=[];result=[];groups=[]
 for g in GROUPS[:8]:
  gid,ss,anchor,name,attrs,reason,risk=g
  groups.append(dict(方案ID=gid,原子需求=';'.join(ss),承接已有ID=anchor or '',新名称=name,操作='扩写已有定义并更名' if anchor else '重组为新子需求',保留属性=attrs,依据=reason,损失风险=risk,净减少=len(ss)-(0 if anchor else 1),证据级别='定义相容性草案；未证明统计等价',已应用=False))
 for r in selected:
  sid=r['子需求ID'];status,note=notes[sid];count=int(r['明确命中案例数']);t=tax[sid];refs=[]
  if sid in strong:
   refs=[dict(next(p for p in proposals if p['subrequirement_id']==sid),evidence_type='原文补标建议')]
  elif count==0:
   prior=[p for p in old if p['subrequirement_id']==sid]
   if prior:refs=[dict(p,evidence_type='沿用六项语义核查') for p in prior]
   else:refs=[dict(p,quote=' […] '.join(p['excerpts']),source_role='user',evidence_type='关键词召回；非命中确认') for p in shortlist.get(sid,[])]
  else:
   c=next(c for c in cases.values() if sid in c['subrequirements'])
   ms=c['mappings'].get(sid,[])
   evs=[(m,e) for m in ms for e in m.get('evidence',[])]
   chosen=next(((m,e) for m,e in evs if e.get('source_role')=='user'),evs[0] if evs else None)
   if chosen:
    m,e=chosen
    refs=[dict(subrequirement_id=sid,case_id=c['case_id'],label_path=c['label_path'],label_line=c['label_line'],mapping_id=m.get('mapping_id'),requirement_text=m.get('requirement_text'),event_id=e.get('event_id'),turn_id=e.get('turn_id'),source_role=e.get('source_role'),locator=e.get('locator'),quote=redact(e.get('quote','')),evidence_type='现有映射定位样例；非随机抽样，未逐条重审全部命中')]
  for ref in refs:
   ref['quote']=redact(ref.get('quote',''));evidence.append(ref)
  g=groupby.get(sid)
  if g:
   gid,ss,anchor,name,attrs,reason,risk=g;decision='扩写归入已有项' if anchor else '重组候选';target=(anchor or gid)+' '+name
   basis=reason+' 本项核查：'+note;loss=risk;pre='先修正疑似错标，再改写定义并用边界样例验证；保留原ID到属性映射。'
  else:
   decision='暂保留' if status in {'样例支持','明确补标建议'} or sid in {'sub_030','sub_077','sub_152','sub_172','sub_217'} else '待定'
   target='原子需求';basis=note;loss='若删除，将失去独立识别“'+t['definition']+'”的能力；尚未证实已有项完整承接。'
   pre='复核原文需求主体、对象与动作；核查相邻类别漏标后再决定。' if decision=='待定' else '保留独立验收边界；补标建议单独复核。'
  result.append(dict(原顺序=r['原顺序'],子需求ID=sid,子需求=r['子需求'],父需求=r['父需求'],原始命中案例数=count,原始命中率百分比=r['总体命中率百分比'],命中会话数=r['命中会话数'],分层='零命中' if count==0 else '非零低于1%',结构建议=decision,证据可靠性状态=status,承接目标=target,建议依据=basis,定义=t['definition'],边界=t['boundary'],验收产物=t['outputs'],必须保留属性=g[4] if g else '',删并损失或风险=loss,前置核查=pre,定位样例数=len(refs),案例ID=';'.join(dict.fromkeys(x['case_id'] for x in refs)),事件ID=';'.join(str(x.get('event_id','')) for x in refs),用户或来源摘录='\n'.join(x.get('quote','')[:1400] for x in refs),原文定位='\n'.join(json.dumps(x.get('locator'),ensure_ascii=False) for x in refs),证据覆盖范围='定义与边界首轮评审；零命中为定向召回，非零为固定顺序定位样例；未全面重标',已应用=False))
 assert [r['子需求ID'] for r in result]==sorted(notes)
 writecsv('145项逐项调整建议.csv',result)
 writecsv('37项零命中核查.csv',[r for r in result if r['原始命中案例数']==0])
 writecsv('108项非零低频核查.csv',[r for r in result if r['原始命中案例数']>0])
 writecsv('8组结构重组草案.csv',groups)
 dump('逐项证据定位.json',evidence);dump('5项补标建议_未应用.json',proposals)
 full=[];byid={r['子需求ID']:r for r in result}
 for r in rows:
  rr=byid.get(r['子需求ID']);full.append(dict(r,本轮结构建议=rr['结构建议'] if rr else '本轮范围外，未复核',本轮证据可靠性=rr['证据可靠性状态'] if rr else '未复核',本轮建议依据=rr['建议依据'] if rr else '',本轮承接目标=rr['承接目标'] if rr else '',本轮已应用=False))
 writecsv('216项全表_保留原顺序.csv',full)
 counts=Counter(r['结构建议'] for r in result)
 summary=[]
 for layer in ['零命中','非零低于1%']:
  cc=Counter(r['结构建议'] for r in result if r['分层']==layer)
  summary.append('|'+layer+'|'+'|'.join(str(cc[x]) for x in ['重组候选','扩写归入已有项','暂保留','待定'])+'|')
 report=f'''# 145项子需求首轮精简诊断（2026-09-21）

本轮覆盖全部37项零命中和108项非零且命中率低于1%的子需求，按原分类表顺序逐项给出定义、边界、证据问题和调整建议。这里的“首轮”指完成逐项结构评审及定位样例核查，不代表重新审核全部1129条案例、全部映射或所有原文。正式分类表和打标文件均未修改。

## 当前结论

结构建议合计：{dict(counts)}。直接删除建议为0项：目前发现的零命中并不足以证明需求不存在，也没有逐项证明删除后的独立需求可以无损承接。这与停止精简不同：8组可审阅草案涉及22个低频原子项，另需扩写承接项sub_105；若全部通过，可净减少15个主子需求，216→201。该方案延续上一轮8组草案，本轮新增的是145项逐项依据与证据问题，不是又找到15项新增删减。

|范围|重组候选|扩写归入已有项|暂保留|待定|
|---|---:|---:|---:|---:|
{chr(10).join(summary)}

暂保留表示当前不建议移除，不代表每条现有标签都正确；待定表示独立价值或证据尚未核实，不能解释为待删除。首轮现有映射样例以固定顺序取例，不能从发现的问题推算整体错标率。关键词未召回也不能证明需求不存在。

## 为什么37项全部进入审查，而不是只审查6项

原先的6项是按父需求内部出现率单侧95%上界低于5%筛出的统计候选，不是从原文证实可删除的6项。零命中时，在独立同分布二项模型下，上界为1−0.05^(1/n)：父需求覆盖20个独立样本约13.9%，覆盖59个约4.95%。它回答的是样本是否足以排除5%的发生率，不回答是否值得保留。37项都值得审查，父需求样本少只会降低结论可信度，不应剥夺进入审查的资格。

这仍是模型条件下的上界；当前四批次不是已证明具有总体代表性的随机样本，会话还可能来自同一用户或项目。尤其标签存在漏标、错标时，区间首先描述已记录标签的发生率，不能当作真实需求发生率。当前“真实打标命中率”应明确称为“真实案例上的现有标签命中率”。

## 已定位的补标需求

原文支持以下5项补标建议，均未写回标签；207为此前已发现，其余4项是本轮补充。

|ID|需求|原文依据|
|---|---|---|
|080|脱敏研究数据|访谈匿名化并提前给受访者编码|
|131|组织模型检查点|指定checkpoint独立存储位置并考虑空间、inode约束|
|164|计算效应量|明确重算η²/ω²|
|207|发布研究资产|打包代码数据后上传HuggingFace，另有上传确认|
|215|请求研究确认|下次使用GPU前需要询问用户|

131支持存储组织要求，不据此宣称用户要求完整检查点轮换。215支持操作前征求确认，但仍应统一“GPU资源使用”与定义中“单个实验内部状态”的边界。保留其他映射并逐条复核，不能自动用新增标签替换已有标签。5项建议的案例、事件和物理行定位见JSON附件。

另有疑似错标，例如把液晶物理取值映射成资源估算基线、把材料配比映射成数据资源估算、把KMO提高要求映射成运行性能瓶颈。这些提示需修正标签口径，不能用可疑命中证明类别有价值，也不能用修正后更低的频率直接判定删除。

## 如何精简

删减要求：确认原文确实缺少此类需求或分类定义本身冗余；检查漏标、独立交付和验收价值；列明已有承接项及丢失信息；对伦理、权限、污染、负结果等低频但重要需求单独评估。频率只确定审查顺序，1%不是统计定理，也不是自动删除线。

重组要求：多个子项具有共同动作和产物，差别可明确写成类型属性；重写一个较宽定义，保留原ID映射及属性。资源估算、数据清理、特征选择、编写测试、记录实验观测、泛化评估、论文与实现一致性这7组属于此类。它们并非已被统计证明等价。

归入已有项要求：承接项的定义、产物和验收条件能够覆盖来源项。本轮只有106→105属于扩写后归入：105必须扩写并更名为“实现算法模块与变体”，保留多变体的独立选择和规格；现有105不能原样承接106。

共同出现并不意味着能合并。设计与执行、写测试与跑测试、日志记录与日志解析、打包与发布、请求授权与核验已有权限均继续区分。此前98对BY校正后显著关联只说明共现证据，没有据此确认强等价或包含关系。

统计验证应在修正标签后重算每项的分批、案例及会话频率。合并验证同时报告P(B|A)、P(A|B)、单独命中数量及会话聚类不确定性，并逐条检查无法承接的案例；很小的支持数即使100%共现也不能证明包含。当前不调整阈值来追求预定删除数量。

8组全部成立时减少15个主标签，但如果仍需打类型属性，细粒度标注工作不会等比例减少。若舍弃属性，必须接受表中列明的信息损失。当前建议先以保留属性的方式评审，不直接压平边界。

## 交付文件

- [145项逐项调整建议.csv](145项逐项调整建议.csv)：主表，每项均有建议与依据。
- [37项零命中核查.csv](37项零命中核查.csv)；[108项非零低频核查.csv](108项非零低频核查.csv)。
- [216项全表_保留原顺序.csv](216项全表_保留原顺序.csv)：全表，范围外71项明确未复核。
- [8组结构重组草案.csv](8组结构重组草案.csv)：新定义方向、归入目标、属性与风险。
- [5项补标建议_未应用.json](5项补标建议_未应用.json)：原文支持的补标建议。
- [逐项证据定位.json](逐项证据定位.json)：样例映射与原文位置；召回条目不是正例判定。

后续应优先修正表中明确补标及疑似错标，再用每组原始案例验证承接边界，生成可比较的新旧分类试标结果。现阶段交付的是完整首轮诊断与可审阅的调整草案，不是已完成全量标签清洗或已执行分类修改。
'''
 (OUT/'首轮核查报告.md').write_text(report)
 inputs={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in [rp,cp,tp,np,Path(__file__),OUT/'召回manifest.json',DOC/'六项零命中语义核查_20260921/逐条语义核查记录.json']}
 inputs.update({str(p):sha(p) for p in evidencepaths})
 dump('核查manifest.json',dict(inputs=inputs,cases=len(cases),sessions=940,items=145,zero_items=37,nonzero_items=108,decisions=dict(counts),direct_deletions=0,group_count=8,net_reduction_proposal=sum(g['净减少'] for g in groups),applied=False,limitation='定向检索与固定顺序样例，不是随机审计或全量重标'))
 # Validate owned deliverables without exposing matching content.
 owned=['145项逐项调整建议.csv','37项零命中核查.csv','108项非零低频核查.csv','216项全表_保留原顺序.csv','8组结构重组草案.csv','逐项证据定位.json','5项补标建议_未应用.json','首轮核查报告.md']
 for name in owned:assert not re.search(r'\b(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}',(OUT/name).read_text()),name
 assert len([r for r in result if r['原始命中案例数']==0])==37
 assert len(full)==216 and sum(g['净减少'] for g in groups)==15
 assert all(sha(ROOT/p if not Path(p).is_absolute() else Path(p))==h for p,h in inputs.items())
 print(json.dumps(dict(decisions=dict(counts),evidence_records=len(evidence),proposals=len(proposals),validated=True),ensure_ascii=False))
if __name__=='__main__':main()
