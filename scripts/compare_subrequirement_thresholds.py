"""Compare incidence cutoffs and explicit, unapplied taxonomy compression drafts."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, hashlib
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/高质量Trace筛选_20260915'
OUT=DOC/'子需求收拢阈值对照_20260921'
THRESHOLDS=[1,2,3]
HOLD={'sub_047','sub_051','sub_054','sub_108','sub_128','sub_207'}
# These are editorial proposals grounded in definitions, NOT empirical equivalence claims.
GROUPS=[
 ('G01',['sub_035','sub_036','sub_037','sub_038','sub_040'],None,'估算研究资源需求与成本',
  '资源类型：数据/计算/外部服务/人员/周期；数量、费用、时间、估算依据',
  '共同动作是基于任务与条件估算资源；不同资源维度转为属性。','不能把采购、配置、执行或详细排期并入估算；丢弃属性会失去资源类型的独立评估。'),
 ('G02',['sub_081','sub_082','sub_083'],None,'清理研究数据',
  '处理类型：缺失/异常/重复；识别规则、处置策略、前后数量',
  '三项都是按规则识别和处理数据问题；须新建上位子项，原定义互相排斥。','缺失填补、异常处理、去重不等价；脱敏、类别平衡和数据泄漏不纳入。'),
 ('G03',['sub_089','sub_090','sub_091'],None,'选择与精简研究特征',
  '选择依据：冗余/预测或领域有效性/组间差异；阈值和保留集合',
  '共同产物是按明确标准保留的特征集合；原sub_090排除仅按冗余选择，不能直接塞入其现有定义。','不合并特征提取或重新执行组间统计；无选择依据属性会掩盖不同判断标准。'),
 ('G04',['sub_115','sub_116','sub_117'],None,'编写代码测试',
  '测试层级：单元/集成/端到端；验证对象、入口、断言与覆盖',
  '共同动作是设计并实现测试，层级和范围可作为属性。','保留sub_118执行回归测试；编写与运行、测试与修复不混为一项。'),
 ('G05',['sub_133','sub_134'],None,'记录实验观测',
  '记录类型：指标/日志；运行身份、时间、存储位置',
  '共同动作是保存实验输出或状态；按记录类型区分。','不合并sub_135日志解析；丢弃类型和运行身份会损失指标追溯要求。'),
 ('G06',['sub_159','sub_160','sub_161'],None,'评估方法泛化表现',
  '评估条件：同域跨数据集/跨领域/分布外；训练测试关系、偏移定义',
  '共同动作是评估迁移或泛化表现；不同适用范围必须显式保留。','原三项边界彼此区分，需要新定义；不能将普通随机划分测试直接当作泛化。'),
 ('G07',['sub_185','sub_186'],None,'核验论文与实现的一致性',
  '核验对象：方法步骤/符号索引参数；论文位置、代码位置、差异',
  '共同动作是对照论文与实际实现；核验对象作为属性。','保留逐项差异证据；不能扩展为一般文字润色或性能评估。'),
 ('G08',['sub_106'],'sub_105','实现算法模块与变体',
  '实现形式：单模块/多个可独立选择的变体；差异规格与开关',
  '可扩写sub_105的单模块定义以覆盖多变体实现；当前定义不是已确认无损承接。','保留多个变体的独立选择与规格；不纳入性能比较，不借机合并待核查的开源组件集成。'),
 ('G09',['sub_074','sub_075'],None,'理解数据结构与字段',
  '动作：结构解析/字段解释；层级、类型、单位、含义和来源',
  '可建立数据理解的较宽子项，但结构解析与语义解释仍须区分。','两项并非同义；仅知道字段结构不等于理解语义，属性不能省略。'),
 ('G10',['sub_176'],'sub_177','生成研究结果图表',
  '呈现类型：数据表/图形；数值来源、编码方式、单位与标注',
  '可扩写sub_177并更名，覆盖表格与图形两种结果呈现。','原始数据表和图形不是同义；不能把论文排版、结果分析或不具备数据依据的配图并入。'),
]

def readcsv(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def writecsv(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):
 def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
 return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(cell(x) for x in r)+' |' for r in rows]

def main():
 rp=DOC/'六项零命中语义核查_20260921/216项决策表_语义核查更新.csv'
 cp=DOC/'子需求精简诊断_20260921/逐案例命中与来源.jsonl'
 tp=ROOT/'src/trace_analysis/pipeline/stage_06_trace_quality_labeling/config/requirement_taxonomy.csv'
 ps=[rp,cp,tp];inputs={str(p):sha(p) for p in ps}
 rows=readcsv(rp); tax={r['subrequirement_id']:r for r in readcsv(tp)}
 cases=[c for c in map(json.loads,cp.open()) if c['counted_in_primary']]
 n=len(cases);assert n==1129
 ids=[r['子需求ID'] for r in rows];byid={r['子需求ID']:r for r in rows}
 hits={sid:{c['case_id'] for c in cases if sid in c['subrequirements']} for sid in ids}
 assert all(len(hits[sid])==int(byid[sid]['明确命中案例数']) for sid in ids)
 parents=defaultdict(list)
 for sid in ids:parents[tax[sid]['requirement_id']].append(sid)
 protected={r['子需求ID'] for r in rows if r['暂定保护规则命中']=='True'}
 blocked=HOLD|protected
 def union(sids):return set().union(*(hits[s] for s in sids)) if sids else set()
 summaries=[];lists=[];parentrows=[];case_rows=[];drafts=[]
 for threshold in THRESHOLDS:
  selected={sid for sid in ids if len(hits[sid])*100<n*threshold}
  eligible=selected-blocked
  affected=union(selected)
  affected_s={s for c in cases if c['case_id'] in affected for s in c['session_ids']}
  isolated={c['case_id'] for c in cases if c['subrequirements'] and set(c['subrequirements'])<=selected}
  all_low_parents=[p for p,ss in parents.items() if set(ss)<=selected]
  for p,ss in parents.items():
   low=set(ss)&selected
   parentrows.append(dict(阈值百分比=threshold,父需求ID=p,父需求=tax[ss[0]]['requirement_name'],
       子项总数=len(ss),阈值内子项数=len(low),全部子项均低于阈值=set(ss)<=selected,
       父需求命中案例数=len(union(ss)),阈值内子项涉及案例数=len(union(low)),
       子项ID=';'.join(s for s in ss if s in low)))
  for c in cases:
   matched=[sid for sid in c['subrequirements'] if sid in selected]
   if matched:
    case_rows.append(dict(threshold_percent=threshold,case_id=c['case_id'],batch=c['batch'],
      session_ids=c['session_ids'],affected_subrequirements=matched,
      retained_subrequirements=[sid for sid in c['subrequirements'] if sid not in selected],
      would_lose_all_supported_labels=c['case_id'] in isolated,label_path=c['label_path'],label_line=c['label_line']))
  active=[]
  for gid,sources,anchor,name,attrs,reason,risk in GROUPS:
   if not set(sources)<=eligible:continue
   members=sources+([anchor] if anchor else [])
   assert len({tax[s]['requirement_id'] for s in members})==1
   involved=union(members);net=len(sources)-(0 if anchor else 1)
   active.append(dict(阈值百分比=threshold,方案ID=gid,原子需求=';'.join(sources),
    原子需求名称=';'.join(tax[s]['subrequirement_name'] for s in sources),
    承接已有ID=anchor or '',建议名称=name,操作='扩写已有定义并更名' if anchor else '新建较宽子项并保留类型属性',
    原子项数=len(members),调整后子项数=1,拟减少子项数=net,
    涉及去重案例数=len(involved),包括已有承接项案例数=len(hits[anchor]) if anchor else 0,
    必须保留的属性=attrs,定义依据=reason,信息损失风险=risk,
    语义等价已确认=False,实际已执行=False))
  drafts.extend(active)
  retired={s for d in active for s in d['原子需求'].split(';')}
  existing_anchors={d['承接已有ID'] for d in active if d['承接已有ID']}
  assert len(retired)==sum(len(d['原子需求'].split(';')) for d in active)
  assert not retired&existing_anchors
  net=sum(d['拟减少子项数'] for d in active)
  draftcases=union(retired|existing_anchors)
  for r in rows:
   sid=r['子需求ID'];d=next((d for d in active if sid in d['原子需求'].split(';')),None)
   gate='未进入该阈值范围' if sid not in selected else '进入范围，承接方案待核查'
   if sid in selected and sid in HOLD:gate='六项语义核查暂缓，不能按频率直接收拢'
   elif sid in selected and sid in protected:gate='价值保护项，单独判断'
   elif d:gate=d['操作']+'（草案）'
   lists.append(dict(阈值百分比=threshold,原顺序=r['原顺序'],子需求ID=sid,子需求=r['子需求'],
    父需求ID=r['父需求ID'],父需求=r['父需求'],命中数=len(hits[sid]),
    命中率百分比=round(len(hits[sid])/n*100,4),进入阈值范围=sid in selected,
    分流=gate,方案ID=d['方案ID'] if d else '',承接项=d['建议名称'] if d else '',
    已确认可直接沿用现有定义承接=False,
    定义依据=d['定义依据'] if d else '尚无经语义核查确认的承接结论',
    需保留区别=d['必须保留的属性'] if d else r['边界'],
    原有语义行动建议=r['最新行动建议']))
  summaries.append(dict(阈值百分比=threshold,最大命中次数=max(k for k in range(n+1) if k*100<n*threshold),
    进入范围子需求数=len(selected),占216项百分比=round(len(selected)/216*100,2),
    零命中项数=sum(not hits[s] for s in selected),非零低频项数=sum(bool(hits[s]) for s in selected),
    涉及去重案例数=len(affected),占1129案例百分比=round(len(affected)/n*100,2),
    涉及去重会话数=len(affected_s),盲删后失去全部明确标签案例数=len(isolated),
    盲删后失去全部子项父需求数=len(all_low_parents),
    六项核查暂缓数=len(selected&HOLD),价值保护项数=len((selected&protected)-HOLD),
    排除暂缓保护后待评估子项数=len(eligible),
    已确认直接承接项数=0,具体草案组数=len(active),草案涉及待撤销旧子项数=len(retired),
    草案新建子项数=sum(not d['承接已有ID'] for d in active),
    草案扩写已有子项数=sum(bool(d['承接已有ID']) for d in active),
    草案净减少子项数=net,草案后主子项数=216-net,
    草案涉及去重案例数=len(draftcases),尚无具体收拢方案子项数=len(eligible-retired)))
 OUT.mkdir(parents=True,exist_ok=False)
 writecsv('三档阈值对照.csv',summaries)
 writecsv('三档216项逐项决策.csv',lists)
 for t in THRESHOLDS:writecsv(f'阈值{t}pct子需求表.csv',[r for r in lists if r['阈值百分比']==t])
 writecsv('父需求覆盖风险.csv',parentrows)
 writecsv('具体收拢草案.csv',drafts)
 (OUT/'各档受影响案例.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in case_rows))
 # Reversible migration preview: never alter source labels; preserve every original tag as a facet.
 previews=[]
 for t in THRESHOLDS:
  active=[d for d in drafts if d['阈值百分比']==t]
  mapping={sid:('draft_'+d['方案ID'] if not d['承接已有ID'] else d['承接已有ID'])
    for d in active for sid in d['原子需求'].split(';')}
  for c in cases:
   if not set(c['subrequirements'])&set(mapping):continue
   grouped=defaultdict(list)
   for sid in c['subrequirements']:grouped[mapping.get(sid,sid)].append(sid)
   assert set(s for ss in grouped.values() for s in ss)==set(c['subrequirements'])
   previews.append(dict(threshold_percent=t,case_id=c['case_id'],
    original_subrequirements=c['subrequirements'],proposed_primary_labels=list(grouped),
    retained_original_ids_by_label=dict(grouped),status='preview_only_not_applied'))
 (OUT/'逐案例迁移预览_未应用.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previews))
 md=['# 子需求收拢：1% / 2% / 3% 阈值对照','',
  '结论：阈值用于圈定核查范围，不等于删除数量。当前1%已纳入约三分之二子项；先用1%推进属性化收拢，再评估扩大范围，比直接提高阈值更可控。','',
  '## 数据与计算口径','',
  '- 使用已有1136条有效标签中去重后的1129个任务片段；一案例一子需求计一次。严格小于阈值，1%为0–11次，2%为0–22次，3%为0–33次。',
  '- 统计描述已有明确标签，不是校正漏标后的真实需求率。已发现的发布漏标尚未写入正式数据；六项语义核查项全部暂缓自动收拢。',
  '- 涉及案例数是至少命中一个阈值内子项的案例并集，不能把各子项次数相加。零命中项增加待评估项数，但不增加已标案例覆盖。',
  '- 价值保护规则沿用上一版的req_036及权限、合规、脱敏、污染、回滚、伦理等，不是“有这些词就永久保留”的最终政策。',
  '- 直接承接已确认项数为0：此前并未完成全量语义等价审核，不能用共现代替承接证明。具体草案是根据定义提出的新宽类或定义扩写，不是假定已有类别天然包含它们。','',
  '## 三档对照','']
 md+=table(['阈值','纳入子项','涉及案例','涉及会话','盲删后无标签案例','全部子项落入范围的父需求','排除暂缓/保护后待评估'],
  [[f'{s["阈值百分比"]}%',s['进入范围子需求数'],s['涉及去重案例数'],s['涉及去重会话数'],s['盲删后失去全部明确标签案例数'],s['盲删后失去全部子项父需求数'],s['排除暂缓保护后待评估子项数']] for s in summaries])
 md+=['','“无标签”只指删掉这些现有明确映射后的机械结果，不表示真实任务不再存在；未把本来无明确标签的案例计入这一损失。','',
  '## 已写出的具体收拢草案及数量','']
 md+=table(['阈值','草案组数','撤销旧子项','新建子项','扩写已有项','净减少','主子项从216变为','涉及案例'],
  [[f'{s["阈值百分比"]}%',s['具体草案组数'],s['草案涉及待撤销旧子项数'],s['草案新建子项数'],s['草案扩写已有子项数'],s['草案净减少子项数'],s['草案后主子项数'],s['草案涉及去重案例数']] for s in summaries])
 md+=['','这不是可精简数量的上限，也不是已批准改动；只是当前逐一定义审阅后写出的10组草案。其他低频项仍未确定承接方式，不能算作可删除。','',
  '**主标签减少，不代表最细能力维度或标注工作同比减少。** 草案保留必要类型属性和原子需求ID，可恢复历史映射；如果未来不再采集这些属性，就会丢失原本可单独评估的区别。迁移预览验证的是标签映射可逆，不证明新标注准确率或语义无损。','',
  '## 具体方案（按最早原子需求顺序）','']
 for gid,sources,anchor,name,attrs,reason,risk in GROUPS:
  ds=[d for d in drafts if d['方案ID']==gid]
  if not ds:continue
  d=ds[0];md += [f'### {gid} {name}','',
   '涉及：'+'、'.join(s+' '+tax[s]['subrequirement_name'] for s in sources)+('；扩写承接项：'+anchor+' '+tax[anchor]['subrequirement_name'] if anchor else '')+'。','',
   f'最早进入：{d["阈值百分比"]}%档；{d["原子项数"]}→1，拟减少{d["拟减少子项数"]}项。','',
   '定义依据：'+reason,'','必须保留：'+attrs,'','风险与边界：'+risk,'']
  md+=table(['原ID','原名称','命中数','原定义','原边界'],[[s,tax[s]['subrequirement_name'],len(hits[s]),tax[s]['definition'],tax[s]['boundary']] for s in sources+([anchor] if anchor else [])])
  examples=sorted(union(sources))[:2]
  for cid in examples:
   c=next(c for c in cases if c['case_id']==cid)
   md+=['',f'定位样例：[原标签]({c["label_path"]}:{c["label_line"]})，`{cid}`：{c["goal"]}。']
  md+=['']
 md+=['## 如何选择阈值','',
  '建议先用1%作为收拢核查入口，优先讨论资源估算、数据清理、代码测试、泛化评估等动作相同而对象/条件不同的属性化方案。阈值提高到2%或3%会进一步触及常用理解、写作和结果呈现能力，不宜整体搬迁。', '',
  '具体执行前：为新宽类写明定义、排除项和属性；用原文正例与反例检查迁移；在新数据上验证无法归类率、争议率和独立评估能力。原统计阈值不需要因操作名称变为收拢就伪装成统计显著性结论。','',
  '## 文件','',
  '- [三档汇总CSV](三档阈值对照.csv)','- [1%逐项表（216项原顺序）](阈值1pct子需求表.csv)',
  '- [2%逐项表（216项原顺序）](阈值2pct子需求表.csv)','- [3%逐项表（216项原顺序）](阈值3pct子需求表.csv)',
  '- [具体收拢草案](具体收拢草案.csv)','- [父需求覆盖风险](父需求覆盖风险.csv)',
  '- [受影响案例JSONL](各档受影响案例.jsonl)','- [可逆迁移预览，未应用](逐案例迁移预览_未应用.jsonl)',
  '- [来源与校验](manifest.json)','']
 (OUT/'子需求收拢阈值对照报告.md').write_text('\n'.join(md))
 for t in THRESHOLDS:assert [r['子需求ID'] for r in lists if r['阈值百分比']==t]==ids
 assert all(sha(Path(p))==h for p,h in inputs.items())
 dump('manifest.json',dict(generated_at=datetime.now(timezone.utc).isoformat(),inputs=inputs,
  generator_sha256=sha(Path(__file__)),summary=summaries,
  validation=dict(original_order=True,case_counts_verified=True,preview_original_labels_preserved=True,applied_changes=0),
  outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
 print(json.dumps(summaries,ensure_ascii=False))

if __name__=='__main__':main()
