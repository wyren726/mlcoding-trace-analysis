"""Retrieve zero-label counterexamples; do not interpret absence as absence of need."""
from pathlib import Path
from collections import defaultdict
import csv,json,re,hashlib
from audit_six_zero_hit_requirements import redact

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/高质量Trace筛选_20260915'
OUT=DOC/'145项子需求逐项核查_20260921'
PATTERNS={
36:r'显存.{0,12}(够|需要)|需要.{0,12}(显存|算力)|预[算估].{0,15}(GPU|显卡|计算)|GPU.{0,12}(成本|费用)|训练.{0,8}(成本|费用)',
38:r'工时|人天|人月|人员投入|需要.{0,8}(人手|人员)|人工成本',
41:r'(比较|对比|选|哪种|哪个).{0,20}(显卡|GPU|服务器|资源方案)|性价比|资源.{0,8}方案',
42:r'资源估算|预算报告|成本估算|经费预算|预算表',
66:r'Dockerfile|构建.{0,8}(镜像|容器)|打包.{0,8}(镜像|容器)|制作.{0,8}镜像|docker\s+build',
80:r'脱敏|匿名化|去标识|去除.{0,8}(姓名|身份)|隐私信息',
84:r'SMOTE|过采样|欠采样|类别.{0,8}(不平衡|失衡)|class.weight|重采样',
91:r'筛选.{0,12}(差异|基因|蛋白)|差异.{0,10}(筛选|阈值)|按.{0,8}p值',
94:r'提取.{0,12}(核心|代码)|核心代码|代码片段|摘取.{0,10}代码',
110:r'代码.{0,12}(修改计划|改动计划)|修改.{0,8}大纲|先.{0,10}(计划|方案).{0,10}(代码|修改)|列出.{0,8}(大纲|修改)',
116:r'集成测试|接口测试|联调|integration.test',
118:r'回归测试|regression.test|修改前后|原有功能.{0,12}(正常|测试)|不破坏.{0,12}(功能|行为)',
123:r'跨模型|不同.{0,5}模型.{0,12}(配置|评测)|多模型.{0,12}(评测|配置)|统一.{0,8}评测',
124:r'生效.{0,5}(配置|参数)|配置继承|参数覆盖|总结.{0,8}(配置|参数)|配置.{0,8}汇总',
130:r'断点续训|断点恢复|恢复训练|续训|接着.{0,10}训练|resume',
131:r'检查点|checkpoint|保存.{0,8}(模型|权重)|best.model',
133:r'(记录|保存).{0,12}(指标|metric)|指标.{0,12}(保存|记录)',
135:r'日志.{0,15}(提取|指标|曲线)|提取.{0,12}日志|从.{0,5}日志',
145:r'(是否|有没有|还有没有).{0,10}(继续|必要).{0,8}(跑|训练)|停止.{0,8}(建议|时机)|有没有继续跑的必要',
149:r'回滚.{0,12}(模型|检查点|checkpoint|训练)|恢复.{0,12}(权重|优化器|checkpoint)',
152:r'数据.{0,5}(污染|泄漏)|训练.{0,10}测试.{0,10}(重叠|泄漏)|数据泄露|data.leak',
155:r'(哪些|什么|哪个).{0,15}(参数|学习率|配置).{0,8}(调|改)|可调参数|调整.{0,5}(学习率|batch|批次)|batch.size',
160:r'跨领域|跨域|cross.domain|迁移到.{0,8}(领域|任务)',
161:r'分布外|\bOOD\b|out.of.distribution|分布偏移|domain.shift',
164:r'效应量|Cohen|η²|ω²|effect.size',
172:r'负面结果|负结果|如实.{0,10}(结果|汇报)|失败.{0,10}(报告|汇报)|没有.{0,5}提升.{0,12}(写|报告)',
210:r'任务.{0,10}(依赖|阻塞)|前置任务|依赖关系|阻塞关系',
211:r'任务.{0,10}(进度|状态)|进度表|任务清单|已完成.{0,15}未完成',
214:r'记住.{0,15}(要求|约束|目标)|不要忘记|之前的记忆|上下文|恢复记忆',
215:r'先.{0,10}(问我|确认|授权)|需要问我|未经.{0,8}(允许|授权)|征得.{0,8}(同意|授权)',
217:r'回滚方案|回退方案|恢复方案|操作前.{0,10}备份|修改前.{0,10}备份',
}

def main():
 cp=DOC/'子需求精简诊断_20260921/逐案例命中与来源.jsonl'
 cases={c['case_id']:c for c in map(json.loads,cp.open()) if c['counted_in_primary']}
 inputs={str(cp):hashlib.sha256(cp.read_bytes()).hexdigest()}
 found=[];seen=0
 for p in sorted({Path(c['label_path']).parent/'evidence.jsonl' for c in cases.values()}):
  inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for line in p.open():
   r=json.loads(line);cid=r['case_id']
   if cid not in cases or Path(cases[cid]['label_path']).parent!=p.parent:continue
   for e in r['events']:
    if e['role']!='user' or e.get('episode_membership')!='episode' or not e.get('body_provided'):continue
    seen+=1;txt=e['content']
    for num,pat in PATTERNS.items():
     ms=list(re.finditer(pat,txt,re.I))
     if not ms:continue
     c=cases[cid]
     found.append(dict(subrequirement_id=f'sub_{num:03}',case_id=cid,batch=c['batch'],goal=c['goal'],
       existing_subrequirements=c['subrequirements'],event_id=e['event_id'],turn_id=e['turn_id'],
       locator=e['locator'],label_path=c['label_path'],label_line=c['label_line'],
       user_content_chars=len(txt),match_count=len(ms),
       excerpts=[redact(txt[max(0,m.start()-100):min(len(txt),m.end()+220)]) for m in ms[:3]]))
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'31项零命中原文召回.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in found))
 groups=defaultdict(list)
 for r in found:groups[r['subrequirement_id']].append(r)
 shortlist={}
 for num in PATTERNS:
  sid=f'sub_{num:03}';rows=sorted(groups[sid],key=lambda r:(r['user_content_chars']>1500,-r['match_count'],r['user_content_chars'],r['case_id']))
  used=set();picked=[]
  for r in rows:
   if r['case_id'] in used:continue
   picked.append(r);used.add(r['case_id'])
   if len(picked)==3:break
  shortlist[sid]=picked
 (OUT/'零命中定位样例.json').write_text(json.dumps(shortlist,ensure_ascii=False,indent=2))
 (OUT/'召回manifest.json').write_text(json.dumps(dict(inputs=inputs,patterns=PATTERNS,user_events=seen,
   candidate_case_counts={sid:len({r['case_id'] for r in rr}) for sid,rr in groups.items()},
   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
 print('events',seen,'retrieval_records',len(found))
 for sid,rr in shortlist.items():print(sid,len({r['case_id'] for r in groups[sid]}),[(r['case_id'],r['excerpts'][0][:240].replace('\n',' ')) for r in rr[:2]])

if __name__=='__main__':main()
