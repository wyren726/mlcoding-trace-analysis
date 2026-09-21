"""Retrieve user evidence for six zero-hit labels; retrieval is not relabeling."""
from pathlib import Path
from collections import Counter
import json,re,hashlib

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/高质量Trace筛选_20260915/子需求精简诊断_20260921/逐案例命中与来源.jsonl'
OUT=ROOT/'docs/高质量Trace筛选_20260915/六项零命中语义核查_20260921'
PATTERNS={
 'sub_047':r'随机分[配组]|随机化|随机对照|随机顺序|分层随机|randomi[sz]|random\s+assign|random\s+allocation',
 'sub_051':r'超参数|调参|调优|网格搜索|贝叶斯优化|hyperparam|grid\s*search|optuna|search\s*space',
 'sub_054':r'稳健|鲁棒|robust|扰动|分布偏移|distribution\s*shift|抗噪',
 'sub_108':r'开源|第三方|集成|组件|接入|适配|integrat|open.source|github|huggingface',
 'sub_128':r'分布式|多卡|多机|多GPU|多\s*GPU|DDP|torchrun|deepspeed|distributed|multi.gpu',
 'sub_207':r'发布|上传|推送|开源|publish|upload|push|zenodo|huggingface|github',
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def redact(text):
 return re.sub(r'\b(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}', '[REDACTED_TOKEN]', text)

def main():
 cases={c['case_id']:c for c in map(json.loads,SOURCE.open()) if c['counted_in_primary']}
 paths=sorted({Path(c['label_path']).parent/'evidence.jsonl' for c in cases.values()})
 found=[];coverage=[];inputs={str(SOURCE):sha(SOURCE)}
 for p in paths:
  inputs[str(p)]=sha(p)
  for line in p.open():
   r=json.loads(line);cid=r['case_id']
   if cid not in cases:continue
   # Each case belongs to one selected label snapshot only.
   if Path(cases[cid]['label_path']).parent!=p.parent:continue
   events=r['events'];users=[e for e in events if e['role']=='user' and e.get('episode_membership')=='episode']
   coverage.append(dict(case_id=cid,user_events=len(users),provided_user_events=sum(bool(e.get('body_provided')) for e in users),scope=r['scope']))
   for sid,pattern in PATTERNS.items():
    matches=[]
    for e in users:
     if not e.get('body_provided'):continue
     text=e.get('content','')
     ms=list(re.finditer(pattern,text,re.I))
     if not ms:continue
     matches.append(dict(event_id=e['event_id'],turn_id=e['turn_id'],locator=e['locator'],
       content_chars=len(text),keywords=sorted({m.group() for m in ms}),
       snippets=[redact(text[max(0,m.start()-200):min(len(text),m.end()+350)]) for m in ms[:4]]))
    if matches:
     c=cases[cid]
     found.append(dict(subrequirement_id=sid,case_id=cid,batch=c['batch'],goal=c['goal'],
        session_ids=c['session_ids'],label_path=c['label_path'],label_line=c['label_line'],
        evidence_path=str(p),existing_subrequirements=c['subrequirements'],user_matches=matches))
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'用户原文检索候选.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in found))
 summary=dict(cases=len(coverage),user_events=sum(c['user_events'] for c in coverage),
   provided_user_events=sum(c['provided_user_events'] for c in coverage),
   keyword_candidate_cases=dict(Counter(r['subrequirement_id'] for r in found)),
   note='关键词召回，不是命中判定；不用于估计漏标率。')
 (OUT/'检索覆盖.json').write_text(json.dumps(dict(summary=summary,cases=coverage),ensure_ascii=False,indent=2))
 (OUT/'检索manifest.json').write_text(json.dumps(dict(inputs=inputs,patterns=PATTERNS,
   generator_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='检索manifest.json'}),ensure_ascii=False,indent=2))
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
