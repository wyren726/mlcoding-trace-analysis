from __future__ import annotations
import glob, html, json, re
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = Path(glob.glob(str(root / 'workspace/analysis-runs/batch_*/run_*/06_trace_quality_labeling/trace-label-model-v3-20260916/labeled_candidates*.jsonl'))[0])
all_rows = [json.loads(x) for x in src.read_text(encoding='utf-8').splitlines() if x.strip()]
rows = [x for x in all_rows if x.get('historical_result', {}).get('pain_judgment') == 'confirmed' and x.get('labels', {}).get('trace_quality') == 'high']

def clean_name(value):
    return re.sub(r'\s*@[^,，;；\r\n]+\s*$', '', str(value or '')).strip()

def clean_obj(value):
    if isinstance(value, str): return clean_name(value)
    if isinstance(value, list): return [clean_obj(x) for x in value]
    if isinstance(value, dict): return {k: clean_obj(v) for k, v in value.items()}
    return value

def goal_text(value):
    if isinstance(value, dict):
        return value.get('goal') or value.get('text') or json.dumps(value, ensure_ascii=False)
    return value or '未提供'

items = []
for i, r in enumerate(rows, 1):
    s, l, reused = r.get('source', {}), r.get('labels', {}), r.get('reused', {})
    mappings = clean_obj(l.get('capability_mapping') or [])
    items.append({
        'index': i, 'line': s.get('source_line_number'),
        'session': (s.get('session_ids') or ['unknown'])[0],
        'trace': s.get('trace_id'), 'episode': s.get('episode_id'),
        'goal': goal_text(reused.get('preliminary_goal') or reused.get('initial_query')),
        'domain': clean_name(l.get('domain_primary') or 'unknown'),
        'stage': l.get('research_stage_primary') or 'unknown',
        'difficulty': l.get('difficulty') or 'unknown',
        'factors': l.get('difficulty_factors') or [], 'failures': l.get('failure_pattern') or [],
        'integrity': l.get('episode_integrity'), 'evidence_status': l.get('evidence_status'),
        'gates': r.get('quality_gates') or {}, 'gate_reasons': r.get('quality_gate_reasons') or {},
        'reason': r.get('label_reasons') or '', 'mappings': mappings,
        'refs': r.get('evidence_refs') or [],
    })
payload = json.dumps(items, ensure_ascii=False).replace('</', '<\\/')
domains = Counter(x['domain'] for x in items)
difficulties = Counter(x['difficulty'] for x in items)
failures = Counter(f for x in items for f in x['failures'])
domain_options = ''.join(f'<option value="{html.escape(k)}">{html.escape(k)} ({v})</option>' for k, v in sorted(domains.items()))
failure_options = ''.join(f'<option value="{html.escape(k)}">{html.escape(k)} ({v})</option>' for k, v in sorted(failures.items()))
out = root / 'workspace/analysis-runs/batch_20260828_162718_1dd984cc/run_20260828_164200_wisland_v1/06_trace_quality_labeling/trace-label-model-v3-20260916/review/confirmed_high_review.html'
html_doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>高质量 Trace 抽查</title><style>
:root{--ink:#172b4d;--muted:#6b778c;--line:#e6eaf0;--bg:#f5f7fa;--blue:#1769aa;--green:#087f5b;--amber:#9a6700}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#1f2933;font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}header{background:linear-gradient(120deg,#142b4a,#1d527d);color:#fff;padding:26px max(24px,calc((100vw - 1380px)/2)) 22px}header h1{margin:0;font-size:25px;letter-spacing:.2px}header p{margin:5px 0 0;color:#d9e7f3}.wrap{max-width:1380px;margin:0 auto;padding:20px 24px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px}.stat,.panel,.case{background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #102a4309}.stat{padding:14px 16px}.stat b{display:block;font-size:25px;color:var(--ink)}.stat span{color:var(--muted);font-size:12px}.layout{display:grid;grid-template-columns:260px minmax(0,1fr) 410px;gap:16px;align-items:start}.panel{padding:16px}.panel h3{margin:0 0 12px;color:var(--ink);font-size:15px}label{display:block;color:var(--muted);font-size:12px;margin:12px 0 5px}input,select{width:100%;border:1px solid #cbd5e1;border-radius:7px;padding:8px;background:#fff;color:#1f2933}button{border:0;border-radius:7px;padding:8px 11px;background:#e8f1fa;color:var(--blue);cursor:pointer}.list-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;color:var(--muted)}.case{padding:14px 16px;margin-bottom:10px;cursor:pointer;transition:.15s}.case:hover,.case.active{border-color:#5b9bd5;box-shadow:0 0 0 2px #5b9bd522}.case h3{margin:0 0 5px;color:var(--ink);font-size:15px}.meta{display:flex;gap:6px;flex-wrap:wrap;color:var(--muted);font-size:12px}.tag{border-radius:99px;padding:2px 8px;background:#eef5fb;color:var(--blue)}.tag.green{background:#e5f6ef;color:var(--green)}.tag.amber{background:#fff5d6;color:var(--amber)}.detail{position:sticky;top:15px;max-height:calc(100vh - 30px);overflow:auto}.detail h2{font-size:18px;color:var(--ink);margin:0 0 5px}.detail .sub{color:var(--muted);font-size:12px;margin-bottom:15px}dl{display:grid;grid-template-columns:100px 1fr;gap:8px 10px;margin:0}dt{font-weight:650;color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}.goal{background:#f6f9fc;border-left:3px solid #5b9bd5;padding:10px;margin:10px 0 15px;white-space:pre-wrap}.mapping{padding:7px 0;border-bottom:1px solid #f0f2f5}.mapping:last-child{border-bottom:0}.mapping small{display:block;color:var(--muted)}details{margin-top:15px;border-top:1px solid var(--line);padding-top:11px}summary{cursor:pointer;color:var(--blue);font-weight:650}pre{white-space:pre-wrap;background:#f7f9fb;padding:10px;border-radius:6px;font-size:12px}ol{padding-left:22px;font-size:12px}.empty{color:var(--muted);padding:35px;text-align:center}@media(max-width:1050px){.layout{grid-template-columns:220px minmax(0,1fr)}.detail{grid-column:1/-1;position:static;max-height:none}.stats{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){.wrap{padding:12px}.layout{display:block}.panel{margin-bottom:12px}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><h1>高质量 Trace 抽查</h1><p>confirmed + high · batch_20260828_162718_1dd984cc · v3 · 40 条候选</p></header><main class="wrap"><section class="stats"><div class="stat"><b id="shown">40</b><span>当前显示</span></div><div class="stat"><b>40</b><span>confirmed + high</span></div><div class="stat"><b>''' + str(len(domains)) + '''</b><span>涉及领域</span></div><div class="stat"><b>''' + str(sum(1 for x in items if x['difficulty']=='high')) + '''</b><span>高难度</span></div><div class="stat"><b>''' + str(sum(1 for x in items if x['integrity']=='complete')) + '''</b><span>完整 Episode</span></div></section><section class="layout"><aside class="panel"><h3>筛选</h3><label>关键词</label><input id="q" placeholder="目标、session、trace…"><label>领域</label><select id="domain"><option value="">全部领域</option>''' + domain_options + '''</select><label>失败模式</label><select id="failure"><option value="">全部失败模式</option>''' + failure_options + '''</select><label>难度</label><select id="difficulty"><option value="">全部难度</option><option>high</option><option>medium</option><option>low</option><option>unknown</option></select><button id="clear" style="margin-top:14px">清除筛选</button></aside><section><div class="list-head"><span id="count"></span><span>点击案例查看详情</span></div><div id="list"></div></section><aside class="panel detail" id="detail"><div class="empty">选择左侧案例查看详情</div></aside></section></main><script>
const DATA=''' + payload + ''';const $=id=>document.getElementById(id);let selected=null;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function filtered(){let q=$('q').value.toLowerCase(),d=$('domain').value,f=$('failure').value,di=$('difficulty').value;return DATA.filter(x=>(!q||JSON.stringify(x).toLowerCase().includes(q))&&(!d||x.domain===d)&&(!f||x.failures.includes(f))&&(!di||x.difficulty===di))}
function render(){let a=filtered();$('shown').textContent=a.length;$('count').textContent=`${a.length} / ${DATA.length} 条`; $('list').innerHTML=a.length?a.map(x=>`<article class="case ${selected===x.index?'active':''}" data-i="${x.index}"><h3>#${x.index} · 来源行 ${esc(x.line)}</h3><div class="meta"><span class="tag">${esc(x.domain)}</span><span class="tag">${esc(x.stage)}</span><span class="tag ${x.difficulty==='high'?'amber':''}">${esc(x.difficulty)}</span><span class="tag green">${esc(x.failures[0]||'未标记')}</span></div><p>${esc(x.goal).slice(0,180)}${String(x.goal).length>180?'…':''}</p></article>`).join(''):'<div class="empty">没有符合条件的案例</div>';document.querySelectorAll('.case').forEach(e=>e.onclick=()=>{selected=Number(e.dataset.i);render();detail(DATA.find(x=>x.index===selected))});if(selected&&a.some(x=>x.index===selected))detail(DATA.find(x=>x.index===selected))}
function detail(x){if(!x)return;$('detail').innerHTML=`<h2>#${x.index} · 来源行 ${esc(x.line)}</h2><div class="sub">session=${esc(x.session)}<br>trace=${esc(x.trace)}<br>episode=${esc(x.episode)}</div><div class="goal">${esc(x.goal)}</div><dl><dt>领域</dt><dd>${esc(x.domain)}</dd><dt>科研阶段</dt><dd>${esc(x.stage)}</dd><dt>难度</dt><dd>${esc(x.difficulty)}；${esc(x.factors.join('、')||'无因素')}</dd><dt>失败模式</dt><dd>${esc(x.failures.join('、')||'无')}</dd><dt>Episode</dt><dd>${esc(x.integrity)}；证据 ${esc(x.evidence_status)}</dd></dl><details open><summary>需求点与子需求点（${x.mappings.length}）</summary>${x.mappings.length?x.mappings.map(m=>`<div class="mapping"><b>${esc(m.requirement_name)}</b><br>${esc(m.subrequirement_name)} <small>CSV 第${esc(m.taxonomy_row)}行</small></div>`).join(''):'<p>未映射</p>'}</details><details><summary>质量闸门理由</summary><pre>${esc(JSON.stringify(x.gate_reasons,null,2))}</pre></details><details><summary>模型理由</summary><p>${esc(x.reason)}</p></details><details><summary>证据引用（${x.refs.length}）</summary><ol>${x.refs.map(e=>`<li><code>${esc(e.event_id)}</code> ${esc(e.evidence_role)}：${esc(e.quote)}</li>`).join('')||'<li>无</li>'}</ol></details>`}
['q','domain','failure','difficulty'].forEach(id=>$(id).oninput=render);$('clear').onclick=()=>{['q','domain','failure','difficulty'].forEach(id=>$(id).value='');selected=null;render()};render();
</script></body></html>'''
out = root / 'workspace/analysis-runs/batch_20260828_162718_1dd984cc/run_20260828_164200_wisland_v1/06_trace_quality_labeling/trace-label-model-v3-20260916/review/confirmed_high_review.html'
out.write_text(html_doc, encoding='utf-8')
print(out, len(items))
