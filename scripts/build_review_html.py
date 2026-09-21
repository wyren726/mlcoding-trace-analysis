import glob, html, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = Path(glob.glob(str(root / 'workspace/analysis-runs/batch_*/run_*/06_trace_quality_labeling/trace-label-model-v3-20260916/labeled_candidates*.jsonl'))[0])
rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
rows = [x for x in rows if x['historical_result'].get('pain_judgment') == 'confirmed' and x['labels'].get('trace_quality') == 'high']

def esc(x): return html.escape(str(x if x is not None else ''))
cards = []
for i, r in enumerate(rows, 1):
    s, l = r['source'], r['labels']
    goal = r.get('reused', {}).get('preliminary_goal') or r.get('reused', {}).get('initial_query') or '未提供'
    mappings = l.get('capability_mapping') or []
    maps = '；'.join(f"{m.get('requirement_name','')} / {m.get('subrequirement_name','')}（第{m.get('taxonomy_row','?')}行）" for m in mappings) or '未映射'
    gates = '；'.join(f'{k}={v}' for k, v in r.get('quality_gates', {}).items())
    refs = ''.join(f"<li><code>{esc(e.get('event_id'))}</code> [{esc(e.get('evidence_role'))}] {esc(e.get('quote'))}</li>" for e in r.get('evidence_refs', [])) or '<li>无</li>'
    cards.append(f'''<article><h2>#{i} · 来源行 {esc(s.get('source_line_number'))} <small>{esc(l.get('domain_primary'))} · {esc(l.get('difficulty'))}</small></h2><p><b>定位：</b>session={esc((s.get('session_ids') or [''])[0])} · trace={esc(s.get('trace_id'))} · episode={esc(s.get('episode_id'))}</p><p><b>用户目标：</b>{esc(goal)}</p><p><b>科研阶段：</b>{esc(l.get('research_stage_primary'))}；<b>难度因素：</b>{esc(', '.join(l.get('difficulty_factors') or [])) or '无'}</p><p><b>失败模式：</b>{esc(', '.join(l.get('failure_pattern') or [])) or '无'}</p><p><b>质量闸门：</b><code>{esc(gates)}</code></p><p><b>需求/子需求：</b>{esc(maps)}</p><p><b>理由：</b>{esc(r.get('label_reasons'))}</p><details><summary>展开闸门理由和证据（{len(r.get('evidence_refs', []))} 条）</summary><pre>{esc(json.dumps(r.get('quality_gate_reasons', {}), ensure_ascii=False, indent=2))}</pre><ol>{refs}</ol></details></article>''')

out = root / 'workspace/analysis-runs/batch_20260828_162718_1dd984cc/run_20260828_164200_wisland_v1/06_trace_quality_labeling/trace-label-model-v3-20260916/review/confirmed_high_review.html'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('<!doctype html><meta charset="utf-8"><title>confirmed + high Trace 抽查</title><style>body{font:14px/1.6 system-ui;background:#f4f6f8;max-width:1100px;margin:auto;padding:20px}article{background:white;border:1px solid #ddd;border-radius:9px;padding:16px;margin:14px 0}h1{color:#172b4d}h2{font-size:18px;border-bottom:1px solid #eee;padding-bottom:8px}small{font-weight:normal;color:#1769aa}pre{background:#f7f9fb;padding:10px;white-space:pre-wrap}code{font-size:12px}</style><h1>confirmed + high Trace 抽查</h1><p>批次 batch_20260828_162718_1dd984cc · v3 · 共 '+str(len(rows))+' 条。筛选条件：历史判断 confirmed 且 trace_quality=high。high 表示材料可分析，不等于最终 Benchmark 通过。</p>'+''.join(cards), encoding='utf-8')
print(out, len(rows))
