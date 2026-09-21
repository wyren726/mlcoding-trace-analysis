"""Read-only analysis of four frozen Stage06 label snapshots; no model calls."""
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime, timezone
import csv
import hashlib
import json

from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import Selector, BASE, normalize_path
from trace_analysis.pipeline.stage_06_trace_quality_labeling.distribution_report import requirement_distribution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/高质量Trace筛选_20260915/子需求精简诊断_20260921'
SPECS = [('20260813', '', 14), ('20260828', '', 81),
         ('20260820', 'review-observed-failures-20260920', 242),
         ('20260822', 'merged-repairs-20260920', 799)]


def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csvwrite(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    def cell(v):
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] + [
        '| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]


def pct(a, b):
    return round(100 * a / b, 3) if b else ''


def main():
    selector = Selector()
    taxonomy = list(selector.taxonomy.values())  # Preserve original CSV order, including zero hits.
    byid = {t['subrequirement_id']: t for t in taxonomy}
    order = {sid: i for i, sid in enumerate(byid)}
    cases, labels, inputs, batch_counts = {}, {}, {}, {}
    hits = {sid: set() for sid in byid}
    raw = {sid: set() for sid in byid}
    units = {sid: set() for sid in byid}
    refs = defaultdict(list)
    excluded = []
    mapping_count = 0
    for batch, suffix, expected in SPECS:
        bases = list((ROOT / 'workspace/analysis-runs').glob(
            f'batch_{batch}_*/*/06_trace_quality_labeling/trace-label-selection-v6-20260917'))
        assert len(bases) == 1
        directory = bases[0] / suffix
        lp, sp = directory / 'labels.jsonl', directory / 'sources.jsonl'
        records = read(lp)
        sources = {r['case_id']: r for r in read(sp)}
        assert len(records) == expected
        batch_counts[batch] = len(records)
        inputs[str(lp)] = sha(lp)
        inputs[str(sp)] = sha(sp)
        for line, label in enumerate(records, 1):
            cid = label['case_id']
            assert cid not in labels
            selector.validate('selection_input.schema.json', label)
            source = sources[cid]
            src = source['source_row']
            labels[cid] = label
            cases[cid] = dict(case_id=cid, batch=batch, episode_id=src['episode_id'],
                trace_id=src['trace_id'], session_ids=source['session_ids'],
                goal=label['assessment_target']['goal'], label_path=str(lp), label_line=line,
                source_line=source['source_line'], episode_boundary=src['episode_boundary'],
                models=src.get('models', []))
            claims = {r['claim_id']: r for r in label['claims']}
            evidence = {r['evidence_id']: r for r in label['evidence_refs']}
            reqs = {r['case_requirement_id']: r for r in label['case_requirements']}
            for i, m in enumerate(label['capability_mapping']):
                t = selector.taxonomy.get(m.get('taxonomy_row'))
                sid = t['subrequirement_id'] if t else None
                if sid:
                    raw[sid].add(cid)
                ids = m.get('claim_ids', [])
                cs = [claims[k] for k in ids if k in claims]
                valid = (m['mapping_status'] == 'supported' and t is not None
                    and m['parent_requirement_id'] == t['requirement_id']
                    and bool(ids) and len(cs) == len(ids)
                    and all(c['support_status'] == 'supported' and c['evidence_ids']
                        and not c.get('conflicting_evidence_ids')
                        and c['target_id'] == m['case_requirement_id'] for c in cs)
                    and any(normalize_path(c['field_path']) == f'capability_mapping.{i}'
                        or normalize_path(c['field_path']).startswith(f'capability_mapping.{i}.') for c in cs))
                if not valid:
                    excluded.append(dict(case_id=cid, batch=batch, mapping=m,
                        claim_support_statuses=[c['support_status'] for c in cs]))
                    continue
                assert m['case_requirement_id'] in reqs
                eids = sorted({e for c in cs for e in c['evidence_ids']})
                assert all(e in evidence and evidence[e]['quote'] for e in eids)
                assert m.get('taxonomy_sha256', selector.taxonomy_sha256) == selector.taxonomy_sha256
                hits[sid].add(cid)
                units[sid].add((cid, m['case_requirement_id']))
                refs[cid, sid].append(dict(mapping_id=m['mapping_id'],
                    requirement_id=m['case_requirement_id'],
                    requirement_text=reqs[m['case_requirement_id']]['text'],
                    reasons=[c['reason'] for c in cs], evidence=[evidence[e] for e in eids]))
                mapping_count += 1
    n = len(cases)
    assert n == 1136
    assert len({c['episode_id'] for c in cases.values()}) == n
    # Independent comparison against the existing distribution implementation.
    baseline = requirement_distribution(labels, [], selector)
    assert not baseline['invalid_label_case_ids']
    for t in taxonomy:
        assert hits[t['subrequirement_id']] == baseline['children'][int(t['taxonomy_row'])]['all_cases']
    original_cases = dict(cases)
    original_batch_counts = dict(batch_counts)
    groups = defaultdict(list)
    for cid, c in cases.items():
        boundary = c['episode_boundary']
        groups[(c['trace_id'], boundary['start_turn_id'], boundary['end_turn_id'])].append(cid)
    representatives, duplicates = {}, []
    for key, ids in groups.items():
        # Fixed provenance-based choice, independent of labels or hit counts.
        winner = sorted(ids, key=lambda cid: (-int(cases[cid]['batch']), cases[cid]['source_line'], cid))[0]
        for cid in ids:
            representatives[cid] = winner
        if len(ids) > 1:
            sets = [{sid for sid in hits if cid in hits[sid]} for cid in ids]
            duplicates.append(dict(trace_and_boundaries=list(key), representative=winner,
                records=[dict(**cases[cid], supported_children=sorted(s)) for cid, s in zip(ids, sets)],
                disagreeing_children=sorted(set.union(*sets)-set.intersection(*sets))))
    retained = set(representatives.values())
    cases = {cid:c for cid,c in cases.items() if cid in retained}
    for sid in hits:
        hits[sid] &= retained
        raw[sid] &= retained
        units[sid] = {(cid, req) for cid, req in units[sid] if cid in retained}
    batch_counts = dict(Counter(c['batch'] for c in cases.values()))
    n = len(cases)
    assert n == len(groups)
    all_sessions = {s for c in cases.values() for s in c['session_ids']}
    session_hits = {sid: {s for cid in ids for s in cases[cid]['session_ids']} for sid, ids in hits.items()}
    parents = defaultdict(set)
    for sid, ids in hits.items():
        parents[byid[sid]['requirement_id']].update(ids)

    def example(cid, sids):
        return dict(**cases[cid], mappings={sid: refs[cid, sid] for sid in sids})

    pairs, candidates = [], []
    for a, b in combinations(byid, 2):
        aa, bb = hits[a], hits[b]
        both = aa & bb
        shared_req = {cid for cid, _ in units[a] & units[b]}
        assert shared_req <= both
        ab = len(both) / len(aa) if aa else 0
        ba = len(both) / len(bb) if bb else 0
        same_ratio = len(shared_req) / len(both) if both else 0
        relation, direction = '', ''
        if len(both) >= 10 and ab >= .8 and ba >= .8:
            relation = '合并核查候选' if same_ratio >= .5 else '高共现但多为不同具体需求'
        elif len(both) >= 10 and max(ab, ba) >= .9 and min(ab, ba) <= .5:
            direction = f'{a} → {b}' if ab >= ba else f'{b} → {a}'
            relation = '归属核查候选' if same_ratio >= .5 else '单向共现，先核查流程依赖'
        sa, sb = session_hits[a], session_hits[b]
        row = dict(A=a, A名称=byid[a]['subrequirement_name'], B=b, B名称=byid[b]['subrequirement_name'],
            同父需求=byid[a]['requirement_id'] == byid[b]['requirement_id'],
            A命中数=len(aa), B命中数=len(bb), 共同命中数=len(both),
            A出现时B出现百分比=pct(len(both), len(aa)), B出现时A出现百分比=pct(len(both), len(bb)),
            Jaccard百分比=pct(len(both), len(aa | bb)),
            lift=round(len(both) * n / (len(aa) * len(bb)), 4) if aa and bb else '',
            同一具体需求共现案例数=len(shared_req), 同需求共现占比=pct(len(shared_req), len(both)),
            仅A案例数=len(aa-bb), 仅B案例数=len(bb-aa),
            会话A出现时B出现百分比=pct(len(sa & sb), len(sa)),
            会话B出现时A出现百分比=pct(len(sa & sb), len(sb)),
            信号=relation, 方向=direction)
        pairs.append(row)
        if relation:
            candidates.append(dict(metrics=row,
                A定义=byid[a]['definition'], A边界=byid[a]['boundary'],
                B定义=byid[b]['definition'], B边界=byid[b]['boundary'],
                共现案例=[example(cid, [a, b]) for cid in sorted(both)[:3]],
                同需求共现案例=[example(cid, [a, b]) for cid in sorted(shared_req)[:2]],
                仅A案例=[example(cid, [a]) for cid in sorted(aa-bb)[:2]],
                仅B案例=[example(cid, [b]) for cid in sorted(bb-aa)[:2]]))
    rows = []
    for t in taxonomy:
        sid, pid = t['subrequirement_id'], t['requirement_id']
        h = hits[sid]
        related = [r for r in pairs if sid in (r['A'], r['B']) and r['共同命中数']]
        related.sort(key=lambda r: (-r['共同命中数'], -float(r['Jaccard百分比']), order[r['B'] if r['A'] == sid else r['A']]))
        top = related[:3]
        signals = [c['metrics'] for c in candidates if sid in (c['metrics']['A'], c['metrics']['B'])]
        frequency = '零命中' if not h else '极低频（1–5）' if len(h) <= 5 else '低频（<1%）' if len(h)/n < .01 else '非低频'
        if not h:
            advice = '删除/收拢候选：先查漏标与场景覆盖'
        elif len(h) / n < .01:
            advice = '低频收拢候选：先核查独立价值'
        else:
            advice = '暂保留'
        if not parents[pid]:
            advice = '父需求整体零覆盖，暂缓删项判断'
        if any(r['信号'] == '合并核查候选' for r in signals):
            advice += '；核查合并'
        if any(r['信号'] == '归属核查候选' for r in signals):
            advice += '；核查归属'
        reasons = f'{len(h)}/{n}例；父需求覆盖{len(parents[pid])}例；未明确支持但曾指向本项{len(raw[sid]-h)}例'
        rows.append(dict(原顺序=len(rows)+1, 分类表行=t['taxonomy_row'], 子需求ID=sid,
            子需求=t['subrequirement_name'], 父需求ID=pid, 父需求=t['requirement_name'],
            明确命中案例数=len(h), 总体命中率百分比=pct(len(h), n), 父需求命中案例数=len(parents[pid]),
            父需求内命中率百分比=pct(len(h), len(parents[pid])),
            命中会话数=len(session_hits[sid]), 会话命中率百分比=pct(len(session_hits[sid]), len(all_sessions)),
            原始指向案例数=len(raw[sid]), 仅未明确支持案例数=len(raw[sid]-h),
            **{f'{b}命中数':sum(cases[c]['batch'] == b for c in h) for b, _, _ in SPECS},
            **{f'{b}命中率百分比':pct(sum(cases[c]['batch'] == b for c in h), batch_counts[b]) for b, _, _ in SPECS},
            频率标记=frequency, 常见共现对象='；'.join(f'{r["B"] if r["A"] == sid else r["A"]}（{r["共同命中数"]}例）' for r in top),
            关系核查对象='；'.join(f'{r["B"] if r["A"] == sid else r["A"]}：{r["信号"]} {r["方向"]}' for r in signals),
            初步建议=advice, 依据=reasons,
            代表案例='；'.join(sorted(h)[:3]), 定义=t['definition'], 边界=t['boundary']))
    assert [r['子需求ID'] for r in rows] == list(byid)
    assert len(rows) == 216 and len(pairs) == 216*215//2
    summary = dict(valid_label_records=len(original_cases), deduplicated_cases=n,
        duplicate_groups=len(duplicates), duplicate_records_removed=len(original_cases)-n,
        unique_sessions=len(all_sessions), raw_batches=original_batch_counts,
        deduplicated_batches=batch_counts, supported_mapping_records_before_dedup=mapping_count,
        excluded_mapping_records_before_dedup=len(excluded),
        covered_children=sum(bool(v) for v in hits.values()), zero_hit_children=sum(not v for v in hits.values()),
        one_to_five_hit_children=sum(1 <= len(v) <= 5 for v in hits.values()),
        nonzero_below_one_percent_children=sum(0 < len(v)/n < .01 for v in hits.values()),
        relation_signals=dict(Counter(c['metrics']['信号'] for c in candidates)),
        pending_label_failures=123, pending_boundary_issues=58)
    OUT.mkdir(parents=True, exist_ok=False)
    csvwrite('子需求精简诊断表.csv', rows)
    csvwrite('全部子需求两两关系.csv', pairs)
    if candidates:
        csvwrite('关系核查候选.csv', [c['metrics'] for c in candidates])
    dump('关系核查证据.json', candidates)
    dump('未计入明确命中的映射.json', excluded)
    dump('统计摘要.json', summary)
    dump('重复任务片段与标签差异.json', duplicates)
    with (OUT/'逐案例命中与来源.jsonl').open('w') as f:
        for cid, case in original_cases.items():
            sids = [sid for sid in byid if refs.get((cid, sid))]
            f.write(json.dumps(dict(**case, subrequirements=sids,
                counted_in_primary=cid in retained, representative_case_id=representatives[cid],
                mappings={sid: refs[cid, sid] for sid in sids}), ensure_ascii=False)+'\n')
    detail = ['# 子需求代表案例（原始分类顺序）', '', '证据为已有模型打标记录，未作新增人工或模型复核。每项最多展示3例；完整命中见逐案例JSONL。', '']
    for r in rows:
        sid = r['子需求ID']
        detail += [f'## {sid} {r["子需求"]}', '', r['依据'], '', '定义：'+r['定义'], '', '边界：'+r['边界'], '']
        if not hits[sid]:
            detail += ['没有明确命中案例；不能据此断言真实需求不存在。', '']
        for cid in sorted(hits[sid])[:3]:
            c = cases[cid]
            detail += [f'### {cid}', '', f'批次：{c["batch"]}；Session：{", ".join(c["session_ids"])}。', '',
                f'[原始标签]({c["label_path"]}:{c["label_line"]})', '', '任务：'+c['goal'], '']
            for ref in refs[cid, sid][:2]:
                detail += ['用户具体需求：'+ref['requirement_text'], '', '打标理由：'+'；'.join(ref['reasons']), '']
                for e in ref['evidence'][:2]:
                    detail += [f'> {e["source_role"]}：'+e['quote'].replace('\n', '\n> '), '']
    (OUT/'子需求代表案例.md').write_text('\n'.join(detail)+'\n')
    md = ['# 四批真实打标子需求精简诊断', '',
        f'基于四批全部 **1136条有效打标记录**，去除7条重复记录后为 **{n}个任务片段、{len(all_sessions)}个去重会话**；按原分类表顺序列出全部216项，包含零命中项。', '',
        f'明确命中 {summary["covered_children"]} 项，零命中 {summary["zero_hit_children"]} 项；1–5例 {summary["one_to_five_hit_children"]} 项；非零且总体命中率低于1%共 {summary["nonzero_below_one_percent_children"]} 项（包含1–5例）。', '',
        '## 数据范围和口径', '',
        '- 采用0813/0828原始有效标签、0820已应用局部修正版本、0822技术修复合并版本；纳入所有有效标签，不按优先候选筛选。数据与780条候选报告口径不同。',
        '- Episode ID虽不重复，但按trace_id + 起始turn_id + 结束turn_id发现7组重复。每组固定选批次日期较新者，同批选来源行较小者，不按标签数量挑选；不同边界的任务片段保留。重复版本及标签差异完整另列，未将不同版本标签合并制造共现。',
        '- 本次保留全部有效标签中的模型来源（含混合模型），因为分析的是用户需求体系；此前排除混合模型的限制应用于最终候选交付清单。',
        '- 明确命中要求映射supported、父子关系合法、相关判断supported、有登记证据、无冲突且指向该映射及具体需求。非明确支持的映射另列，不作为明确命中。未额外覆盖未应用的二次复核建议。',
        f'- 总体率=去重任务片段命中数/{n}；分批率以去重后归入该批的案例数为分母；父需求内率=命中数/至少明确命中该父需求一个子项的案例数。父需求分母为0时留空，不写0%。',
        '- 命中数为案例去重计数，不是映射条数；模型打标支持状态不等于人工金标准。123条打标失败、58条边界异常不进入分母。',
        '- 样本来自前序痛点筛选，并非所有原始用户对话的随机样本；低频项可能受样本组成、漏标、打标提示影响。', '',
        '## 调整信号的规则', '',
        '- 零命中列为删除/收拢核查候选；1–5例单独标记，非零且低于1%列为低频。阈值为本次诊断启发式，不是已确认的删项标准。父需求整体零覆盖时暂缓判断。',
        '- 共同出现至少10例且两个方向的条件比例都≥80%：高共现；其中至少50%共现案例映射到同一条具体用户需求，才列为合并核查候选。',
        '- 共同出现至少10例、一个方向≥90%、反向≤50%：单向包含信号；同需求共现比例≥50%才列为归属核查候选，否则优先核查流程依赖。箭头A→B表示A出现时通常也出现B，不是已证实的语义上下位关系。',
        '- Jaccard=交集/并集；lift=共同出现率/(A出现率×B出现率)。同需求共现指同一案例内至少一个case_requirement_id同时映射到A、B。',
        '- 同一具体用户需求仍可能包含多个操作，因此所有关系都是核查候选。关系证据附件给出原定义、边界、共现和各自独立案例；未直接删除、合并或修改分类表。', '',
        '## 各批有效标签数', '']
    md += table(['批次', '原始有效标签', '去重后归入本批'], [(b, original_batch_counts[b], batch_counts[b]) for b, _, _ in SPECS])
    md += ['', '## 子需求精简诊断表（原始顺序）', '', '完整数值、分批命中率、会话统计、定义与边界见CSV；示例见[代表案例](子需求代表案例.md)。', '']
    md += table(['序号', '子需求ID', '子需求', '父需求', '命中数', '总体%', '父内%', '0813', '0828', '0820', '0822', '常见共现', '初步建议'],
        [[r[k] for k in ['原顺序', '子需求ID', '子需求', '父需求', '明确命中案例数', '总体命中率百分比', '父需求内命中率百分比', '20260813命中数', '20260828命中数', '20260820命中数', '20260822命中数', '常见共现对象', '初步建议']] for r in rows])
    md += ['', '## 关系核查候选（按原始子需求顺序）', '']
    md += table(['A', 'A名称', 'B', 'B名称', '共同数', 'A→B%', 'B→A%', '同需求共现%', '信号', '方向'],
        [[c['metrics'][k] for k in ['A','A名称','B','B名称','共同命中数','A出现时B出现百分比','B出现时A出现百分比','同需求共现占比','信号','方向']] for c in candidates])
    md += ['', '## 附件', '', '- [完整诊断表CSV](子需求精简诊断表.csv)',
        '- [全部23220对子需求关系CSV](全部子需求两两关系.csv)',
        '- [关系核查候选CSV](关系核查候选.csv)', '- [关系核查证据JSON](关系核查证据.json)',
        '- [逐案例命中与来源JSONL](逐案例命中与来源.jsonl)',
        '- [子需求代表案例](子需求代表案例.md)', '- [未计入明确命中的映射](未计入明确命中的映射.json)',
        '- [统计摘要](统计摘要.json)', '- [来源版本与校验记录](manifest.json)', '']
    md += ['- [重复任务片段与标签差异](重复任务片段与标签差异.json)', '']
    (OUT/'子需求精简诊断报告.md').write_text('\n'.join(md))
    inputs[str(BASE/'config/requirement_taxonomy.csv')] = sha(BASE/'config/requirement_taxonomy.csv')
    for path in inputs:
        assert sha(Path(path)) == inputs[path], f'Input changed during generation: {path}'
    dump('manifest.json', dict(generated_at=datetime.now(timezone.utc).isoformat(),
        generator=str(Path(__file__).resolve()), generator_sha256=sha(Path(__file__)),
        taxonomy_sha256=selector.taxonomy_sha256, inputs=inputs, summary=summary,
        validation=dict(schema_validated_cases=len(original_cases), episode_ids_unique=True,
            trace_and_boundary_deduplicated=True,
            counts_match_existing_distribution=True, original_taxonomy_order=True,
            taxonomy_rows=len(rows), pair_rows=len(pairs)),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(summary, ensure_ascii=False))
    print(OUT)


if __name__ == '__main__':
    main()
