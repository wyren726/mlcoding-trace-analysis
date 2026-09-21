"""Deterministic label/distribution report for each completed selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from os.path import relpath
from pathlib import Path
from tempfile import NamedTemporaryFile
from jsonschema.exceptions import ValidationError

from .selection import BASE, content_hash, normalize_path, read_jsonl, unique_index


REPORT_NAME = '06阶段标签质量与分布审查报告.md'


def requirement_distribution(labels, results, selector, reviews=()):
    """Count case incidence, never mapping frequency or a sum of child counts."""
    priority = [r for r in results if r['decision'] == 'priority_candidate']
    scopes = {}
    for result in priority:
        cid = result['case_id']
        if result['mode'] == 'episode':
            scopes[cid] = None
        elif cid not in scopes or scopes[cid] is not None:
            scopes.setdefault(cid, set()).add(result['assessment_id'])
    parents = {pid: dict(all_cases=set(), priority_cases=set(), children=set())
               for pid in selector.parent_names}
    children = {row: dict(all_cases=set(), priority_cases=set()) for row in selector.taxonomy}
    unresolved = []
    mapping_statuses = Counter()
    supported_mappings = 0
    invalid_labels = set()
    for cid, label in labels.items():
        model = label.get('model_output', label)
        try:
            selector.validate('selection_input.schema.json', model)
        except ValidationError:
            invalid_labels.add(cid)
            continue
        claims = {c['claim_id']: c for c in model.get('claims', [])}
        support = {key: c.get('support_status') for key, c in claims.items()}
        evidence = {key: c.get('evidence_ids', []) for key, c in claims.items()}
        reviewed = defaultdict(list)
        for review in reviews:
            if review.get('case_id') != cid or review.get('label_sha256') != content_hash(label):
                continue
            for item in review.get('claim_reviews', []):
                reviewed[item['claim_id']].append((item, review['status']))
        for key, values in reviewed.items():
            statuses = {value['support_status'] for value, _ in values}
            if len(statuses) == 1 and all(state == 'accepted' for _, state in values):
                support[key] = statuses.pop()
                evidence[key] = sorted({eid for value, _ in values for eid in value['evidence_ids']})
            else:
                support[key] = 'unknown'
        for index, mapping in enumerate(model.get('capability_mapping', [])):
            selected = cid in scopes and (scopes[cid] is None or mapping.get('case_requirement_id') in scopes[cid])
            if selected:
                mapping_statuses[mapping.get('mapping_status', 'unknown')] += 1
            row = selector.taxonomy.get(mapping.get('taxonomy_row'))
            ids = mapping.get('claim_ids', [])
            valid = (mapping.get('mapping_status') == 'supported' and row is not None
                     and mapping.get('parent_requirement_id') == row['requirement_id']
                     and bool(ids) and all(
                         key in claims and support.get(key) == 'supported' and evidence.get(key)
                         and not claims[key].get('conflicting_evidence_ids')
                         and claims[key].get('target_id') == mapping.get('case_requirement_id')
                         for key in ids)
                     and any(normalize_path(claims[key]['field_path']) == f'capability_mapping.{index}'
                             or normalize_path(claims[key]['field_path']).startswith(f'capability_mapping.{index}.')
                             for key in ids))
            if not valid:
                if selected:
                    unresolved.append(dict(case_id=cid, mapping=mapping,
                        support_statuses=sorted({str(support.get(key, 'missing')) for key in ids})))
                continue
            parent = parents[row['requirement_id']]
            child = children[mapping['taxonomy_row']]
            parent['all_cases'].add(cid)
            child['all_cases'].add(cid)
            if selected:
                supported_mappings += 1
                parent['priority_cases'].add(cid)
                parent['children'].add(mapping['taxonomy_row'])
                child['priority_cases'].add(cid)
    return dict(parents=parents, children=children, priority_case_ids=set(scopes),
                priority_units=len(priority), supported_mappings=supported_mappings,
                mapping_statuses=mapping_statuses, unresolved=unresolved,
                invalid_label_case_ids=invalid_labels)


def write_distribution_report(labels_path, selection_dir, results, sources, selector, summary,
                              *, labels=None, reviews=()):
    """Refresh the human-readable report in the review directory; no model calls."""
    labels = labels if labels is not None else unique_index(read_jsonl(labels_path), 'case_id')
    stats = requirement_distribution(labels, results, selector, reviews)
    from .report_paths import review_directory
    report_dir = review_directory(labels_path.parent)
    path = report_dir / REPORT_NAME
    selected = stats['priority_case_ids']
    total, count = len(labels), len(selected)
    covered_parents = sum(bool(p['priority_cases']) for p in stats['parents'].values())
    covered_children = sum(bool(c['priority_cases']) for c in stats['children'].values())
    mapped_cases = set().union(*(p['priority_cases'] for p in stats['parents'].values()))
    session_ids = set()
    missing_sources = []

    def model_for(cid):
        if cid in stats['invalid_label_case_ids']:
            return {}
        return labels[cid].get('model_output', labels[cid])

    def sessions(cid):
        source = sources.get(cid, {})
        original = source.get('source_row') or source
        return source.get('session_ids') or original.get('session_ids') or []

    for cid in selected:
        ids = sessions(cid)
        session_ids.update(ids)
        if not ids:
            missing_sources.append(cid)

    def cell(value):
        return str(value).replace('|', '\\|').replace('\r', '').replace('\n', '<br>')

    def table(headers, rows):
        return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
            '| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]

    def pct(n):
        return f'{n / count:.1%}' if count else '—'

    def source_key(cid):
        value = sources.get(cid, {}).get('source_line')
        return (value if isinstance(value, int) else float('inf'), cid)

    def locators(cases):
        return ', '.join(str(sources.get(cid, {}).get('source_line') or cid)
                         for cid in sorted(cases, key=source_key)) or '—'

    def link(label, target):
        return f'[{label}](<{relpath(target, report_dir)}>)'

    now = datetime.now(timezone.utc).isoformat()
    text = ['# 06 阶段标签质量与分布审查报告', '',
            f'筛选运行：`{summary["selection_run_id"]}`；规则版本：`{summary["policy_version"]}`。',
            f'生成时间（UTC）：{now}。', '', '## 1. 本轮结论', '',
            f'本轮有 **{count} 条优先候选案例**，对应 **{len(session_ids)} 个已知 session**；'
            f'明确且有支持证据的映射覆盖 **{covered_parents}/{len(stats["parents"])} 个需求点**、'
            f'**{covered_children}/{len(stats["children"])} 个子需求点**。', '',
            '本报告由已有标签、筛选结果和分类表自动统计，没有重新打标或调用模型。'
            '候选分流与标签支持状态仍属于模型/已提供复核记录的判断，不能解释为人工确认或每个需求均发生能力缺陷。', '',
            '## 2. 统计口径与覆盖', '',
            '- 分布单位为案例（Episode/case_id），同一案例对同一需求或子需求只计一次。多选标签的各行之和可超过案例总数。',
            '- 需求点案例数按案例集合去重；“命中子需求点数”是该需求下不同子需求的数量，不能用案例数或映射记录数代替。',
            '- 主表只统计 mapping_status=supported、父子层级合法且关联 claim 有证据支持、无冲突的映射；有效逐标签复核覆盖初标支持状态。部分匹配和未确定匹配单列。',
            '- 只筛选 decision=priority_candidate，不再附加历史 confirmed/high 条件。局部模式仅计入所选具体要求的映射，同一案例的多个入选局部要求仍按案例去重。',
            '- req_034–req_036 是辅助需求，展示其覆盖情况，但不能独立作为主需求准入依据。',
            '- 子需求表的来源行是原 04 JSONL 的来源行；taxonomy_row 是分类表记录编号（含表头），两者不同。session ID 见候选索引。', '']
    text += table(['覆盖指标', '优先候选'], [
        ('候选案例数 / 全部输入案例数', f'{count} / {total}'),
        ('有明确且有据映射的候选案例数', f'{len(mapped_cases)} / {count}'),
        ('覆盖主需求数', f'{sum(bool(stats["parents"][k]["priority_cases"]) for k in selector.primary)} / {len(selector.primary)}'),
        ('覆盖辅助需求数', f'{sum(bool(stats["parents"][k]["priority_cases"]) for k in selector.auxiliary)} / {len(selector.auxiliary)}'),
        ('覆盖子需求数', f'{covered_children} / {len(stats["children"])}'),
        ('有据映射记录数（未按案例/分类去重）', stats['supported_mappings']),
        ('未纳入主表的映射记录数', len(stats['unresolved'])),
        ('缺少 session 来源的候选案例数', len(missing_sources))])
    text += ['', f'## 3. 需求点分布（完整 {len(stats["parents"])} 项，包含零覆盖）', '']
    if summary.get('input_issue_records'):
        text += [f'其中 **{summary["input_issue_records"]} 条案例存在输入或处理问题，尚未生成语义标签**；'
                 '保留为待复核，不计入需求覆盖，不作为低质量 Trace。详细原因见同目录 input_issues.jsonl。', '']
    text += table(['需求 ID', '需求点', '用途', f'全部 {total} 条案例数',
                   '优先候选案例数', f'占 {count} 条比例', '优先候选命中子需求点数'], [
        (pid, selector.parent_names[pid], '辅助' if pid in selector.auxiliary else '主需求',
         len(p['all_cases']), len(p['priority_cases']), pct(len(p['priority_cases'])), len(p['children']))
        for pid, p in stats['parents'].items()])
    text += ['', f'## 4. 子需求点分布（完整 {len(stats["children"])} 项，包含零覆盖）', '',
             '所有比例均以本轮优先候选案例数为分母；来源行可在第 6 节对应到 session ID。', '']
    for pid, parent in stats['parents'].items():
        text += [f'### {pid} {selector.parent_names[pid]}', '',
                 f'优先候选覆盖 {len(parent["priority_cases"])} 条案例，命中 {len(parent["children"])} 个不同子需求。', '']
        text += table(['taxonomy_row', '子需求 ID', '子需求点', f'全部 {total} 条案例数',
                       '优先候选案例数', f'占 {count} 条比例', '优先候选的 04 来源行'], [
            (row_id, row['subrequirement_id'], row['subrequirement_name'],
             len(stats['children'][row_id]['all_cases']), len(stats['children'][row_id]['priority_cases']),
             pct(len(stats['children'][row_id]['priority_cases'])), locators(stats['children'][row_id]['priority_cases']))
            for row_id, row in selector.taxonomy.items() if row['requirement_id'] == pid])
        text += ['']
    text += ['## 5. 未计入明确覆盖的映射', '',
             '这些记录属于已入选案例中的部分/未确定或证据待核映射，不会被当作明确子需求覆盖，也不改变案例的筛选决定。', '']
    if stats['unresolved']:
        text += table(['04 来源行', 'mapping_id', '已有父需求', 'taxonomy_row', '映射状态', 'claim 支持状态'], [
            (sources.get(item['case_id'], {}).get('source_line') or item['case_id'],
             item['mapping'].get('mapping_id'), selector.parent_names.get(item['mapping'].get('parent_requirement_id'), '未确定'),
             item['mapping'].get('taxonomy_row') or '—', item['mapping'].get('mapping_status'), ', '.join(item['support_statuses']))
            for item in sorted(stats['unresolved'], key=lambda x: source_key(x['case_id']))])
    else:
        text += ['无。']
    text += ['', '## 6. 优先候选与 session 索引', '']
    text += table(['04 来源行', 'case_id', 'session ID', '任务目标'], [
        (sources.get(cid, {}).get('source_line') or '—', cid, ', '.join(sessions(cid)) or '来源缺失',
         model_for(cid).get('assessment_target', {}).get('goal', '—'))
        for cid in sorted(selected, key=source_key)])
    text += ['', '## 7. 筛选完成情况与标签属性', '',
             f'筛选模式为 `{summary["mode"]}`，共 {len(results)} 个筛选单元，'
             f'其中 {stats["priority_units"]} 个优先候选单元，去重后为 {count} 个案例。以下规则分流按筛选单元计数。', '']
    decisions = Counter(r['decision'] for r in results)
    text += table(['分流', '筛选单元数'], [('优先候选', decisions['priority_candidate']),
                  ('待复核', decisions['review']), ('不入选', decisions['not_selected'])])
    text += [''] + table(['规则', '判断内容', '通过', '待复核', '不满足'], [
        (rule['rule_id'], rule['name'], *[sum(r['rule_checks'][rule['rule_id']]['status'] == state for r in results)
                                        for state in ['pass', 'pending', 'fail']])
        for rule in selector.policy['rules']])
    prechecks = Counter(r['prechecks']['valid_label_and_references'] for r in results)
    text += ['', f'标签与核验引用前置检查：通过 {prechecks["pass"]}，待修正 {prechecks["pending"]}，不满足 {prechecks["fail"]}。'
             '这是程序结构与引用检查，不等于独立语义审核。', '',
             f'无法按标签 schema 解析的案例为 {len(stats["invalid_label_case_ids"])} 条，保留在总分母中但不计入映射覆盖。', '',
             '以下是 Episode 标签取值分布；局部筛选时也仅用于说明所属案例的属性，不替代局部难度核验。']
    for field, title in [('difficulty', '任务难度'), ('work_object_primary', '主要工作对象'), ('domain_primary', '主要领域')]:
        all_values = Counter(str(model_for(cid).get('task_attributes', {}).get(field) or 'unknown')
                             for cid in labels)
        selected_values = Counter(str(model_for(cid).get('task_attributes', {}).get(field) or 'unknown')
                                  for cid in selected)
        text += ['', f'### {title}', ''] + table([title, f'全部 {total} 条', f'优先候选 {count} 条', '优先候选占比'],
            [(value, all_values[value], selected_values[value], pct(selected_values[value]))
             for value in sorted(all_values, key=lambda x: (-selected_values[x], -all_values[x], x))])
    text += ['', '## 8. 文件与版本依据', '',
             '- ' + link('本次逐条筛选结果', selection_dir / 'selection.jsonl'),
             '- ' + link('本次筛选统计及输入摘要', selection_dir / 'selection_summary.json'),
             '- ' + link('完整标签、解释与原文引文', labels_path),
             '- ' + link('需求分类表', BASE / 'config/requirement_taxonomy.csv'),
             f'- 分类表 SHA256：`{selector.taxonomy_sha256}`。',
             f'- 政策 SHA256：`{selector.policy_sha256}`。',
             f'- 标签 SHA256：`{summary.get("inputs", {}).get(str(labels_path.resolve()), hashlib.sha256(labels_path.read_bytes()).hexdigest())}`。']
    if summary.get('session_export'):
        text += ['- ' + link('优先候选 session 清单', summary['session_export']['output'])]
    text += ['', '此报告随标签运行目录中的最新筛选结果更新；具体筛选版本以页首运行 ID 及其独立 selection.jsonl 为准。', '']
    content = '\n'.join(text)
    with NamedTemporaryFile('w', encoding='utf-8', dir=report_dir, prefix='.distribution-report-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(output=str(path), sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(),
                generated_at=now, priority_cases=count, priority_sessions=len(session_ids),
                covered_requirements=covered_parents, covered_subrequirements=covered_children,
                supported_mapping_records=stats['supported_mappings'],
                uncounted_mapping_records=len(stats['unresolved']))
