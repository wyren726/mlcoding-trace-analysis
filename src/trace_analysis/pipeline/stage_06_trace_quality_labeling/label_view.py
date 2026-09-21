"""Deterministic, inline human-readable labels. Never generates semantic judgments."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

VERSION = 'inline-label-review-v2'
VALUES = {
    'low': '低', 'medium': '中', 'high': '高', 'unknown': '未知',
    'confirmed': '边界明确（模型判断）', 'needs_review': '边界待核验',
    'initial_query': '初始请求', 'clarified_later': '后续澄清', 'new_later': '后续新增',
    'missing_initial_spec': '补充初始规格', 'model_correction': '纠正已有输出',
    'goal_change': '改变目标', 'natural_deepening': '沿原任务继续深入',
    'acknowledgement': '确认或认可', 'uncertain': '不确定',
    'verification_failure': '核验缺失或错误', 'incomplete_execution': '执行不完整',
    'constraint_loss': '约束遗漏', 'wrong_assumption': '错误假设', 'tool_misuse': '工具误用',
    'error_recovery_failure': '错误恢复失败', 'context_tracking_failure': '上下文跟踪失败',
    'artifact_delivery_failure': '交付失败',
    'multi_step': '多步骤', 'cross_artifact': '跨产物', 'tool_dependency': '工具依赖',
    'long_horizon_state': '长期状态维护', 'goal_evolution': '需求演进',
    'scientific_reasoning': '科学推理', 'verification_burden': '验证负担',
    'external_information_dependency': '外部信息依赖',
}
DIMENSIONS = {
    'scientific_reasoning': '科学推理与方法选择', 'constraint_complexity': '约束复杂度',
    'workflow_dependencies': '跨步骤与跨产物依赖', 'environment_and_tools': '环境与工具挑战',
    'verification_burden': '结果验证负担',
}


def _path(value):
    return re.sub(r'\[(\d+)\]', r'.\1', value.lstrip('/').replace('/', '.'))


def _display(value):
    if isinstance(value, list):
        return [_display(v) for v in value]
    if isinstance(value, str):
        return VALUES.get(value, value)
    return value


def without_view(model):
    return {k: v for k, v in model.items() if k != 'review_view'}


def build_view(model):
    """Join field-specific claims/quotes; unrelated object-level claim lists stay out."""
    model = without_view(model)
    refs = {e['evidence_id']: e for e in model['evidence_refs']}
    requirements = {r['case_requirement_id']: r['text'] for r in model['case_requirements']}
    episode_id = model['assessment_target']['target_id']

    def quote(eid):
        e = refs[eid]
        return {'source_role': e['source_role'], 'quote': e['quote'],
                'episode_membership': e['episode_membership'],
                'locator': deepcopy(e['locator']), 'evidence_id': eid, 'event_id': e['event_id'], 'turn_id': e['turn_id']}

    def cited(ids):
        return [quote(eid) for eid in dict.fromkeys(ids)]

    def requirement_texts(ids):
        return [{'text': requirements[rid], 'case_requirement_id': rid} for rid in ids]

    def item(title, value, path, target, *, object_path=None, supplemental=None, include_children=False):
        paths = {_path(path)}
        if object_path:
            paths.add(_path(object_path))
        matching = [c for c in model['claims'] if c['target_id'] == target and
                    (_path(c['field_path']) in paths or (include_children and _path(c['field_path']).startswith(_path(path)+'.')))]
        bases = []
        for c in matching:
            bases.append({'reason': c['reason'], 'support_status': c['support_status'], 'evidence': cited(c['evidence_ids']),
                          'missing_evidence_queries': list(c['missing_evidence_queries']),
                          'conflicting_evidence': cited(c['conflicting_evidence_ids']),
                          'claim_id': c['claim_id'], 'field_path': c['field_path']})
        hints = []
        if not matching:
            hints.append('缺少对应字段的判断说明。')
        elif not any(_path(c['field_path']) == _path(path) for c in matching):
            hints.append('关联解释的范围不同，请核对 field_path。')
        supplement = []
        for purpose, ids in (supplemental or {}).items():
            if ids:
                supplement.append({'purpose': purpose, 'evidence': cited(ids)})
        return {'label_name': title, 'value': deepcopy(value), 'claims': bases, 'supplemental_evidence': supplement,
                'notes': hints, 'field_path': path, 'target_id': target}

    target = model['assessment_target']
    goals = [item('任务目标', target['goal'], 'assessment_target.goal', episode_id),
             item('核心具体要求', requirement_texts(target['core_requirement_ids']), 'assessment_target.core_requirement_ids', episode_id),
             item('非核心具体要求', requirement_texts(target['noncore_requirement_ids']), 'assessment_target.noncore_requirement_ids', episode_id),
             item('任务边界', target['boundary_status'], 'assessment_target.boundary_status', episode_id)]
    reqs = []
    for i, req in enumerate(model['case_requirements']):
        path = f'case_requirements[{i}]'
        rid = req['case_requirement_id']
        # Present all requirement attributes together; don't repeat the same quote per attribute.
        reqs.append(item('用户具体要求', {'text': req['text'], 'origin': req['origin'], 'origin_label': _display(req['origin']),
                         'is_core': req['is_core'], 'historical_requirement_id': req['historical_requirement_id']},
                         path, rid, object_path=path+'.text', supplemental={'要求原文（原记录关联）': req['evidence_ids']}, include_children=True))
    attrs = model['task_attributes']
    attributes = [item(title, attrs[key], f'task_attributes.{key}', episode_id) for key, title in [
        ('domain_primary', '主要研究领域'), ('domain_secondary', '其他研究领域'),
        ('work_object_primary', '主要工作对象'), ('work_object_secondary', '其他工作对象'),
        ('difficulty', '整体任务难度'), ('difficulty_factors', '难度因素')]]
    for key, title in DIMENSIONS.items():
        path = f'task_attributes.difficulty_dimensions.{key}'
        attributes.append(item(title, attrs['difficulty_dimensions'][key]['level'],
                               path+'.level', episode_id, object_path=path))
    mappings = []
    for i, m in enumerate(model['capability_mapping']):
        path = f'capability_mapping[{i}]'
        value = {'case_requirement_text': requirements[m['case_requirement_id']], 'requirement_name': m.get('requirement_name'),
                 'subrequirement_name': m.get('subrequirement_name'), 'mapping_status': m['mapping_status'],
                 'parent_requirement_id': m['parent_requirement_id'], 'subrequirement_id': m.get('subrequirement_id'), 'taxonomy_row': m['taxonomy_row']}
        mappings.append(item('需求／子需求匹配', value, path, m['case_requirement_id'], object_path=path+'.taxonomy_row', include_children=True))
    feedbacks = []
    for i, f in enumerate(model['feedback_annotations']):
        path = f'feedback_annotations[{i}]'
        # A single feedback may affect multiple requirements. Include its explicit claims,
        # but only when their field path identifies this feedback object/type.
        tids = f['target_requirement_ids'] or [episode_id]
        out = item('用户反馈', {'type': f['type'], 'type_label': _display(f['type']), 'target_requirements': requirement_texts(f['target_requirement_ids'])},
                   path, tids[0], object_path=path+'.type', supplemental={
                       '反馈本身': [f['evidence_id']], '反馈针对的先前输出': f['prior_output_evidence_ids'],
                       '当时要求的依据': f['requirement_evidence_ids']}, include_children=True)
        for tid in tids[1:]:
            out['claims'].extend(item('', None, path, tid, object_path=path+'.type', include_children=True)['claims'])
        if out['claims']:
            out['notes'] = []
        if refs[f['evidence_id']]['source_role'] != 'user':
            out['notes'].append('反馈主引用角色不是用户，请核验是否误把助手陈述标为用户反馈。')
        feedbacks.append(out)
    failures = []
    for i, f in enumerate(model['failure_annotations']):
        path = f'failure_annotations[{i}]'
        tids = f['target_requirement_ids']
        out = item('失败机制', {'type': f['type'], 'type_label': _display(f['type']), 'target_requirements': requirement_texts(tids)}, path,tids[0],object_path=path+'.type', include_children=True)
        for tid in tids[1:]:
            out['claims'].extend(item('',None,path,tid,object_path=path+'.type', include_children=True)['claims'])
        if out['claims']:
            out['notes'] = []
        failures.append(out)
    return {'version': VERSION,
            'description': '已有标签与证据的关联展示；support_status为模型初标，非人工复核结论。',
            'source_label_sha256': hashlib.sha256(json.dumps(model, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'assessment_target': goals, 'case_requirements': reqs, 'task_attributes': attributes,
            'requirement_mappings': mappings, 'feedback_annotations': feedbacks, 'failure_annotations': failures,
            'limitations': deepcopy(model['limitations'])}


def enrich_view(model):
    model = deepcopy(without_view(model))
    return {'schema_version': model['schema_version'], 'case_id': model['case_id'],
            'review_view': build_view(model),
            **{k: v for k, v in model.items() if k not in ('schema_version', 'case_id')}}
