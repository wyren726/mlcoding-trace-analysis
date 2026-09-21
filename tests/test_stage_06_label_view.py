from copy import deepcopy
import pytest
from test_stage_06_selection import case, selector
from trace_analysis.pipeline.stage_06_trace_quality_labeling.label_view import build_view, enrich_view


def test_view_deterministic_and_preserves_labels(selector, case):
    model,bundle=case
    original=deepcopy(model)
    rich=selector.enrich_names(model)
    out=enrich_view(rich)
    assert model==original
    assert list(out)[:3]==['schema_version','case_id','review_view']
    assert {k:v for k,v in out.items() if k!='review_view'}==rich
    assert enrich_view(out)==out
    selector.validate('selection_input.schema.json',out)
    assert selector.select(out,bundle)['rule_checks']==selector.select(rich,bundle)['rule_checks']


def test_field_specific_reason_and_exact_quote(selector,case):
    model,_=case
    view=build_view(selector.enrich_names(model))
    difficulty=next(x for x in view['task_attributes'] if x['label_name']=='整体任务难度')
    assert [x['claim_id'] for x in difficulty['claims']]==['difficulty']
    assert difficulty['claims'][0]['evidence'][0]['quote']==model['evidence_refs'][0]['quote']
    assert difficulty['claims'][0]['support_status']=='supported'
    assert '模型初标' in view['description']
    assert view['requirement_mappings'][0]['value']['subrequirement_name']=='制定文献调研协议'
    assert view['requirement_mappings'][0]['value']['case_requirement_text']==model['case_requirements'][0]['text']
    # shared object claim_ids must not invent evidence for an unrelated field
    domain=next(x for x in view['task_attributes'] if x['label_name']=='主要研究领域')
    assert domain['claims']==[] and domain['notes']


def test_forged_display_is_rejected(selector,case):
    model,bundle=case
    out=enrich_view(selector.enrich_names(model))
    out['review_view']['assessment_target'][0]['claims'][0]['evidence'][0]['quote']='假原文'
    assert selector.select(out,bundle)['prechecks']['valid_label_and_references']=='pending'


def test_feedback_roles_and_distinct_evidence_purposes(selector,case):
    model,_=case
    model['feedback_annotations']=[dict(feedback_id='f1',type='model_correction',evidence_id='e1',
        target_requirement_ids=['r1'],prior_output_evidence_ids=['e1'],requirement_evidence_ids=['e1'],claim_ids=[])]
    model['evidence_refs'][0]['source_role']='assistant'
    f=build_view(selector.enrich_names(model))['feedback_annotations'][0]
    assert any('不是用户' in s for s in f['notes'])
    assert [e['purpose'] for e in f['supplemental_evidence']]==['反馈本身','反馈针对的先前输出','当时要求的依据']
    assert all(e['evidence'][0]['source_role']=='assistant' for e in f['supplemental_evidence'])


def test_conflicts_and_partial_support_are_not_lost(selector,case):
    model,_=case
    c=model['claims'][1];c.update(support_status='partial',conflicting_evidence_ids=['e1'],missing_evidence_queries=['缺少后续验证'])
    view=build_view(selector.enrich_names(model))
    d=next(x for x in view['task_attributes'] if x['label_name']=='整体任务难度')['claims'][0]
    assert d['support_status']=='partial' and d['conflicting_evidence'][0]['evidence_id']=='e1'
    assert d['missing_evidence_queries']==['缺少后续验证']


def test_view_keys_are_english_and_enum_values_are_unchanged(selector,case):
    model,_=case
    view=build_view(selector.enrich_names(model))
    def check_keys(value):
        if isinstance(value,dict):
            assert all(key.isascii() for key in value)
            for child in value.values():check_keys(child)
        elif isinstance(value,list):
            for child in value:check_keys(child)
    check_keys(view)
    assert view['version']=='inline-label-review-v2'
    assert next(x for x in view['task_attributes'] if x['label_name']=='整体任务难度')['value']=='medium'
    assert view['requirement_mappings'][0]['value']['mapping_status']=='supported'
