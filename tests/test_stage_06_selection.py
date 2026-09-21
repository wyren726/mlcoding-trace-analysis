import copy
import json

import pytest

from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import Selector, content_hash, run_selection


@pytest.fixture
def selector():
    return Selector()


@pytest.fixture
def case():
    locator = dict(source_file='original.jsonl', record_index=0, physical_line=1)
    text = '修改代码并保持接口，通过回归测试。起始源码及版本已归档，测试输入可取得。'
    ref = dict(evidence_id='e1', event_id='event1', turn_id='turn1', source_role='user', quote=text,
               locator=locator, episode_membership='episode')
    def claim(cid, path, target):
        return dict(claim_id=cid, field_path=path, target_id=target, reason='原文依据', evidence_ids=['e1'],
                    missing_evidence_queries=[], conflicting_evidence_ids=[], support_status='supported', is_core=True)
    label = dict(schema_version='stage06-v1-draft', case_id='case1',
        assessment_target=dict(target_id='ep1', goal='修改代码', core_requirement_ids=['r1'],
                               noncore_requirement_ids=[], boundary_status='confirmed', claim_ids=['goal']),
        case_requirements=[dict(case_requirement_id='r1', text=text, origin='initial_query',
                                is_core=True, evidence_ids=['e1'], historical_requirement_id=None)],
        task_attributes=dict(domain_primary='计算机', domain_secondary=[], work_object_primary='代码',
                             work_object_secondary=[], difficulty='medium', difficulty_factors=['multi_step'],
                             difficulty_dimensions={k: dict(level=1, claim_ids=['difficulty']) for k in
                                 ['scientific_reasoning','constraint_complexity','workflow_dependencies','environment_and_tools','verification_burden']},
                             claim_ids=['difficulty']),
        capability_mapping=[dict(mapping_id='m1', case_requirement_id='r1', taxonomy_row=2,
                                 parent_requirement_id='req_001', mapping_status='supported', claim_ids=['mapping'])],
        feedback_annotations=[], failure_annotations=[], evidence_refs=[ref], limitations=[],
        claims=[claim('goal','assessment_target.goal','ep1'),
                claim('difficulty','task_attributes.difficulty','ep1'),
                claim('mapping','capability_mapping[0].taxonomy_row','r1'),
                claim('requirement','case_requirements[0].text','r1')])
    bundle = dict(schema_version='stage06-v1-draft', case_id='case1', source_sha256='0'*64,
        scope=dict(strategy='complete_episode', full_episode_read=True, provided_event_ids=['event1'],
                   omitted_body_event_ids=[], retrieval_rounds=0, feedback_scan_status='complete', coverage_notes=[]),
        events=[dict(event_id='event1',turn_id='turn1',role='user',content=text,body_provided=True,
                     locator=locator,episode_membership='episode')])
    return label, bundle


def audit(selector, label, statuses=None, mode='episode', target='ep1'):
    return dict(audit_id='audit1', case_id=label['case_id'], assessment_id=target, mode=mode,
                label_sha256=content_hash(label), policy_sha256=selector.policy_sha256,
                reviewer_type='human', reviewer_id='test-reviewer', review_time='2026-09-17T00:00:00Z',
                rule_checks={k: dict(status=v,reason='测试核验记录',claim_ids=['requirement'],evidence_ids=['e1'])
                             for k,v in (statuses or dict(S3='pass',S4='pass')).items()})


def test_no_audit_is_review_and_no_quality_gates_needed(selector, case):
    result = selector.select(*case)
    assert result['rule_checks']['S1']['status'] == 'pass'
    assert result['rule_checks']['S2']['status'] == 'pass'
    assert result['decision'] == 'review'
    assert result['rule_checks']['S4']['status'] == 'pending'
    assert result['evaluation']['reviewer_type'] == 'program'


def test_source_issues_remain_review_without_fabricating_labels(tmp_path):
    labels=tmp_path/'labels.jsonl';labels.write_text('')
    issue=dict(case_id='broken',episode_id='ep',kind='invalid_episode_boundary',
               error='Missing authoritative boundary',source_line=1,session_ids=['session'],
               source_row=dict(batch_id='batch',analysis_run_id='run',session_ids=['session']))
    (tmp_path/'input_issues.jsonl').write_text(json.dumps(issue)+'\n')
    class Client:
        def complete_json(self,*args):
            raise AssertionError('Do not label or model-review invalid source boundaries')
    summary=run_selection(labels,tmp_path/'selection',client=Client())
    result=json.loads((tmp_path/'selection/selection.jsonl').read_text())
    assert result['decision']=='review'
    assert '尚未生成语义标签' in result['reason']
    assert summary['input_issue_records']==1 and summary['valid_label_records']==0
    assert summary['counts']==dict(priority_candidate=0,review=1,not_selected=0)
    assert labels.read_text()==''
    assert '1 条案例存在输入或处理问题' in (tmp_path/'06阶段标签质量与分布审查报告.md').read_text()


def test_all_four_pass_without_historical_confirmed(selector, case):
    result = selector.select(*case, audits=[audit(selector,case[0])])
    assert result['decision'] == 'priority_candidate'
    assert result['evaluation']['review_ids'] == ['audit1']


def test_review_registers_exact_evidence_and_replays_without_changing_labels(tmp_path, case):
    label, bundle = case
    event = dict(event_id='tool_check', turn_id='turn2', role='tool', content='ZIP structure: OK\nXML parse: OK',
                 body_provided=True, locator=dict(source_file='original.jsonl', record_index=1, physical_line=2),
                 episode_membership='episode')
    bundle['events'].append(event)
    bundle['scope']['provided_event_ids'].append(event['event_id'])
    before = copy.deepcopy(label)
    labels = tmp_path / 'labels.jsonl'
    evidence = tmp_path / 'evidence.jsonl'
    labels.write_text(json.dumps(label) + '\n')
    evidence.write_text(json.dumps(bundle) + '\n')
    class Client:
        model = 'test-model'
        calls = 0
        def complete_json(self, system, user):
            self.calls += 1
            payload = json.loads(user)
            assert 'evidence_requests' in payload['unregistered_evidence_instruction']
            return dict(rule_checks={k: dict(status='pass', reason='用户要求与工具实际结果',
                                             claim_ids=['requirement'], evidence_ids=['e1', 'tool_check'])
                                     for k in payload['requested_rules']},
                        evidence_requests=[dict(event_id='tool_check', quote='ZIP structure: OK')]), {}
    client = Client()
    output = tmp_path / 'selection'
    summary = run_selection(labels, output, evidence_path=evidence, client=client)
    assert summary['counts']['priority_candidate'] == 1 and client.calls == 1
    saved = json.loads((output / 'model_selection_audits.jsonl').read_text())
    extra = saved['additional_evidence_refs'][0]
    assert extra['source_role'] == 'tool' and extra['locator'] == event['locator']
    assert saved['label_sha256'] == content_hash(before)
    assert json.loads(labels.read_text()) == before
    assert Selector().select(label, bundle, audits=[saved])['decision'] == 'priority_candidate'
    log = json.loads((output / 'model_review_log.jsonl').read_text())
    assert any(r['operation'] == 'register_exact_evidence' for r in log['reference_resolutions'])
    # Even a signed audit must pass source checks again when replayed.
    for defect in ['quote', 'role', 'locator', 'turn', 'membership', 'hidden', 'collision']:
        bad, source = copy.deepcopy(saved), copy.deepcopy(bundle)
        ref = bad['additional_evidence_refs'][0]
        if defect == 'quote': ref['quote'] = '不存在的结果'
        if defect == 'role': ref['source_role'] = 'user'
        if defect == 'locator': ref['locator']['physical_line'] = 99
        if defect == 'turn': ref['turn_id'] = 'wrong'
        if defect == 'membership': ref['episode_membership'] = 'prior_context'
        if defect == 'hidden':
            source['scope']['full_episode_read'] = False
            source['events'][1]['body_provided'] = False
            source['scope']['provided_event_ids'].remove('tool_check')
            source['scope']['omitted_body_event_ids'].append('tool_check')
        if defect == 'collision': ref['evidence_id'] = 'e1'
        result = Selector().select(label, source, audits=[bad])
        assert result['decision'] == 'review', defect
        assert result['prechecks']['valid_label_and_references'] == 'pending', defect


def test_reference_normalization_does_not_choose_ambiguous_or_unsupported_claims(selector, case):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection_review import reference_catalog
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.reference_repairs import review_references
    label, bundle = case
    baseline = selector.select(label, bundle)
    response = dict(rule_checks=dict(S3=dict(status='pass', reason='保留原判断',
                                            claim_ids=['r1'], evidence_ids=['event1'])))
    # r1 has two supported claims: no guessing which one the model intended.
    normalized, extra, changes = review_references(response, label, bundle, reference_catalog(label, baseline))
    assert normalized['rule_checks']['S3']['claim_ids'] == ['r1']
    assert normalized['rule_checks']['S3']['evidence_ids'] == ['e1']
    label['claims'][2]['support_status'] = 'partial'
    normalized, extra, changes = review_references(response, label, bundle, reference_catalog(label, baseline))
    assert normalized['rule_checks']['S3']['claim_ids'] == ['requirement']
    assert normalized['rule_checks']['S3']['status'] == 'pass'
    assert normalized['rule_checks']['S3']['reason'] == '保留原判断'
    label['claims'][3]['support_status'] = 'unknown'
    normalized, _, _ = review_references(response, label, bundle, reference_catalog(label, baseline))
    assert normalized['rule_checks']['S3']['claim_ids'] == ['r1']
    assert response['rule_checks']['S3']['claim_ids'] == ['r1']


@pytest.mark.parametrize('defect', ['quote', 'event', 'hidden', 'unused', 'multiple_quotes'])
def test_invalid_evidence_requests_do_not_register(selector, case, defect):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.reference_repairs import review_references
    label, bundle = case
    response = dict(rule_checks=dict(S3=dict(status='pass', reason='test', claim_ids=['requirement'],
                                            evidence_ids=['event1'])),
                    evidence_requests=[dict(event_id='event1', quote='修改代码')])
    if defect == 'quote': response['evidence_requests'][0]['quote'] = '伪造原文'
    if defect == 'event': response['evidence_requests'][0]['event_id'] = 'absent'
    if defect == 'hidden': bundle['events'][0]['body_provided'] = False
    if defect == 'unused': response['rule_checks']['S3']['evidence_ids'] = ['e1']
    if defect == 'multiple_quotes': response['evidence_requests'].append(dict(event_id='event1', quote='通过回归测试'))
    with pytest.raises(ValueError):
        review_references(response, label, bundle, [])


def test_supplementary_evidence_cannot_upgrade_partial_claim(selector, case):
    label, bundle = case
    label['claims'][3]['support_status'] = 'partial'
    record = audit(selector, label)
    record['additional_evidence_refs'] = [{**label['evidence_refs'][0], 'evidence_id': 'new'}]
    record['rule_checks']['S3']['evidence_ids'] = ['new']
    assert selector.select(label, bundle, audits=[record])['decision'] == 'review'


def test_rejected_registration_can_be_repaired_with_original_event_ids(tmp_path, case):
    label, bundle = case
    labels, evidence = tmp_path / 'labels.jsonl', tmp_path / 'evidence.jsonl'
    labels.write_text(json.dumps(label) + '\n')
    evidence.write_text(json.dumps(bundle) + '\n')
    class Client:
        model = 'test-model'
        calls = 0
        def complete_json(self, system, user):
            self.calls += 1
            payload = json.loads(user)
            if self.calls == 2:
                previous = payload['repair']['previous_response']
                assert previous['evidence_requests'] == [dict(event_id='event1', quote='修改代码')]
                assert previous['rule_checks']['S3']['evidence_ids'] == ['event1']
            return dict(rule_checks={k: dict(status='pass', reason='test',
                        claim_ids=['absent' if self.calls == 1 else 'requirement'], evidence_ids=['event1'])
                        for k in payload['requested_rules']},
                        evidence_requests=[dict(event_id='event1', quote='修改代码')]), {}
    client = Client()
    output = tmp_path / 'selection'
    summary = run_selection(labels, output, evidence_path=evidence, client=client)
    assert client.calls == 2 and summary['counts']['priority_candidate'] == 1
    logs = [json.loads(line) for line in (output / 'model_review_log.jsonl').read_text().splitlines()]
    assert [r['status'] for r in logs] == ['rejected', 'accepted']
    assert not any('_repair_response' in log for log in logs)


def test_supplementary_evidence_is_scoped_to_its_audit(selector, case):
    label, bundle = case
    first = audit(selector, label, dict(S3='pass'))
    first['additional_evidence_refs'] = [{**label['evidence_refs'][0], 'evidence_id': 'extra'}]
    first['rule_checks']['S3']['evidence_ids'] = ['extra']
    second = audit(selector, label, dict(S4='pass'))
    second['audit_id'] = 'second'
    second['rule_checks']['S4']['evidence_ids'] = ['extra']
    assert selector.select(label, bundle, audits=[first, second])['decision'] == 'review'


@pytest.mark.parametrize('defect', ['quote','role','mapping','dangling','body','duplicate','field_path','claim_target'])
def test_invalid_data_never_selected_or_excluded(selector, case, defect):
    label,bundle = case
    if defect=='quote': label['evidence_refs'][0]['quote']='不存在的原文'
    if defect=='role': bundle['events'][0]['role']='assistant'
    if defect=='mapping': label['capability_mapping'][0]['parent_requirement_id']='req_002'
    if defect=='dangling': label['claims'][0]['evidence_ids']=['missing']
    if defect=='body': bundle['events'][0]['body_provided']=False
    if defect=='duplicate': label['claims'].append(copy.deepcopy(label['claims'][0]))
    if defect=='field_path': label['claims'][0]['field_path']='missing.field'
    if defect=='claim_target': label['claims'][2]['target_id']='ep1'
    a=audit(selector,label,dict(S4='fail'))
    result=selector.select(label,bundle,audits=[a])
    assert result['decision']=='review'
    assert result['prechecks']['valid_label_and_references']=='pending'


def test_missing_evidence_bundle(selector,case):
    assert selector.select(case[0],None)['decision']=='review'


def test_auxiliary_only_needs_negative_audit(selector,case):
    label,bundle=case
    label['capability_mapping'][0].update(taxonomy_row=209,parent_requirement_id='req_034')
    assert selector.select(label,bundle)['decision']=='review'
    a=audit(selector,label,dict(S1='fail'))
    result=selector.select(label,bundle,audits=[a])
    assert result['decision']=='not_selected'
    assert result['primary_requirement_ids']==[]


def test_parent_only_is_pending(selector,case):
    label,bundle=case
    label['capability_mapping'][0].update(taxonomy_row=None,mapping_status='partial')
    result=selector.select(label,bundle)
    assert result['primary_requirement_ids']==['req_001']
    assert result['rule_checks']['S1']['status']=='pending'


def test_low_requires_actual_verification(selector,case):
    label,bundle=case
    label['task_attributes']['difficulty']='low'
    assert selector.select(label,bundle)['decision']=='review'
    a=audit(selector,label,dict(S2='fail'));a['rule_checks']['S2']['claim_ids']=['difficulty']
    assert selector.select(label,bundle,audits=[a])['decision']=='not_selected'


@pytest.mark.parametrize('defect',['stale','missing_evidence','unsupported','duplicate_audit','policy','unknown_claim'])
def test_audit_integrity(selector,case,defect):
    label,bundle=case
    if defect=='unsupported': label['claims'][3]['support_status']='unsupported'
    a=audit(selector,label)
    if defect=='stale': a['label_sha256']='0'*64
    if defect=='policy': a['policy_sha256']='0'*64
    if defect=='missing_evidence': a['rule_checks']['S4']['evidence_ids']=['missing']
    if defect=='unknown_claim': a['rule_checks']['S4']['claim_ids']=['missing']
    audits=[a,a] if defect=='duplicate_audit' else [a]
    assert selector.select(label,bundle,audits=audits)['decision']=='review'


def test_failed_rule_beats_unrelated_unknown(selector,case):
    a=audit(selector,case[0],dict(S4='fail'))
    assert selector.select(*case,audits=[a])['decision']=='not_selected'


def test_local_does_not_inherit_episode_difficulty(selector,case):
    a=audit(selector,case[0],mode='case_requirement',target='r1')
    r=selector.select(*case,audits=[a],mode='case_requirement',requirement_id='r1')
    assert r['rule_checks']['S2']['status']=='pending'
    a['rule_checks']['S2']=dict(status='pass',reason='核验该局部任务本身具有中等难度',claim_ids=['requirement'],evidence_ids=['e1'])
    assert selector.select(*case,audits=[a],mode='case_requirement',requirement_id='r1')['decision']=='priority_candidate'


def test_jsonl_cli_outputs_and_no_overwrite(tmp_path,selector,case):
    label,bundle=case
    def save(name,rows):
        p=tmp_path/name;p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));return p
    labels=save('labels.jsonl',[label,label]); evidence=save('evidence.jsonl',[bundle]); audits=save('audits.jsonl',[audit(selector,label)])
    out=tmp_path/'run'
    summary=run_selection(labels,out,evidence_path=evidence,audits_path=audits)
    assert summary['counts']['priority_candidate']==1
    assert summary['duplicate_records_collapsed']==1
    assert (out/'selection_report.md').exists()
    with pytest.raises(FileExistsError): run_selection(labels,out,evidence_path=evidence)


def test_conflicting_versions_rejected(tmp_path,case):
    label,_=case; other=copy.deepcopy(label);other['task_attributes']['difficulty']='high'
    p=tmp_path/'labels.jsonl';p.write_text(json.dumps(label)+'\n'+json.dumps(other)+'\n')
    with pytest.raises(ValueError,match='Conflicting label'):run_selection(p,tmp_path/'out')


def test_command_parser():
    from trace_analysis.cli_current import parser
    args=parser().parse_args(['pipeline','select','--labels','labels.jsonl','--output-dir','run','--sources','sources.jsonl'])
    assert args.pipeline_action=='select'
    assert args.sources=='sources.jsonl'


def claim_review(selector,label,claim_id='difficulty',status='supported'):
    return dict(schema_version='stage06-v1-draft',review_id='rev1',case_id=label['case_id'],
                label_sha256=content_hash(label),reviewer_type='human',reviewer_id='tester',
                review_time='2026-09-17T00:00:00Z',reviewed_claim_ids=[claim_id],status='accepted',
                saw_proposed_label=True,reason='核对原文',evidence_ids=['e1'],corrections=[],resulting_label_sha256=None,
                claim_reviews=[dict(claim_id=claim_id,support_status=status,reason='核对原文',evidence_ids=['e1'],missing_evidence_queries=[])])


def test_claim_review_can_overturn_initial_supported(selector,case):
    label,bundle=case
    review=claim_review(selector,label,status='unsupported')
    result=selector.select(label,bundle,audits=[audit(selector,label)],reviews=[review])
    assert result['decision']=='review'
    assert result['rule_checks']['S2']['status']=='pending'


def test_reviewed_low_excluded(selector,case):
    label,bundle=case;label['task_attributes']['difficulty']='low'
    result=selector.select(label,bundle,reviews=[claim_review(selector,label)])
    assert result['decision']=='not_selected'


def test_conflicting_reviews_pending(selector,case):
    label,bundle=case
    r1=claim_review(selector,label);r2=claim_review(selector,label,status='unsupported');r2['review_id']='rev2'
    assert selector.select(label,bundle,audits=[audit(selector,label)],reviews=[r1,r2])['decision']=='review'


def test_missing_formal_reviews_not_ignored(selector,case):
    label,bundle=case
    formal=dict(case_id=label['case_id'],model_output=label,review_record_paths=['review.jsonl'])
    assert selector.select(formal,bundle,audits=[audit(selector,formal)])['decision']=='review'


def test_scope_filter_uses_supported_claims(tmp_path,case):
    label,bundle=case
    selector=Selector()
    selector.policy['scope_filters']['work_objects']=['代码']
    assert selector.select(label,bundle)['prechecks']['scope_match']=='pending'
    c=copy.deepcopy(label['claims'][0]);c.update(claim_id='object',field_path='task_attributes.work_object_primary')
    label['claims'].append(c)
    assert selector.select(label,bundle)['prechecks']['scope_match']=='pass'


def test_s1_wrong_claim_field_rejected(selector,case):
    label,bundle=case
    label['claims'][2]['field_path']='case_requirements[0].text'
    assert selector.select(label,bundle)['prechecks']['valid_label_and_references']=='pending'


class AuditClient:
    model='test-model'
    def __init__(self, status='pass', defect=None):
        self.status=status;self.defect=defect;self.calls=0
    def complete_json(self,system,user):
        self.calls+=1
        self.payload=json.loads(user)
        if self.defect=='exception':raise RuntimeError('provider unavailable')
        checks={k:dict(status=self.status,reason='已有原文核验',claim_ids=['requirement'],evidence_ids=['e1']) for k in self.payload['requested_rules']}
        if self.defect=='reference':checks['S3']['evidence_ids']=['missing']
        if self.defect=='extra':checks['S1']=checks['S3']
        return dict(rule_checks=checks),dict(cache_hit=False)


def run_model(tmp_path,case,client,**kwargs):
    for n,value in zip(['labels','evidence'],case):
        (tmp_path/f'{n}.jsonl').write_text(json.dumps(value,ensure_ascii=False)+'\n')
    summary=run_selection(tmp_path/'labels.jsonl',tmp_path/'run',evidence_path=tmp_path/'evidence.jsonl',client=client,**kwargs)
    result=json.loads((tmp_path/'run/selection.jsonl').read_text())
    return summary,result


def test_model_pipeline_completes_s3_s4(tmp_path,case):
    client=AuditClient();summary,result=run_model(tmp_path,case,client)
    assert result['decision']=='priority_candidate'
    assert summary['model_calls']==1 and summary['model_audits_accepted']==1
    a=json.loads((tmp_path/'run/model_selection_audits.jsonl').read_text())
    assert a['reviewer_type']=='model' and a['reviewer_id']=='test-model'
    assert client.payload['requested_rules']==['S3','S4']
    assert client.payload['evidence_bundle']==case[1]
    assert 'episode_assessment' not in client.payload['labels']


def test_priority_sessions_exported_automatically_from_sibling_sources(tmp_path,case):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import read_jsonl
    path='/recorded/workspace/_internal/session.jsonl'
    source=dict(case_id='case1',session_ids=['real-session-id'],source_row=dict(
        batch_id='batch_demo',analysis_run_id='run_demo',pain_judgment='review',
        source_references=[dict(source=dict(input_file=path))]))
    (tmp_path/'sources.jsonl').write_text(json.dumps(source)+'\n')
    output=tmp_path/'session_paths__priority_candidate__batch_demo__run_demo.jsonl'
    output.write_text('{"session_id":"previous-selection"}\n')
    summary,result=run_model(tmp_path,case,AuditClient())
    assert result['decision']=='priority_candidate'
    assert read_jsonl(output)==[dict(session_id='real-session-id',session_jsonl_path=path,
        workspace_path='/recorded/workspace',case_ids=['case1'])]
    assert summary['session_export']['count']==1
    assert summary['session_export']['status']=='complete'
    assert str((tmp_path/'sources.jsonl').resolve()) in summary['inputs']
    assert not (tmp_path/'run'/output.name).exists()
    assert f'(../{output.name})' in (tmp_path/'run/selection_report.md').read_text()
    report=tmp_path/'06阶段标签质量与分布审查报告.md'
    assert report.exists()
    assert summary['distribution_report']['priority_cases']==1
    assert summary['distribution_report']['covered_requirements']==1
    assert summary['distribution_report']['covered_subrequirements']==1
    assert '完整 36 项' in report.read_text() and '完整 216 项' in report.read_text()
    assert '| req_001 | 文献调研 | 主需求 | 1 | 1 | 100.0% | 1 |' in report.read_text()


def test_session_export_deduplicates_only_priority_candidates_and_does_not_guess_paths(tmp_path):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.session_export import export_priority_sessions
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import read_jsonl
    results=[dict(case_id=k,decision=d) for k,d in [('a','priority_candidate'),
        ('b','priority_candidate'),('c','review'),('d','not_selected')]]
    sources={k:dict(session_ids=ids) for k,ids in [('a',['shared']),('b',['shared','second']),
        ('c',['review-only']),('d',['excluded-only'])]}
    sources['a']['source_references']=[dict(source=dict(input_file='/one/_internal/session.jsonl'))]
    sources['b']['source_references']=[dict(source=dict(input_file='/ambiguous/_internal/session.jsonl'))]
    summary=export_priority_sessions(tmp_path,results,sources)
    rows={r['session_id']:r for r in read_jsonl(tmp_path/'session_paths__priority_candidate.jsonl')}
    assert set(rows)=={'shared','second'}
    assert rows['shared']['case_ids']==['a','b']
    assert rows['shared']['session_jsonl_path']=='/one/_internal/session.jsonl'
    assert rows['second']['session_jsonl_path'] is None
    assert summary['unresolved_path_session_ids']==['second']
    with pytest.raises(FileExistsError):export_priority_sessions(tmp_path,results,sources)


def test_session_export_empty_pool_and_missing_ids_are_explicit(tmp_path):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.session_export import export_priority_sessions
    empty=tmp_path/'empty';empty.mkdir()
    summary=export_priority_sessions(empty,[dict(case_id='a',decision='review')],{})
    assert summary['status']=='complete' and summary['count']==0
    assert (empty/'session_paths__priority_candidate.jsonl').read_text()==''
    missing=tmp_path/'missing';missing.mkdir()
    summary=export_priority_sessions(missing,[dict(case_id='a',decision='priority_candidate')],{})
    assert summary['status']=='incomplete' and summary['missing_session_case_ids']==['a']


def test_requirement_distribution_counts_cases_and_distinct_children(selector,case):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.distribution_report import requirement_distribution
    label,_=case
    # One case maps repeatedly to row 2 and also to a second child under req_001.
    for row,status in [(2,'supported'),(3,'supported'),(4,'partial')]:
        index=len(label['capability_mapping']);cid=f'map-claim-{index}'
        mapping={**label['capability_mapping'][0], 'mapping_id':f'm{index+1}',
                 'taxonomy_row':row if status=='supported' else None,
                 'mapping_status':status,'claim_ids':[cid]}
        label['capability_mapping'].append(mapping)
        claim=copy.deepcopy(label['claims'][2]);claim.update(claim_id=cid,
            field_path=f'capability_mapping[{index}].taxonomy_row',support_status=status)
        label['claims'].append(claim)
    second=copy.deepcopy(label);second['case_id']='case2'
    # Existing optional mapping is disputed, so it cannot inflate child coverage.
    second['claims'][-2]['support_status']='unsupported'
    third=copy.deepcopy(label);third['case_id']='review-case'
    labels={x['case_id']:x for x in [label,second,third]}
    results=[dict(case_id=cid,mode='episode',assessment_id='ep1',decision=decision)
             for cid,decision in [('case1','priority_candidate'),('case2','priority_candidate'),('review-case','review')]]
    value=requirement_distribution(labels,results,selector)
    assert value['parents']['req_001']['priority_cases']=={'case1','case2'}
    assert value['parents']['req_001']['children']=={2,3}
    assert value['children'][2]['priority_cases']=={'case1','case2'}
    assert value['children'][3]['priority_cases']=={'case1'}
    assert value['children'][4]['priority_cases']==set()
    assert len(value['unresolved'])==3
    assert value['children'][2]['all_cases']=={'case1','case2','review-case'}
    review=claim_review(selector,second,claim_id=second['claims'][-2]['claim_id'])
    updated=requirement_distribution(labels,results,selector,reviews=[review])
    assert updated['children'][3]['priority_cases']=={'case1','case2'}


def test_requirement_distribution_local_scope_does_not_import_other_requirements(selector,case):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.distribution_report import requirement_distribution
    label,_=case
    extra={**label['capability_mapping'][0],'mapping_id':'m2','case_requirement_id':'r2',
           'taxonomy_row':3,'claim_ids':['mapping2']}
    label['capability_mapping'].append(extra)
    claim=copy.deepcopy(label['claims'][2]);claim.update(claim_id='mapping2',target_id='r2',
        field_path='capability_mapping[1].taxonomy_row');label['claims'].append(claim)
    selected=dict(case_id='case1',mode='case_requirement',assessment_id='r1',decision='priority_candidate')
    value=requirement_distribution({'case1':label},[selected],selector)
    assert value['parents']['req_001']['children']=={2}
    both=requirement_distribution({'case1':label},[selected,{**selected,'assessment_id':'r2'}],selector)
    assert both['priority_units']==2 and both['priority_case_ids']=={'case1'}
    assert both['parents']['req_001']['children']=={2,3}


def test_empty_priority_pool_still_produces_full_distribution_report(tmp_path,case):
    summary,_=run_model(tmp_path,case,AuditClient(status='pending'))
    report=(tmp_path/'06阶段标签质量与分布审查报告.md').read_text()
    assert summary['distribution_report']['priority_cases']==0
    assert summary['distribution_report']['covered_requirements']==0
    assert summary['distribution_report']['covered_subrequirements']==0
    assert '| req_001 | 文献调研 | 主需求 | 1 | 0 | — | 0 |' in report
    assert '完整 216 项' in report


def test_distribution_report_keeps_malformed_label_as_review(tmp_path,case):
    case[0]['claims']=None
    case[0]['task_attributes']=None
    client=AuditClient()
    summary,result=run_model(tmp_path,case,client)
    assert result['decision']=='review' and client.calls==0
    assert summary['distribution_report']['priority_cases']==0
    assert '无法按标签 schema 解析的案例为 1 条' in (tmp_path/'06阶段标签质量与分布审查报告.md').read_text()


@pytest.mark.parametrize('defect',['reference','extra','exception'])
def test_model_failures_stay_pending(tmp_path,case,defect):
    summary,result=run_model(tmp_path,case,AuditClient(defect=defect))
    assert result['decision']=='review'
    assert summary['model_audits_accepted']==0
    assert (tmp_path/'run/model_review_log.jsonl').exists()


@pytest.mark.parametrize('defect', ['unknown_evidence', 'rule_set'])
def test_model_repairs_invalid_audit_once(tmp_path,case,defect):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import read_jsonl
    label,_=case
    label['evidence_refs'].append({**label['evidence_refs'][0], 'evidence_id':'e2'})
    class RepairClient(AuditClient):
        def complete_json(self,system,user):
            response,metadata=super().complete_json(system,user)
            if self.calls==1:
                if defect=='unknown_evidence':
                    response['rule_checks']['S3']['evidence_ids']=['missing']
                else:
                    response['rule_checks']['S1']=response['rule_checks']['S3']
            else:
                assert self.calls==2 and self.payload['repair']['validation_error']
                links={c['claim_id']:c for c in self.payload['claim_evidence_links']}
                assert links['requirement']['evidence_ids']==['e1']
                if defect=='unknown_evidence':
                    assert '不存在' in self.payload['repair']['validation_error']
            return response,metadata
    client=RepairClient()
    summary,result=run_model(tmp_path,case,client)
    assert summary['model_calls']==2 and summary['model_audits_accepted']==1
    assert result['decision']=='priority_candidate'
    logs=read_jsonl(tmp_path/'run/model_review_log.jsonl')
    assert [x['status'] for x in logs]==[('rejected' if defect=='unknown_evidence' else 'error'),'accepted']
    assert [x['repair_attempt'] for x in logs]==[False,True]
    assert all('_invalid_response' not in x for x in logs)
    assert len(read_jsonl(tmp_path/'run/selection_checkpoint.jsonl'))==1
    assert len(read_jsonl(tmp_path/'run/model_selection_audits.jsonl'))==1


def test_pending_audit_is_accepted_without_retry(tmp_path,case):
    client=AuditClient(status='pending')
    summary,result=run_model(tmp_path,case,client)
    assert client.calls==1 and summary['model_audits_accepted']==1
    assert result['decision']=='review'


def test_verified_case_evidence_need_not_be_linked_to_selected_claim(tmp_path,case):
    label, bundle = case
    ref = copy.deepcopy(label['evidence_refs'][0])
    ref.update(evidence_id='e2', event_id='output', turn_id='turn2',
               source_role='assistant', quote='产物已导出',
               locator={**ref['locator'], 'record_index':1, 'physical_line':2})
    label['evidence_refs'].append(ref)
    bundle['events'].append(dict(event_id='output', turn_id='turn2', role='assistant',
        content=ref['quote'], body_provided=True, locator=ref['locator'], episode_membership='episode'))
    bundle['scope']['provided_event_ids'].append('output')
    class DirectEvidenceClient(AuditClient):
        def complete_json(self,system,user):
            response, metadata = super().complete_json(system,user)
            response['rule_checks']['S4']['evidence_ids'] = ['e1', 'e2']
            response['rule_checks']['S4']['reason'] = '用户要求给出可核对标准，助手产物记录仅补充交付形式。'
            return response, metadata
    client = DirectEvidenceClient()
    summary, result = run_model(tmp_path,case,client)
    assert client.calls == 1 and summary['model_audits_accepted'] == 1
    assert result['decision'] == 'priority_candidate'
    assert result['rule_checks']['S4']['evidence_ids'] == ['e1','e2']


def test_s5_is_absent_from_current_decision_and_rejected_in_new_audits(selector,case):
    result = selector.select(*case, audits=[audit(selector,case[0])])
    assert set(result['rule_checks']) == {'S1','S2','S3','S4'}
    assert result['decision'] == 'priority_candidate'
    for status in ['pass','pending','fail']:
        extra = audit(selector,case[0],dict(S3='pass',S4='pass',S5=status))
        invalid = selector.select(*case,audits=[extra])
        assert invalid['prechecks']['valid_label_and_references'] == 'pending'


def test_explicit_policy_migration_preserves_judgments_and_provenance(tmp_path,case,selector):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import read_jsonl
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection_migration import migrate_v2_audits
    old_policy = copy.deepcopy(selector.policy)
    old_policy['version'] = 'stage06-selection-v2'
    old_policy['rules'].append(dict(rule_id='S5',name='材料',criterion='材料是否可恢复'))
    policy_path = tmp_path/'old-policy.json'
    policy_path.write_text(json.dumps(old_policy))
    old_selector = Selector(policy_path)
    label,bundle = case
    original = audit(old_selector,label,dict(S3='pass',S4='pending',S5='fail'))
    paths = []
    for name, row in [('labels',label),('evidence',bundle),('audits',original)]:
        p=tmp_path/f'{name}.jsonl';p.write_text(json.dumps(row)+'\n');paths.append(p)
    before = old_selector.select(label,bundle,audits=[original])
    assert before['decision'] == 'not_selected'
    out = tmp_path/'migrated.jsonl'
    migrate_v2_audits(paths[0],paths[1],[paths[2]],policy_path,out)
    migrated = read_jsonl(out)[0]
    for field in ['reviewer_type','reviewer_id','review_time','label_sha256']:
        assert migrated[field] == original[field]
    assert migrated['rule_checks'] == {k:v for k,v in original['rule_checks'].items() if k!='S5'}
    assert migrated['policy_migration']['source_audit_sha256'] == content_hash(original)
    assert migrated['policy_sha256'] == selector.policy_sha256
    assert read_jsonl(paths[2]) == [original]
    result=selector.select(label,bundle,audits=[migrated])
    assert result['decision'] == 'review' and result['rule_checks']['S4']['status'] == 'pending'
    with pytest.raises(FileExistsError):
        migrate_v2_audits(paths[0],paths[1],[paths[2]],policy_path,out)
    label['task_attributes']['difficulty']='high'
    paths[0].write_text(json.dumps(label)+'\n')
    with pytest.raises(ValueError,match='invalid source audit'):
        migrate_v2_audits(paths[0],paths[1],[paths[2]],policy_path,tmp_path/'stale.jsonl')
    assert not (tmp_path/'stale.jsonl').exists()
    old_policy['rules'][2]['criterion']='changed meaning'
    policy_path.write_text(json.dumps(old_policy))
    with pytest.raises(ValueError,match='Changed rule definition'):
        migrate_v2_audits(paths[0],paths[1],[paths[2]],policy_path,tmp_path/'changed.jsonl')


def test_repair_feedback_reports_all_invalid_rules(tmp_path,case):
    label,_=case
    label['evidence_refs'].append({**label['evidence_refs'][0], 'evidence_id':'e2'})
    class MultipleErrorsClient(AuditClient):
        def complete_json(self,system,user):
            response,metadata=super().complete_json(system,user)
            if self.calls==1:
                for rule in ['S3','S4']:
                    response['rule_checks'][rule]['evidence_ids']=['missing']
            else:
                errors=self.payload['repair']['rule_validation_errors']
                assert set(errors)=={'S3','S4'}
                assert all('不存在' in message for message in errors.values())
            return response,metadata
    summary,result=run_model(tmp_path,case,MultipleErrorsClient())
    assert summary['model_calls']==2 and result['decision']=='priority_candidate'


def test_model_negative_evidence_can_exclude(tmp_path,case):
    _,result=run_model(tmp_path,case,AuditClient(status='fail'))
    assert result['decision']=='not_selected'


def test_model_no_truncation_or_call_when_too_large(tmp_path,case):
    client=AuditClient();summary,result=run_model(tmp_path,case,client,max_input_chars=1)
    assert client.calls==0 and summary['model_calls']==0
    assert result['decision']=='review'


def test_model_not_called_on_invalid_label(tmp_path,case):
    case[0]['evidence_refs'][0]['quote']='不存在'
    client=AuditClient();summary,result=run_model(tmp_path,case,client)
    assert client.calls==0 and result['decision']=='review'


def test_model_preserves_supplied_audits(tmp_path,case,selector):
    supplied=audit(selector,case[0],dict(S3='pending'))
    p=tmp_path/'audit.jsonl';p.write_text(json.dumps(supplied)+'\n')
    client=AuditClient();summary,result=run_model(tmp_path,case,client,audits_path=p)
    assert client.payload['requested_rules']==['S4']
    assert result['rule_checks']['S3']['status']=='pending'
    assert result['decision']=='review'


def test_existing_output_rejected_before_model_call(tmp_path,case):
    (tmp_path/'run').mkdir();client=AuditClient()
    with pytest.raises(FileExistsError):run_model(tmp_path,case,client)
    assert client.calls==0


def test_mapping_claim_can_reference_whole_mapping(selector,case):
    label,bundle=case
    label['claims'][2]['field_path']='capability_mapping[0]'
    result=selector.select(label,bundle)
    assert result['prechecks']['valid_label_and_references']=='pass'
    assert result['rule_checks']['S1']['status']=='pass'


def test_mapping_names_are_resolved_without_changing_judgments(selector,case):
    label,bundle=case
    value=selector.enrich_names(label)
    m=value['capability_mapping'][0]
    assert m['requirement_name']=='文献调研'
    assert m['subrequirement_id']=='sub_002'
    assert m['subrequirement_name']=='制定文献调研协议'
    assert m['case_requirement_text']==label['case_requirements'][0]['text']
    assert 'requirement_name' not in label['capability_mapping'][0]
    assert selector.select(value,bundle)['rule_checks']==selector.select(label,bundle)['rule_checks']
    m['requirement_name']='错误名称'
    assert selector.select(value,bundle)['prechecks']['valid_label_and_references']=='pending'


def test_mapping_names_preserve_partial_and_unknown(selector,case):
    label,_=case
    m=label['capability_mapping'][0]
    m.update(mapping_status='partial',taxonomy_row=None)
    v=selector.enrich_names(label)['capability_mapping'][0]
    assert v['requirement_name']=='文献调研' and v['subrequirement_name'] is None
    m.update(mapping_status='unresolved',parent_requirement_id=None)
    assert selector.enrich_names(label)['capability_mapping'][0]['requirement_name'] is None
def test_jsonl_reader_preserves_unicode_line_separators(tmp_path):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import read_jsonl
    import json
    path=tmp_path/'evidence.jsonl'
    rows=[dict(content='第一行\u2028第二行\u2029尾行'),dict(content='正常')]
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    assert read_jsonl(path)==rows


def test_parallel_selection_checkpoints_each_case_once(tmp_path,case):
    from threading import Barrier
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import read_jsonl
    barrier=Barrier(2)
    class ConcurrentClient:
        model='test-model'
        def complete_json(self,system,user):
            payload=json.loads(user)
            barrier.wait(timeout=5)
            return dict(rule_checks={rule:dict(status='pass',reason='原文核验',claim_ids=['requirement'],evidence_ids=['e1'])
                                      for rule in payload['requested_rules']}),dict(cache_hit=False)
    labels=[];bundles=[]
    for case_id in ['case1','case2']:
        label,bundle=copy.deepcopy(case);label['case_id']=case_id;bundle['case_id']=case_id
        labels.append(label);bundles.append(bundle)
    for name,rows in [('labels',labels),('evidence',bundles)]:
        (tmp_path/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary=run_selection(tmp_path/'labels.jsonl',tmp_path/'run',evidence_path=tmp_path/'evidence.jsonl',
                          client=ConcurrentClient(),workers=2)
    assert summary['workers']==2 and summary['counts']['priority_candidate']==2
    assert summary['model_calls']==2 and summary['model_audits_accepted']==2
    for filename in ['selection.jsonl','selection_checkpoint.jsonl','model_selection_audits.jsonl','model_review_log.jsonl']:
        rows=read_jsonl(tmp_path/'run'/filename)
        assert len(rows)==2 and {row['case_id'] for row in rows}=={'case1','case2'}
