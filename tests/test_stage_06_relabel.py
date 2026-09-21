from pathlib import Path
import json
from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import make_bundle
from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection import Selector


def test_requirement_id_paths_are_resolved_only_with_matching_unique_target():
    from copy import deepcopy
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import normalize_serialization
    label = dict(case_requirements=[dict(case_requirement_id='req_other', text='其他'),
                                   dict(case_requirement_id='req_write', text='写论文')],
                 claims=[dict(claim_id='c', target_id='req_write',
                              field_path='case_requirements[req_write].text')])
    original = deepcopy(label)
    result, changes = normalize_serialization(label, dict(case_id='case', events=[]), 'episode')
    assert result['claims'][0]['field_path'] == 'case_requirements[1].text'
    assert changes[0]['operation'] == 'requirement_id_to_index'
    assert label == original
    for defect in ['duplicate', 'target', 'missing_field', 'numeric_out_of_range']:
        broken = deepcopy(label)
        if defect == 'duplicate': broken['case_requirements'].append(broken['case_requirements'][1])
        if defect == 'target': broken['claims'][0]['target_id'] = 'req_other'
        if defect == 'missing_field': broken['claims'][0]['field_path'] = 'case_requirements[req_write].absent'
        if defect == 'numeric_out_of_range': broken['claims'][0]['field_path'] = 'case_requirements[99].text'
        unchanged, changes = normalize_serialization(broken, dict(case_id='case', events=[]), 'episode')
        assert unchanged == broken and changes == []


def test_bundle_preserves_roles_boundary_and_exact_text():
    row={'lineage':{'preprocessed_input':'source.jsonl'}}
    episode={'episode_evidence_spans':[{'post_episode_context_turn_id':'t2'}], 'episode_turns':[
        {'turn_id':'t1','events':[{'event_id':'u1','type':'user_message','data':{'content':'原文'}},
                                {'event_id':'a1','type':'tool_call','data':{'name':'read','arguments':'file'}}]},
        {'turn_id':'t2','events':[{'event_id':'r1','type':'tool_result','data':{'content':'结果'}}]}]}
    b=make_bundle('case',row,episode,{},'0'*64, {
        (str(Path('source.jsonl').resolve()), t['turn_id']): dict(source_file=str(Path('source.jsonl').resolve()), record_index=i, physical_line=i+1)
        for i,t in enumerate(episode['episode_turns'])})
    Selector().validate('evidence.schema.json',b)
    assert [e['role'] for e in b['events']]==['user','assistant','tool']
    assert b['events'][0]['content']=='原文'
    assert b['events'][2]['episode_membership']=='post_episode_context'
    assert b['scope']['full_episode_read']


def test_large_bundle_does_not_claim_omitted_tools_read():
    row={'lineage':{'preprocessed_input':'source.jsonl'}}
    episode={'trace_id':'trace','episode_evidence_spans':[{}],'episode_turns':[{'turn_id':'t','events':[
        {'event_id':'u','type':'user_message','data':{'content':'请分析'}},
        {'event_id':'tool','type':'tool_result','data':{'content':'x'*210000}}]}]}
    b=make_bundle('case',row,episode,{},'0'*64, {
        (str(Path('source.jsonl').resolve()), t['turn_id']): dict(source_file=str(Path('source.jsonl').resolve()), record_index=i, physical_line=i+1)
        for i,t in enumerate(episode['episode_turns'])})
    Selector().validate('evidence.schema.json',b)
    assert not b['scope']['full_episode_read']
    assert b['scope']['omitted_body_event_ids']==['tool']
    assert not b['events'][1]['body_provided']


def test_source_locations_use_actual_jsonl_lines(tmp_path):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import source_locations
    p=tmp_path/'turns.jsonl'
    p.write_text('\n'+json.dumps(dict(turn_id='t1'))+'\n\n'+json.dumps(dict(turn_id='t2'))+'\n')
    locations=source_locations([(1,dict(lineage=dict(preprocessed_input=str(p))))])
    assert locations[(str(p),'t2')]==dict(source_file=str(p),record_index=1,physical_line=4)


def test_failure_retains_cache_pointers_without_per_case_files(tmp_path,monkeypatch):
    from trace_analysis.pipeline.stage_06_trace_quality_labeling import relabel
    p=tmp_path/'turns.jsonl';p.write_text(json.dumps(dict(turn_id='t1',trace_id='trace',batch_id='batch',
        turn_index=1,events=[dict(event_id='e',type='user_message',data=dict(content='请求'))]))+'\n')
    row=dict(episode_id='ep',trace_id='trace',batch_id='batch',pain_judgment='confirmed',
             episode_boundary=dict(start_turn_id='t1',end_turn_id='t1'),lineage=dict(preprocessed_input=str(p)))
    input_path=tmp_path/'input.jsonl';input_path.write_text(json.dumps(row)+'\n')
    old=tmp_path/'old.jsonl';old.write_text('')
    class Client:
        model='test-model'
        cache_dir=tmp_path/'cache'
        def complete_json(self,system,user):return {}, dict(cache_key='a'*64)
    output=tmp_path/'run'
    result=relabel.run(input_path,old,output,Client(),workers=1)
    assert result['failed']==1
    assert not (output/'raw_episodes').exists()
    assert not list(output.glob('response_*.json'))
    error=json.loads((output/'label_errors.jsonl').read_text())
    assert len(error['attempts'])==4
    assert error['attempts'][0]['metadata']['cache_key']=='a'*64
    assert all('response' not in attempt for attempt in error['attempts'])


def test_retry_locates_changed_quote_and_wrong_event_without_mutating():
    from copy import deepcopy
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import validation_feedback
    event=dict(event_id='e1',turn_id='t1',role='assistant',locator={},episode_membership='episode',
               body_provided=True,content='**市场收入**：3507亿元\n**用户规模**：6.83亿人')
    other={**event,'event_id':'e2','content':'需要导出文档吗？'}
    label=dict(evidence_refs=[
        dict(evidence_id='bad1',event_id='e1',quote='市场收入3507亿元，用户规模6.83亿人'),
        dict(evidence_id='bad2',event_id='e1',quote='需要导出文档吗？')])
    original=deepcopy(label)
    feedback=validation_feedback(label,dict(events=[event,other]),ValueError('引文不匹配'))
    details=json.loads(feedback.split('\n')[-1])
    assert details[0]['evidence_id']=='bad1'
    assert details[0]['source_event_content']==event['content']
    assert details[1]['quote_matches_other_provided_event_ids']==['e2']
    assert '重核关联claim' in feedback
    assert label==original


def test_retry_excerpt_is_exact_and_bounded():
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import validation_feedback
    content='a'*20000+'**关键结论**：有差异。'+'z'*20000
    event=dict(event_id='e',turn_id='t',role='assistant',locator={},episode_membership='episode',
               body_provided=True,content=content)
    feedback=validation_feedback(dict(evidence_refs=[dict(evidence_id='ref',event_id='e',quote='关键结论：有差异。')]),
                                 dict(events=[event]),ValueError('quote'))
    excerpt=json.loads(feedback.split('\n')[-1])[0]['source_event_excerpt']
    assert excerpt['text']==content[excerpt['start']:excerpt['end']]
    assert '**关键结论**：有差异。' in excerpt['text']
    assert len(feedback)<16000


def test_locator_resolution_requires_exact_quote_and_identity():
    from copy import deepcopy
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import resolve_exact_citation_locators
    event=dict(event_id='e',turn_id='t',role='user',episode_membership='episode',body_provided=True,
               content='原文要求',locator=dict(source_file='actual.jsonl',record_index=7,physical_line=8))
    ref=dict(evidence_id='ref',event_id='e',turn_id='t',source_role='user',episode_membership='episode',
             quote='原文要求',locator=dict(source_file='typo.jsonl',record_index=7,physical_line=8))
    label=dict(evidence_refs=[ref]);original=deepcopy(label)
    resolved,changes=resolve_exact_citation_locators(label,dict(events=[event]))
    assert resolved['evidence_refs'][0]['locator']==event['locator']
    assert changes[0]['previous']==ref['locator']
    assert label==original
    for field,value in [('quote','改写的要求'),('source_role','assistant'),('turn_id','wrong'),('episode_membership','post_episode_context')]:
        broken=deepcopy(label);broken['evidence_refs'][0][field]=value
        resolved,changes=resolve_exact_citation_locators(broken,dict(events=[event]))
        assert resolved==broken and changes==[]


def test_retry_identifies_unprovided_event_and_visible_chunk():
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import validation_feedback
    event=dict(event_id='tool#chunk_0001',body_provided=True,content='实际展示片段')
    label=dict(evidence_refs=[dict(evidence_id='ref',event_id='tool',quote='实际展示片段')])
    feedback=validation_feedback(label,dict(events=[event]),ValueError('未提供正文'))
    detail=json.loads(feedback.split('\n')[-1])[0]
    assert detail['provided_chunk_ids']==['tool#chunk_0001']
    assert detail['quote_matches_other_provided_event_ids']==['tool#chunk_0001']
    assert '不能引用未展示' in detail['error']


def test_request_compaction_preserves_all_visible_evidence_and_scope():
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import prompt_bundle
    bundle=dict(case_id='case',scope=dict(omitted_body_event_ids=['omitted']),events=[
        dict(event_id='shown',body_provided=True,content=' 原文\n第二行 '),
        dict(event_id='omitted',body_provided=False,content='')])
    compact=prompt_bundle(bundle)
    assert compact['scope']==bundle['scope']
    assert compact['events']==[bundle['events'][0]]
    assert len(bundle['events'])==2


def test_serialization_fixes_only_exact_encoding_and_unambiguous_identity():
    from copy import deepcopy
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import normalize_serialization
    text='前一行\n后一行'
    bundle=dict(case_id='ep__line_1',events=[dict(event_id='tool',body_provided=True,
                 content=json.dumps(dict(content=text),ensure_ascii=False))])
    label=dict(case_id='ep__line_1',assessment_target=dict(target_id='ep__line_1'),
               case_requirements=[dict(case_requirement_id='r')],claims=[dict(target_id='ep__line_1'),dict(target_id='r')],
               evidence_refs=[dict(evidence_id='ev',event_id='tool',quote=text)])
    label['']=''
    original=deepcopy(label)
    result,changes=normalize_serialization(label,bundle,'ep')
    assert result['assessment_target']['target_id']=='ep'
    assert [c['target_id'] for c in result['claims']]==['ep','r']
    assert result['evidence_refs'][0]['quote'] in bundle['events'][0]['content']
    assert json.loads('"'+result['evidence_refs'][0]['quote']+'"')==text
    assert '' not in result and len(changes)==3
    assert label==original
    label['']='nonempty';label['assessment_target']['target_id']='other_episode'
    label['evidence_refs'][0]['quote']='前一行\n改写后一行'
    result,changes=normalize_serialization(label,bundle,'ep')
    assert result==label and not changes


def test_schema_error_message_is_concise_and_contains_path():
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import error_message
    from jsonschema import ValidationError
    message=error_message(ValidationError('unexpected property',path=['claims',2],schema={'description':'x'*5000}))
    assert message=='claims.2: unexpected property'


def test_context_retry_preserves_prompt_and_shared_client_budget():
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import complete_label_json
    class ContextError(Exception):
        status_code=400
    class Client:
        max_completion_tokens=32768
        calls=[]
        def complete_json(self,system,user):
            self.calls.append((system,user,self.max_completion_tokens))
            if self.max_completion_tokens==32768:
                raise ContextError('maximum context length is 262144 tokens')
            return {'unchanged_response':True},{'cache_key':'a'*64}
    client=Client()
    value,metadata=complete_label_json(client,'rules','complete original evidence')
    assert client.max_completion_tokens==32768
    assert client.calls==[('rules','complete original evidence',32768),
                          ('rules','complete original evidence',16384)]
    assert value=={'unchanged_response':True}
    assert metadata['output_budget_adjustment']['evidence_truncated'] is False


def test_context_retry_does_not_retry_unrelated_provider_errors():
    import pytest
    from trace_analysis.pipeline.stage_06_trace_quality_labeling.relabel import complete_label_json
    class ProviderError(Exception):
        status_code=429
    class Client:
        max_completion_tokens=32768
        def complete_json(self,*args):
            raise ProviderError('rate limit')
    with pytest.raises(ProviderError,match='rate limit'):
        complete_label_json(Client(),'rules','evidence')
