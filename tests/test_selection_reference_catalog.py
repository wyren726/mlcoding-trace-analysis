from trace_analysis.pipeline.stage_06_trace_quality_labeling.selection_review import reference_catalog


def test_catalog_distinguishes_requirement_id_and_supported_claim():
    model = dict(case_requirements=[dict(case_requirement_id='req_write',text='写论文')],
                 assessment_target=dict(target_id='episode'),
                 evidence_refs=[dict(evidence_id='ev1',event_id='evt1',source_role='user',quote='写论文')],
                 claims=[dict(claim_id='map_write',target_id='req_write',field_path='capability_mapping[0]',
                              support_status='supported',reason='用户要求',evidence_ids=['ev1'],conflicting_evidence_ids=[])])
    baseline=dict(case_id='case',assessment_id='episode',mode='episode')
    row=reference_catalog(model,baseline)[0]
    assert row['claim_id']=='map_write' and row['target_id']=='req_write'
    assert row['usable_for_pass_fail'] and row['evidence'][0]['quote']=='写论文'
    review=dict(case_id='case',status='accepted',claim_reviews=[dict(claim_id='map_write',support_status='partial',evidence_ids=['ev1'])])
    assert not reference_catalog(model,baseline,[review])[0]['usable_for_pass_fail']
    model['claims'][0]['conflicting_evidence_ids']=['ev1']
    assert not reference_catalog(model,baseline)[0]['usable_for_pass_fail']
    baseline.update(mode='case_requirement',assessment_id='other')
    assert reference_catalog(model,baseline)==[]
