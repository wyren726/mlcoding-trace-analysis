import json
import pytest

from trace_analysis.pipeline.stage_06_trace_quality_labeling.source_index import TurnSourceIndex
from trace_analysis.pipeline.stage_06_trace_quality_labeling.runner import _prepare_episode_inputs
from trace_analysis.pipeline.stage_06_trace_quality_labeling.evidence_retrieval import stored_selection, retrieve_positions


def test_index_matches_existing_episode_view_and_physical_locations(tmp_path):
    source = tmp_path/'turns.jsonl'
    def turn(i, role='user_message'):
        return dict(turn_id=f't{i}',trace_id='trace',batch_id='batch',turn_index=i,
                    events=[dict(event_id=f'e{i}',type=role,data=dict(content=f'实际内容 {i}'))])
    turns = [turn(1),turn(2),turn(3,'assistant_message'),turn(4),turn(5)]
    source.write_text('\n'+'\n\n'.join(json.dumps(t,ensure_ascii=False) for t in turns)+'\n')
    row=dict(episode_id='episode',trace_id='trace',batch_id='batch',query_turn_id='t1',
             episode_boundary=dict(start_turn_id='t1',end_turn_id='t2'),
             lineage=dict(preprocessed_input=str(source)))
    index=TurnSourceIndex(source,tmp_path/'cache')
    episode,locations=index.load_episode(row)
    assert episode==_prepare_episode_inputs([(1,row)])[1]
    assert [t['turn_id'] for t in episode['episode_turns']]==['t1','t2','t4']
    assert locations[(str(source),'t4')]['physical_line']==8
    assert locations[(str(source),'t4')]['record_index']==3
    assert index.metadata==TurnSourceIndex(source,tmp_path/'cache').metadata
    for changes in [dict(episode_boundary=dict(start_turn_id='t2',end_turn_id='missing')),
                    dict(query_turn_id='t5'),dict(batch_id='wrong')]:
        with pytest.raises(ValueError):
            index.load_episode({**row,**changes})


def test_source_change_invalidates_old_index(tmp_path):
    source=tmp_path/'turns.jsonl'
    turn=dict(turn_id='t1',trace_id='trace',batch_id='batch',events=[])
    source.write_text(json.dumps(turn)+'\n')
    index=TurnSourceIndex(source,tmp_path/'cache')
    source.write_text(json.dumps({**turn,'turn_id':'t2_different_length'})+'\n')
    row=dict(trace_id='trace',batch_id='batch',episode_boundary=dict(start_turn_id='t1',end_turn_id='t1'))
    with pytest.raises(ValueError,match='Source changed'):
        index.load_episode(row)
    rebuilt=TurnSourceIndex(source,tmp_path/'cache')
    assert index.metadata['source_sha256']!=rebuilt.metadata['source_sha256']


def test_partitioned_provenance_uses_only_current_episode():
    episode=dict(episode_turns=[dict(events=[dict(event_id='tool',type='tool_result',data=dict(content='正文'))])])
    row=dict(episode_id='ep',analysis_provenance=dict(episode_verifications=[
        dict(episode_id='other',selected_tool_event_ids=['wrong']),
        dict(episode_id='ep',selected_tool_event_ids=['tool','outside'])]))
    ids,provenance=stored_selection(row,{},episode)
    assert ids==['tool']
    assert provenance['discarded_stored_ids']==['outside']


def test_first_run_retrieves_without_old_labels():
    episode=dict(trace_id='trace',episode_turns=[dict(turn_id='t',events=[
        dict(event_id='u',type='user_message',data=dict(content='复核实验数据和输出指标')),
        dict(event_id='tool',type='tool_result',data=dict(content='指标为0.9'))])])
    class Client:
        def complete_json(self,system,user):
            assert '五维难度' in system
            return dict(trace_id='trace',selected_event_ids=['tool'],reason='核对实际指标'),dict(cache_key='a'*64)
    ids,provenance=retrieve_positions(dict(episode_id='ep'),{},episode,Client())
    assert ids==['tool']
    assert provenance['source']=='stage06_index_retrieval'
    assert len(provenance['model_calls'])==1


def test_retrieval_corrects_invalid_ids_without_using_unshown_evidence():
    episode=dict(trace_id='trace',episode_turns=[dict(turn_id='t',events=[
        dict(event_id='tool',type='tool_result',data=dict(content='准确原文'))])])
    class Client:
        calls=0
        def complete_json(self,system,user):
            self.calls+=1
            if self.calls==1:
                return dict(selected_event_ids=['made_up']),dict(cache_key='a'*64)
            assert json.loads(user)['repair']['allowed_tool_ids']==['tool']
            return dict(selected_event_ids=['tool'],reason='核对真实工具输出'),dict(cache_key='b'*64)
    client=Client()
    ids,provenance=retrieve_positions(dict(episode_id='ep'),{},episode,client)
    assert ids==['tool'] and client.calls==2
    assert provenance['model_calls'][0]['validation_error']


def test_retrieval_skips_message_only_partitions():
    episode=dict(trace_id='trace',episode_turns=[dict(turn_id='t',events=[
        dict(event_id='u',type='user_message',data=dict(content='用户要求'))])])
    class Client:
        def complete_json(self,*args):
            raise AssertionError('Messages are already supplied in full')
    ids,provenance=retrieve_positions(dict(episode_id='ep'),{},episode,Client())
    assert ids==[] and provenance['model_calls']==[]
