import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from search_r1.dynamic_sampling.stratified import StratifiedQuestionSampler, load_manifest, file_sha256
from search_r1.dynamic_sampling.behavior import compact_trace, summarize_scopes, BehaviorMonitor, flatten
from verl.trainer.ppo.dynamic_sampling_trainer import DynamicSamplingTrainer
from evaluation.build_behavior_strata import build


def sampler(seed=42):
    ids = list(map(str, range(120)))
    roles = {sid: ('first_sufficient' if int(sid)<40 else 'first_insufficient' if int(sid)<80 else 'query_rewrite') for sid in ids}
    return StratifiedQuestionSampler(ids, roles, 8, seed=seed), roles


def test_prompt_quotas_reproducible_unique_and_refill_excludes_previous_questions():
    a, roles = sampler(); b, _ = sampler()
    for _ in range(30):
        excluded = set()
        for _ in range(3):
            indices, report = a.sample(excluded)
            other, other_report = b.sample(excluded)
            assert (indices, report) == (other, other_report)
            assert len(indices)==len(set(indices))==8
            assert not excluded.intersection(map(str, indices))
            assert sum(report['requested'].values())==8
            assert report['requested']['uniform']==2
            assert report['requested']['first_insufficient']==3
            assert report['realized']['first_insufficient']>=3
            assert not report['fallback']
            excluded.update(map(str, indices))


def test_small_role_shortfall_falls_back_without_duplicates():
    s=StratifiedQuestionSampler([str(i) for i in range(12)], {str(i):'first_sufficient' for i in range(12)}, 8)
    indices, report=s.sample()
    assert len(set(indices))==8
    assert sum(report['fallback'].values())>0
    with pytest.raises(ValueError, match='distinct'):s.sample(excluded_ids=range(6))


@pytest.mark.parametrize('fraction', [-.1, 1.1, float('nan')])
def test_invalid_sampling_config(fraction):
    with pytest.raises(ValueError):StratifiedQuestionSampler(['a'], {'a':'unobserved'}, 1, uniform_fraction=fraction)


def test_manifest_rejects_changed_data_and_unknown_sample_ids(tmp_path):
    data=tmp_path/'train';data.write_bytes(b'original')
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1,dataset_sha256=file_sha256(data),roles={'a':'query_rewrite'})))
    assert load_manifest(path,data,['a'])['roles']['a']=='query_rewrite'
    with pytest.raises(ValueError, match='missing'):load_manifest(path,data,['b'])
    data.write_bytes(b'changed')
    with pytest.raises(ValueError, match='different'):load_manifest(path,data,['a'])


def make_trace(calls=1, coverage=.5, answer='x', blocked=False, reward=0.):
    events=[dict(action='search',attempted=True,turn=i) for i in range(calls)]
    events.append(dict(action='search' if blocked else 'answer', turn=calls,
                       error='search_budget_exhausted' if blocked else ''))
    return dict(events=events,retrieval_attempts=calls,evidence_eligible=True,evidence_coverage=coverage,
                total_reward=reward,final_answer=answer,evidence_events=[dict(turn=1,evidence_gain=.5)] if calls>1 else [])


def test_candidate_and_selected_metrics_keep_distinct_group_denominators():
    rows=[compact_trace(make_trace(calls=2,reward=1.),'a','first_insufficient'),
          compact_trace(make_trace(blocked=True,answer=None),'a','first_insufficient'),
          compact_trace(make_trace(coverage=1.),'b','first_sufficient'),
          compact_trace(make_trace(coverage=1.),'b','first_sufficient')]
    report=summarize_scopes(rows,rows[2:])
    assert report['candidate']['overall']['second_search_rate']==.25
    assert report['candidate']['overall']['zero_reward_group_rate']==.5
    assert report['candidate']['by_stratum']['first_insufficient']['missing_answer_rate']==.5
    assert report['selected']['overall']['second_search_rate']==0
    assert report['selected']['overall']['zero_reward_group_rate']==1
    assert flatten(report)['behavior/candidate/trajectories']==4


def test_monitor_alerts_only_after_complete_window_and_logs_default_nonblocking(tmp_path):
    monitor=BehaviorMonitor(tmp_path,window_size=2,warmup_steps=0,min_target_trajectories=2)
    rows=[compact_trace(make_trace(),'a','first_insufficient')]*2
    report=summarize_scopes(rows,rows)
    _, alerts=monitor.record(1,report);assert not alerts
    with pytest.warns(RuntimeWarning, match='SECOND_SEARCH'):
        _, alerts=monitor.record(2,report)
    assert 'SECOND_SEARCH_UNDERUSE_WITH_INCOMPLETE_EVIDENCE' in alerts
    assert not monitor.stop_on_alert
    state=json.loads((tmp_path/'behavior_health.json').read_text())
    assert state['global_step']==2
    assert len((tmp_path/'behavior_metrics.jsonl').read_text().splitlines())==2


def test_monitor_does_not_demand_second_search_when_current_evidence_is_complete(tmp_path):
    monitor=BehaviorMonitor(tmp_path,window_size=1,warmup_steps=0,min_target_trajectories=1)
    rows=[compact_trace(make_trace(coverage=1.),'a','first_insufficient')]*2
    _, alerts=monitor.record(1,summarize_scopes(rows,rows))
    assert not alerts


def test_manifest_builder_uses_only_train_ids_and_records_train_hash(tmp_path):
    dataset=tmp_path/'train.parquet'
    pd.DataFrame({'extra_info':[{'index':i,'split':'train','evidence':{'raw_id':f'r{i}'}} for i in range(4)]}).to_parquet(dataset)
    traces=[]
    for i,coverage in enumerate((1., .5, 0.)):
        r=make_trace(coverage=coverage);r.update(split='train',sample_id=str(i),evidence_reference_id=f'r{i}')
        r['evidence_events']=[dict(turn=0,cumulative_coverage=coverage)]
        traces.append(r)
    path=tmp_path/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in traces))
    result=build(dataset,path,tmp_path/'strata.json')
    assert result['roles']=={'0':'first_sufficient','1':'first_insufficient','2':'query_rewrite','3':'unobserved'}
    traces[0]['split']='validation';path.write_text(json.dumps(traces[0])+'\n')
    with pytest.raises(ValueError, match='training split'):build(dataset,path,tmp_path/'bad.json')


def test_sampler_is_wired_before_rollout_without_changing_prompt_tensors(tmp_path):
    roles={str(i):('first_sufficient' if i<8 else 'first_insufficient' if i<16 else 'query_rewrite') for i in range(24)}
    extras=[dict(index=i,split='train',evidence={'raw_id':str(i)}) for i in range(24)]
    dataset_file=tmp_path/'train.parquet';pd.DataFrame(dict(extra_info=extras)).to_parquet(dataset_file)
    manifest=tmp_path/'behavior_strata.json'
    manifest.write_text(json.dumps(dict(schema_version=1,dataset_sha256=file_sha256(dataset_file),roles=roles)))
    class Dataset:
        dataframe=pd.DataFrame(dict(extra_info=extras))
        def __getitem__(self,index):
            return dict(input_ids=torch.tensor([index,2]),attention_mask=torch.ones(2,dtype=torch.long),
                        position_ids=torch.arange(2),index=index,extra_info=extras[index])
    config=OmegaConf.create(dict(dynamic_sampling=dict(audit_dir=str(tmp_path/'audit')),
        data=dict(train_files=str(dataset_file),train_batch_size=8),trainer=dict(seed=42),
        behavior_sampling=dict(enabled=True,manifest_path=str(manifest),uniform_fraction=.25,
            role_weights=dict(first_sufficient=1,first_insufficient=2,query_rewrite=1)),behavior_monitor=dict(enabled=True)))
    def fake_init(self,*args,**kwargs):self.config=config;self.train_dataset=Dataset()
    with patch('verl.trainer.ppo.dynamic_sampling_trainer.RayPPOTrainer.__init__',new=fake_init):
        trainer=DynamicSamplingTrainer(candidate_reward_fn=None)
    batch=trainer._next_questions()
    assert len(batch['index'])==8
    assert Counter(batch['behavior_stratum'])['first_insufficient']>=3
    for i,index in enumerate(batch['index']):
        torch.testing.assert_close(batch['input_ids'][i],torch.tensor([index,2]))
        assert 'behavior_stratum' not in batch['extra_info'][i]
    assert (tmp_path/'audit/behavior_strata_snapshot.json').is_file()


def test_candidate_uses_scored_records_not_unscored_generation_traces(tmp_path):
    from test_batching_and_trainer import batch as make_batch
    from unittest.mock import MagicMock
    trainer=DynamicSamplingTrainer.__new__(DynamicSamplingTrainer)
    trainer.config=OmegaConf.create(dict(actor_rollout_ref=dict(rollout=dict(n_agent=2)),data=dict(max_start_length=2)))
    trainer._behavior_monitor=object()
    output=make_batch('x')
    output.non_tensor_batch.pop('uid')
    output.non_tensor_batch['behavior_stratum']=np.array(['first_insufficient']*4,dtype=object)
    # Generation-only records deliberately lack fields computed by RewardManager.
    assert 'evidence_coverage' not in output.non_tensor_batch['trajectory'][0]
    def scored(data):
        records=[];scores=torch.zeros_like(data.batch['responses'],dtype=torch.float32)
        for i,sid in enumerate(data.non_tensor_batch['index']):
            record=make_trace(coverage=.5,reward=float(i%2));record['sample_id']=str(sid)
            records.append(record);scores[i,-1]=float(i%2)
        reward.last_records=records
        return scores
    reward=MagicMock(side_effect=scored);trainer.candidate_reward_fn=reward
    manager=MagicMock();manager.run_llm_loop.return_value=output
    questions=dict(input_ids=torch.ones(2,2,dtype=torch.long),attention_mask=torch.ones(2,2,dtype=torch.long),
                   position_ids=torch.zeros(2,2,dtype=torch.long),index=np.array(['xa','xb'],dtype=object),
                   behavior_stratum=np.array(['first_insufficient']*2,dtype=object))
    with patch.object(trainer,'_next_questions',return_value=questions):
        result,_=trainer._candidate(manager,1,0)
    records=result.non_tensor_batch['behavior_record'].tolist()
    assert sum(r['eligible'] for r in records)==4
    assert [r['reward'] for r in records]==[0.,1.,0.,1.]
    assert all(r['coverage']==.5 for r in records)
