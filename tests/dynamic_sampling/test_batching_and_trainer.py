import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from search_r1.dynamic_sampling.batching import merge_groups
from search_r1.dynamic_sampling.selection import CandidateGroup
from verl.trainer.ppo.dynamic_sampling_trainer import DynamicSamplingTrainer


def batch(prefix, p=2, r=3, rewards=(0.,1.,0.,0.)):
    n=len(rewards)
    prompts=torch.arange(1,n*p+1).reshape(n,p)
    responses=torch.full((n,r),3,dtype=torch.long)
    mask=torch.ones(n,p+r,dtype=torch.long)
    full=torch.cat([prompts,responses],dim=-1)
    ids=[prefix+'a']*2+[prefix+'b']*2
    traces=[dict(final_answer='answer',retrieval_attempts=1,retrieval_failures=0,
        events=[dict(attempted=True,observation_tokens_after_truncation=2)]) for _ in range(n)]
    return DataProto.from_dict(dict(prompts=prompts,responses=responses,input_ids=full,
        responses_with_info_mask=responses.clone(),attention_mask=mask,info_mask=mask.clone(),
        position_ids=mask.cumsum(-1)-1,sampling_reward=torch.tensor(rewards)),
        non_tensors=dict(index=ids,uid=[prefix+'u1']*2+[prefix+'u2']*2,
            trajectory=traces,data_source=['hotpotqa']*n),
        meta_info=dict(temperature=1.,turns_stats=[1,2,3,4],active_mask=[False]*n,
            valid_action_stats=[1]*n,valid_search_stats=[1]*n))


class BatchingTests(unittest.TestCase):
    def test_variable_lengths_preserve_tokens_positions_masks_and_group_ids(self):
        a,b=batch('x',p=2,r=3),batch('y',p=4,r=5)
        groups=[CandidateGroup(0,'xa',(0,1),(0,1)),CandidateGroup(1,'yb',(2,3),(0,0))]
        merged=merge_groups([a,b],groups,pad_token_id=99)
        self.assertEqual(merged.batch['input_ids'].shape,(4,9))
        self.assertEqual(merged.non_tensor_batch['index'].tolist(),['xa','xa','yb','yb'])
        self.assertEqual(merged.meta_info['turns_stats'],[1,2,3,4])
        self.assertTrue((merged.batch['prompts'][:2,:2]==99).all())
        self.assertTrue((merged.batch['attention_mask'][:2,:2]==0).all())
        self.assertTrue((merged.batch['attention_mask'][:2,-2:]==0).all())
        for row,original,idx in [(0,a,0),(1,a,1),(2,b,2),(3,b,3)]:
            oldmask=original.batch['attention_mask'][idx].bool()
            newmask=merged.batch['attention_mask'][row].bool()
            for key in ['input_ids','position_ids']:
                torch.testing.assert_close(merged.batch[key][row][newmask],original.batch[key][idx][oldmask])

    def trainer(self,path,enabled=True):
        t=DynamicSamplingTrainer.__new__(DynamicSamplingTrainer)
        t.config=OmegaConf.create(dict(data=dict(train_batch_size=2),trainer=dict(seed=42),
            actor_rollout_ref=dict(rollout=dict(n_agent=2))))
        t.sampling=OmegaConf.create(dict(enabled=enabled,max_candidate_batches=3,reward_range_epsilon=1e-8))
        t.audit_dir=Path(path)
        t.tokenizer=SimpleNamespace(pad_token_id=99)
        t._candidate_epoch=0
        return t

    def test_trainer_refill_counts_discarded_generation_and_aligns_groups(self):
        with tempfile.TemporaryDirectory() as path:
            t=self.trainer(path)
            candidate=[batch('x'),batch('y',p=4,r=5)]
            with patch.object(t,'_candidate',side_effect=[(candidate[0],.1),(candidate[1],.2)]) as call:
                merged,stats=t._collect(None,1)
            self.assertEqual(call.call_count,2)
            self.assertEqual(stats['sampling/candidate_trajectories'],8)
            self.assertEqual(stats['sampling/selected_trajectories'],4)
            self.assertEqual(stats['sampling/candidate_generated_tokens'],32)
            self.assertEqual(stats['sampling/candidate_retrieval_calls'],8)
            self.assertEqual(merged.non_tensor_batch['index'].tolist(),['xa','xa','ya','ya'])
            self.assertTrue((Path(path)/'sampling_decisions.jsonl').exists())

    def test_trainer_uniform_mode_keeps_zero_groups(self):
        with tempfile.TemporaryDirectory() as path:
            t=self.trainer(path,enabled=False)
            with patch.object(t,'_candidate',return_value=(batch('x'),.1)) as call:
                merged,stats=t._collect(None,1)
            self.assertEqual(call.call_count,1)
            self.assertEqual(stats['sampling/selected_zero_groups'],1)
            self.assertEqual(merged.non_tensor_batch['index'].tolist(),['xa','xa','xb','xb'])

    def test_trainer_cap_falls_back_without_extra_generation(self):
        with tempfile.TemporaryDirectory() as path:
            t=self.trainer(path)
            batches=[batch(str(i),rewards=(0.,0.,1.,1.)) for i in range(3)]
            with patch.object(t,'_candidate',side_effect=[(x,.1) for x in batches]) as call:
                merged,stats=t._collect(None,1)
            self.assertEqual(call.call_count,3)
            self.assertEqual(stats['sampling/zero_fallback_groups'],2)
            self.assertEqual(len(merged),4)


if __name__ == '__main__': unittest.main()
