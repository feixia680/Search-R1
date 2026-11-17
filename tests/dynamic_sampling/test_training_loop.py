import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.dynamic_sampling_trainer import DynamicSamplingTrainer
from test_batching_and_trainer import batch


class FakeWorker:
    def __init__(self):self.updates=0
    def compute_log_prob(self,data):
        return DataProto.from_dict({'old_log_probs':torch.zeros_like(data.batch['responses'],dtype=torch.float32)})
    def compute_ref_log_prob(self,data):
        return DataProto.from_dict({'ref_log_prob':torch.zeros_like(data.batch['responses'],dtype=torch.float32)})
    def update_actor(self,data):
        self.updates+=1
        assert torch.isfinite(data.batch['advantages']).all()
        assert data.batch['advantages'].abs().sum()>0
        return DataProto.from_dict({'marker':torch.ones(len(data),1)},
            meta_info={'metrics':{'actor/pg_loss':[.1],'actor/grad_norm':[1.]}})


class FakeReward:
    last_metrics={'algorithm_a/em':.25}
    def __call__(self,data):
        result=torch.zeros_like(data.batch['responses'],dtype=torch.float32)
        for i in range(len(data)):
            p=data.batch['prompts'].shape[1]
            last=int(data.batch['attention_mask'][i,p:].sum())-1
            result[i,last]=data.batch['sampling_reward'][i]
        return result


class TrainingLoopTests(unittest.TestCase):
    def test_two_updates_advantages_validation_checkpoint_and_audit(self):
        with tempfile.TemporaryDirectory() as path:
            t=DynamicSamplingTrainer.__new__(DynamicSamplingTrainer)
            t.config=OmegaConf.create(dict(trainer=dict(val_only=False,val_before_train=False,
                test_freq=-1,save_freq=2,n_gpus_per_node=1,nnodes=1),
                data=dict(max_start_length=32,max_prompt_length=64,max_response_length=16,max_obs_length=32),
                algorithm=dict(no_think_rl=False,gamma=1.,lam=1.),
                actor_rollout_ref=dict(rollout=dict(n=1)),max_turns=2,
                retriever=dict(url='http://unused',topk=3)))
            t.total_training_steps=3
            t.audit_dir=Path(path)
            t.tokenizer=SimpleNamespace(pad_token_id=99)
            t.actor_rollout_wg=FakeWorker();t.ref_policy_wg=FakeWorker()
            t.use_reference_policy=True;t.val_reward_fn=object();t.reward_fn=FakeReward()
            seen=[];t.logger=SimpleNamespace(log=lambda data,step:seen.append(step))
            with patch.object(t,'_collect',side_effect=[(batch('x'),{}),(batch('y'),{})]), \
                 patch.object(t,'_balance_batch',return_value=None), \
                 patch.object(t,'_validate',return_value={'val/test_score/hotpotqa':.25}) as validation, \
                 patch.object(t,'_save_checkpoint') as save, \
                 patch('verl.trainer.ppo.dynamic_sampling_trainer.LLMGenerationManager'):
                t.fit()
            self.assertEqual(t.actor_rollout_wg.updates,2)
            self.assertEqual(seen,[1,2,3])
            self.assertEqual(validation.call_count,1)
            self.assertEqual(save.call_count,1)
            updates=[json.loads(x) for x in (Path(path)/'completed_updates.jsonl').read_text().splitlines()]
            self.assertEqual([x['global_step'] for x in updates],[1,2])


if __name__=='__main__':unittest.main()
