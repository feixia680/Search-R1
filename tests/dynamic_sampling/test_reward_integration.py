import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from search_r1.dynamic_sampling.batching import merge_groups
from search_r1.dynamic_sampling.selection import CandidateGroup
from verl.trainer.main_ppo import RewardManager


@unittest.skipUnless(os.environ.get('ALGORITHM_A_MODEL'), 'Set ALGORITHM_A_MODEL for tokenizer integration')
class RealVisibilityRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'algorithm_a/test_evidence.py'
        spec=importlib.util.spec_from_file_location('evidence_fixture_helpers',path)
        cls.helpers=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helpers)
        cls.helpers.EvidenceTests.setUpClass()

    def candidate(self,prefix,long=False):
        h=self.helpers
        fixture=h.EvidenceTests(methodName='runTest')
        prompts=['Use <answer>example</answer>. Question: Where?'+(' extra context '*12 if long else '')]*4
        docs=[h.doc(),h.doc('France','France is in Europe.'+(' other words '*60 if long else ''))]
        output=fixture.rollout([['<search>Paris France</search>']*4,
            ['<answer>France</answer>']*2+['<answer>Wrong</answer>']*2],
            [[docs]*4],count=4,prompts=prompts)
        output.non_tensor_batch.update(dict(index=np.array([prefix+'a']*2+[prefix+'b']*2,dtype=object),
            uid=np.array([prefix+'u1']*2+[prefix+'u2']*2,dtype=object),
            data_source=np.array(['hotpotqa']*4,dtype=object),
            reward_model=np.array([{'ground_truth':{'target':['France']}}]*4,dtype=object),
            extra_info=np.array([{'evidence':h.reference()}]*4,dtype=object)))
        score=RewardManager(fixture.tokenizer,0,alpha=.1)(output)
        output.batch['sampling_reward']=score.sum(-1)
        return output

    def test_token_provenance_and_reward_survive_cross_round_padding(self):
        a,b=self.candidate('x'),self.candidate('y',long=True)
        groups=[CandidateGroup(0,'xa',(0,1),tuple(a.batch['sampling_reward'][:2].tolist())),
                CandidateGroup(1,'yb',(2,3),tuple(b.batch['sampling_reward'][2:].tolist()))]
        tok=self.helpers.EvidenceTests.tokenizer
        merged=merge_groups([a,b],groups,tok.pad_token_id)
        score=RewardManager(tok,0,alpha=.1)(merged).sum(-1)
        torch.testing.assert_close(score,merged.batch['sampling_reward'])
        self.assertEqual(merged.non_tensor_batch['index'].tolist(),['xa','xa','yb','yb'])
        self.assertTrue(all(t['visible_views'] for x in merged.non_tensor_batch['trajectory']
                            for t in x['events'] if t.get('attempted')))


if __name__=='__main__':unittest.main()
