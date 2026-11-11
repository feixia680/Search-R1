import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests
import torch
from transformers import AutoTokenizer

from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.trajectory import new_trajectory, record_action, reward_components
from verl import DataProto
from verl.trainer.main_ppo import RewardManager
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss
from verl.utils.reward_score import qa_em


class AccountingTests(unittest.TestCase):
    def manager(self, count):
        manager = LLMGenerationManager.__new__(LLMGenerationManager)
        manager.config = SimpleNamespace(topk=3, search_url='http://localhost/retrieve', search_timeout=.01)
        manager._trajectories = [new_trajectory() for _ in range(count)]
        return manager

    def test_failure_duplicate_and_denied_search(self):
        manager = self.manager(2)
        with patch.object(manager, '_batch_search', side_effect=requests.Timeout('test timeout')) as call:
            manager.execute_predictions(['<search>  Paris WEATHER </search>', '<search>inactive</search>'], '', [True, False])
            self.assertEqual(call.call_args.args[0], ['Paris WEATHER'])
            manager.execute_predictions(['<search>paris weather</search>', ''], '', [True, False])
            manager.execute_predictions(['<search>third query</search>', ''], '', [True, False], do_search=False)
            self.assertEqual(call.call_count, 2)
        trace = manager._trajectories[0]
        self.assertEqual((trace['search_actions'], trace['retrieval_attempts'],
                          trace['retrieval_successes'], trace['retrieval_failures'], trace['duplicate_queries']),
                         (3, 2, 0, 2, 1))
        self.assertEqual(manager._trajectories[1]['events'], [])
        self.assertAlmostEqual(reward_components(0, trace, .02, .05)['total_reward'], -.09)

    def test_success_empty_and_malformed_documents(self):
        manager = self.manager(3)
        result = {'result': [[{'document': {'id': '42', 'contents': 'Paris\nCapital of France'}}], [], [{}]]}
        with patch.object(manager, '_batch_search', return_value=result):
            manager.execute_predictions(['<search>Paris</search>'] * 3, '', [True] * 3)
        self.assertEqual([t['retrieval_successes'] for t in manager._trajectories], [1, 1, 0])
        self.assertEqual(manager._trajectories[0]['events'][0]['document_ids'], ['42'])
        self.assertEqual(manager._trajectories[2]['retrieval_failures'], 1)

    def test_http_status_and_timeout(self):
        manager = self.manager(1)
        with patch('search_r1.llm_agent.generation.requests.post') as post:
            post.return_value.raise_for_status.side_effect = requests.HTTPError('503')
            manager.execute_predictions(['<search>Paris</search>'], '', [True])
            self.assertEqual(post.call_args.kwargs['timeout'], .01)
        self.assertEqual(manager._trajectories[0]['retrieval_failures'], 1)

    def test_zero_warmup_updates_the_first_step(self):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
        self.assertEqual(optimizer.param_groups[0]['lr'], .1)
        parameter.backward()
        optimizer.step()
        scheduler.step()
        self.assertLess(parameter.item(), 1.)
        optimizer = torch.optim.SGD([parameter], lr=.1)
        get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.)

    def test_missing_trace_and_bad_coefficients_fail_loudly(self):
        self.assertEqual(reward_components(1, None)['total_reward'], 1)
        with self.assertRaises(ValueError):
            reward_components(1, None, .02)
        for coefficient in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                reward_components(1, {}, coefficient)


class RewardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.tokenizer = AutoTokenizer.from_pretrained(os.environ['ALGORITHM_A_MODEL'], local_files_only=True)

    def batch(self, answers, traces, truths=None):
        tok = self.tokenizer
        prompt = tok(['Use <answer>example</answer>. Question: capital of France?'] * len(answers),
                     return_tensors='pt', padding=True, add_special_tokens=False)
        response = tok(answers, return_tensors='pt', padding=True, add_special_tokens=False)
        attention = torch.cat([prompt.attention_mask, response.attention_mask], dim=1)
        non = dict(data_source=['hotpotqa'] * len(answers), index=list(range(len(answers))),
                   reward_model=[{'ground_truth': {'target': [truth]}, 'style': 'rule'}
                                 for truth in (truths or ['Paris'] * len(answers))])
        if traces is not None:
            non['trajectory'] = traces
        return DataProto.from_dict(dict(prompts=prompt.input_ids, responses=response.input_ids,
                                         attention_mask=attention), non_tensors=non)

    def test_original_em_compatibility_components_and_jsonl(self):
        traces = [new_trajectory() for _ in range(4)]
        success = dict(success=True, error=None, document_ids=['1'], seconds=.1)
        record_action(traces[0], 'answer', 'Paris', '<answer>Paris</answer>')
        record_action(traces[1], 'search', 'Paris', '<search>Paris</search>', success)
        record_action(traces[1], 'answer', 'London', '<answer>London</answer>')
        for query in ['Paris', ' PARIS ']:
            record_action(traces[2], 'search', query, f'<search>{query}</search>', success)
        record_action(traces[2], 'answer', 'Paris', '<answer>Paris</answer>')
        record_action(traces[3], 'search', 'Paris', '<search>Paris</search>', dict(success=False, error='timeout'))
        answers = ['<answer>Paris</answer>', '<search>Paris</search><answer>London</answer>',
                   '<search>Paris</search><answer>Paris</answer>', '<search>Paris</search>']
        batch = self.batch(answers, traces)
        baseline = RewardManager(self.tokenizer, 0)(batch)
        # Explicit comparison to the unchanged upstream scorer.
        for i in range(4):
            seq = self.tokenizer.decode(torch.cat([batch.batch['prompts'][i], batch.batch['responses'][i]]))
            self.assertEqual(float(baseline[i].sum()), qa_em.compute_score_em(seq, {'target': ['Paris']}))
        with tempfile.TemporaryDirectory() as directory:
            manager = RewardManager(self.tokenizer, 0, beta=.02, gamma=.05, trace_dir=directory)
            shaped = manager(batch)
            torch.testing.assert_close(shaped.sum(-1), torch.tensor([1., -.02, .91, -.02]))
            rows = [json.loads(s) for s in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(rows[2]['duplicate_queries'], 1)
            self.assertEqual(rows[2]['final_answer'], 'Paris')
            self.assertAlmostEqual(manager.last_metrics['algorithm_a/em'], .5)
        no_trace = self.batch(answers, None)
        self.assertTrue(torch.equal(baseline, RewardManager(self.tokenizer, 0)(no_trace)))
        # Trace objects must follow rows when the trainer rebalances a batch.
        batch.reorder(torch.tensor([2, 0, 3, 1]))
        self.assertEqual(batch.non_tensor_batch['trajectory'][0]['duplicate_queries'], 1)
        torch.testing.assert_close(RewardManager(self.tokenizer, 0, beta=.02, gamma=.05)(batch).sum(-1),
                                   torch.tensor([.91, 1., -.02, -.02]))

    def test_reward_reaches_grpo_and_gradient_update(self):
        short, long = new_trajectory(), new_trajectory()
        record_action(long, 'search', 'Paris', '<search>Paris</search>', dict(success=True))
        batch = self.batch(['<answer>Paris</answer>'] * 2, [short, long])
        rewards = RewardManager(self.tokenizer, 0, beta=.02)(batch)
        mask = batch.batch['attention_mask'][:, -rewards.shape[1]:]
        advantages, _ = compute_grpo_outcome_advantage(rewards, mask, np.array(['same', 'same']))
        self.assertGreater(float(advantages[0].sum()), 0)
        self.assertLess(float(advantages[1].sum()), 0)
        parameter = torch.nn.Parameter(torch.zeros(2, rewards.shape[1]))
        optimizer = torch.optim.SGD([parameter], lr=.01)
        loss, _, _ = compute_policy_loss(torch.zeros_like(parameter), parameter, advantages, mask, .2)
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertGreater(float(parameter.abs().sum()), 0)

    def test_real_sample_labels_scoring_only(self):
        # Ground-truth answers are used to construct fixtures, NOT model evaluation.
        import os
        import pandas as pd
        rows = pd.read_parquet(os.environ['ALGORITHM_A_DATA']).head(3).to_dict('records')
        truths = [r['reward_model']['ground_truth']['target'][0] for r in rows]
        traces = [new_trajectory() for _ in rows]
        for trace, answer in zip(traces, truths):
            record_action(trace, 'answer', answer, f'<answer>{answer}</answer>')
        batch = self.batch([f'<answer>{a}</answer>' for a in truths], traces, truths)
        torch.testing.assert_close(RewardManager(self.tokenizer, 0)(batch).sum(-1), torch.ones(len(rows)))


if __name__ == '__main__':
    unittest.main()
