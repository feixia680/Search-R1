import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.trainer.main_ppo import RewardManager
from search_r1.llm_agent.evidence import reference_from_raw, score_evidence, contains_sentence
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.llm_agent.trajectory import new_trajectory, record_action, reward_components


def reference():
    return dict(schema_version=1, status='valid', raw_id='fixture', supporting_facts=[
        dict(fact_id='Paris:0', title='Paris', sentence_id=0, sentence='Paris is the capital of France.'),
        dict(fact_id='France:0', title='France', sentence_id=0, sentence='France is in Europe.')])


def doc(title='Paris', body='Paris is the capital of France.'):
    return dict(document=dict(id=title, contents='"' + title + '"\n' + body))


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(os.environ['ALGORITHM_A_MODEL'], local_files_only=True)

    def rollout(self, actions, batches, obs_limit=512, context_limit=2048, count=1, prompts=None, record_evidence=True):
        tok = self.tokenizer
        testcase = self
        class Worker:
            calls = 0
            inputs = None
            def generate_sequences(worker, batch):
                testcase.assertFalse(batch.non_tensor_batch)
                if worker.inputs is None:
                    worker.inputs = []
                worker.inputs.append(batch.batch['input_ids'].clone())
                texts = actions[worker.calls]
                testcase.assertEqual(len(texts), len(batch))
                worker.calls += 1
                ids = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).input_ids
                return DataProto.from_dict({'responses': ids})
        worker = Worker()
        manager = LLMGenerationManager(tok, worker, GenerationConfig(
            max_turns=len(actions) - 1, max_start_length=512, max_prompt_length=context_limit,
            max_response_length=256, max_obs_length=obs_limit, num_gpus=1, record_evidence=record_evidence))
        encoded = tok(prompts or ['Question: Where is Paris?'] * count, return_tensors='pt', padding=True, add_special_tokens=False)
        b = DataProto.from_dict(dict(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
            position_ids=torch.arange(encoded.input_ids.shape[1]).unsqueeze(0).repeat(count, 1)))
        with patch.object(manager, '_batch_search', side_effect=[{'result': x} for x in batches]):
            output = manager.run_llm_loop(b, encoded.input_ids)
        self.last_worker_inputs = worker.inputs
        return output

    def test_tracing_does_not_change_policy_inputs_or_rollout_tokens(self):
        actions = [['<search>Paris</search>'], ['<answer>France</answer>']]
        on = self.rollout(actions, [[[doc()]]])
        inputs = self.last_worker_inputs
        off = self.rollout(actions, [[[doc()]]], record_evidence=False)
        for a, b in zip(inputs, self.last_worker_inputs):
            torch.testing.assert_close(a, b)
        for key in on.batch.keys():
            torch.testing.assert_close(on.batch[key], off.batch[key])

    def test_empty_document_body_cannot_earn_evidence(self):
        output = self.rollout([['<search>Paris</search>'], ['<answer>France</answer>']], [[[doc(body='')]]])
        self.assertEqual(score_evidence(reference(), output.non_tensor_batch['trajectory'][0])['evidence_coverage'], 0.)

    def test_partial_unicode_token_cannot_earn_full_character(self):
        text = '\n\n<information>Doc 1(Title: "Paris") 𰻞</information>\n\n'
        encoded = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        pos = text.index('𰻞')
        parts = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a <= pos < b]
        if len(parts) < 2:
            self.skipTest('Tokenizer does not split this Unicode character')
        output = self.rollout([['<search>Paris</search>'], ['<answer>France</answer>']],
                              [[[doc(body='𰻞')]]], obs_limit=parts[0] + 1)
        ref = reference()
        ref['supporting_facts'] = [dict(fact_id='Paris:0', title='Paris', sentence_id=0, sentence='𰻞')]
        self.assertEqual(score_evidence(ref, output.non_tensor_batch['trajectory'][0])['evidence_coverage'], 0.)

    def test_new_facts_and_duplicate_evidence(self):
        output = self.rollout([['<search>Paris</search>'], ['<search> PARIS </search>'],
                               ['<answer>France</answer>']], [[[doc()]], [[doc()]]])
        trace = output.non_tensor_batch['trajectory'][0]
        score = score_evidence(reference(), trace)
        self.assertEqual([x['evidence_gain'] for x in score['evidence_events']], [.5, 0.])
        self.assertEqual(score['evidence_coverage'], .5)
        self.assertEqual(trace['duplicate_queries'], 1)
        self.assertTrue(trace['events'][0]['visible_views'])

    def test_two_hops_add_different_facts(self):
        output = self.rollout([['<search>Paris</search>'], ['<search>France</search>'],
                               ['<answer>Europe</answer>']], [[[doc()]], [[doc('France', 'France is in Europe.')]]])
        score = score_evidence(reference(), output.non_tensor_batch['trajectory'][0])
        self.assertEqual(score['evidence_coverage'], 1.)
        self.assertEqual([x['evidence_gain'] for x in score['evidence_events']], [.5, .5])

    def test_observation_tail_does_not_earn_reward(self):
        output = self.rollout([['<search>Paris</search>'], ['<answer>France</answer>']],
            [[[doc(body='noise ' * 200 + 'Paris is the capital of France.')]]], obs_limit=32)
        trace = output.non_tensor_batch['trajectory'][0]
        self.assertTrue(trace['events'][0]['observation_truncated'])
        self.assertIn('Paris is the capital of France.', trace['events'][0]['observation_text'])
        self.assertEqual(score_evidence(reference(), trace)['evidence_coverage'], 0.)

    def test_incomplete_sentence_is_not_a_fact(self):
        prefix = '\n\n<information>Doc 1(Title: "Paris") Paris is the capital'
        limit = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        output = self.rollout([['<search>Paris</search>'], ['<answer>France</answer>']],
                              [[[doc()]]], obs_limit=limit)
        self.assertEqual(score_evidence(reference(), output.non_tensor_batch['trajectory'][0])['evidence_coverage'], 0.)

    def test_context_crop_before_generation_does_not_earn_reward(self):
        output = self.rollout([['<search>Paris</search>'], ['<answer>France</answer>']],
            [[[doc(body='Paris is the capital of France. ' + 'noise ' * 200)]]], context_limit=32)
        trace = output.non_tensor_batch['trajectory'][0]
        self.assertFalse(trace['events'][0]['observation_truncated'])
        self.assertEqual(score_evidence(reference(), trace)['evidence_coverage'], 0.)

    def test_earlier_visible_evidence_survives_later_context_crop(self):
        output = self.rollout([['<search>Paris</search>'], ['<search>other</search>'],
            ['<answer>France</answer>']], [[[doc()]], [[doc('Other', 'noise ' * 200)]]], context_limit=128)
        self.assertEqual(score_evidence(reference(), output.non_tensor_batch['trajectory'][0])['evidence_coverage'], .5)

    def test_title_only_match_and_generated_information_are_rejected(self):
        ref = reference()
        ref['supporting_facts'][0]['title'] = 'Paris is the capital of France.'
        output = self.rollout([['<search>Paris</search>'], ['<answer>France</answer>']],
            [[[doc('Paris is the capital of France.', 'Unrelated body.')]]])
        self.assertEqual(score_evidence(ref, output.non_tensor_batch['trajectory'][0])['evidence_coverage'], 0.)
        output = self.rollout([['<think><information>Paris is the capital of France.</information></think><answer>France</answer>']], [])
        self.assertEqual(score_evidence(reference(), output.non_tensor_batch['trajectory'][0])['evidence_coverage'], 0.)

    def test_query_text_is_not_retrieved_evidence(self):
        output = self.rollout([['<search>Paris is the capital of France.</search>'], ['<answer>France</answer>']],
                              [[[doc(body='Unrelated body.')]]])
        self.assertEqual(score_evidence(reference(), output.non_tensor_batch['trajectory'][0])['evidence_coverage'], 0.)

    def test_inactive_rows_and_reordered_reward_metadata(self):
        output = self.rollout([['<answer>France</answer>', '<search>Paris</search>'],
                               ['<answer>France</answer>']], [[[doc()]]], count=2)
        traces = output.non_tensor_batch['trajectory']
        self.assertEqual([score_evidence(reference(), x)['evidence_coverage'] for x in traces], [0., .5])
        tok = self.tokenizer
        prompt = tok(['Use <answer>example</answer>. Question: Where?'] * 2, return_tensors='pt', padding=True)
        response = tok(['<answer>France</answer>'] * 2, return_tensors='pt', padding=True)
        b = DataProto.from_dict(dict(prompts=prompt.input_ids, responses=response.input_ids,
            attention_mask=torch.cat([prompt.attention_mask, response.attention_mask], dim=1)),
            non_tensors=dict(data_source=['hotpotqa'] * 2, index=[0, 1], trajectory=list(traces),
                reward_model=[{'ground_truth': {'target': ['France']}}] * 2,
                extra_info=[{'evidence': reference()} for _ in range(2)]))
        reward = RewardManager(tok, 0, alpha=.5)(b).sum(-1)
        torch.testing.assert_close(reward, torch.tensor([1., 1.25]))
        b.reorder(torch.tensor([1, 0]))
        torch.testing.assert_close(RewardManager(tok, 0, alpha=.5)(b).sum(-1), reward.flip(0))
        torch.testing.assert_close(RewardManager(tok, 0, split='validation')(b).sum(-1), torch.ones(2))

    def test_invalid_annotations_and_missing_provenance_fail_loudly(self):
        raw = dict(id='bad', context=dict(title=['Paris'], sentences=[['One sentence.']]),
                   supporting_facts=dict(title=['Paris'], sent_id=[2]))
        ref = reference_from_raw(raw)
        self.assertEqual(ref['status'], 'invalid_annotation')
        with self.assertRaises(ValueError): score_evidence(ref, {})
        with self.assertRaises(ValueError): score_evidence(reference(), new_trajectory())
        self.assertFalse(contains_sentence('Parisian', 'Paris'))

    def test_denied_search_does_not_create_visible_evidence(self):
        output = self.rollout([['<search>Paris</search>']], [])
        trace = output.non_tensor_batch['trajectory'][0]
        self.assertEqual(trace['retrieval_attempts'], 0)
        self.assertEqual(score_evidence(reference(), trace)['evidence_coverage'], 0.)

    def test_left_and_right_padding_follow_actual_input(self):
        tok = self.tokenizer
        previous = tok.padding_side
        try:
            for side, expected in [('right', .5), ('left', 0.)]:
                tok.padding_side = side
                output = self.rollout([['<search>long</search>', '<search>Paris</search>'],
                    ['<answer>X</answer>', '<answer>France</answer>']],
                    [[[doc('Other', 'noise ' * 200)], [doc()]]], count=2, obs_limit=48,
                    prompts=['Question: X?', 'Question: ' + 'long prompt ' * 20])
                trace = output.non_tensor_batch['trajectory'][1]
                self.assertEqual(score_evidence(reference(), trace)['evidence_coverage'], expected)
        finally:
            tok.padding_side = previous

    def test_reward_guard_and_invalid_reference_policy_integration(self):
        tok = self.tokenizer
        output = self.rollout([['malformed', '<search>Paris</search>'],
                              ['malformed', '<answer>Wrong</answer>']], [[[doc()]]], count=2)
        traces = output.non_tensor_batch['trajectory']
        prompt = tok(['Use <answer>example</answer>. Question: Where?'] * 2, return_tensors='pt', padding=True)
        responses = tok(['malformed', '<answer>Wrong</answer>'], return_tensors='pt', padding=True)
        metadata = [{'evidence': reference()} for _ in range(2)]
        b = DataProto.from_dict(dict(prompts=prompt.input_ids, responses=responses.input_ids,
            attention_mask=torch.cat([prompt.attention_mask, responses.attention_mask], dim=1)),
            non_tensors=dict(data_source=['hotpotqa'] * 2, index=[0, 1], trajectory=list(traces),
                reward_model=[{'ground_truth': {'target': ['France']}}] * 2, extra_info=metadata))
        manager = RewardManager(tok, 0, alpha=.5, beta=.02, require_final_answer=True, max_search_calls=1)
        rewards = manager(b).sum(-1)
        self.assertLess(float(rewards[0]), float(rewards[1]))
        b.non_tensor_batch['extra_info'][1] = {'evidence': dict(reference(), status='invalid_annotation')}
        with self.assertRaises(ValueError): RewardManager(tok, 0, alpha=.5)(b)
        skipped = RewardManager(tok, 0, alpha=.5, invalid_evidence_policy='skip')(b).sum(-1)
        self.assertEqual(float(skipped[1]), 0.)
        del b.non_tensor_batch['extra_info']
        with self.assertRaises(ValueError): RewardManager(tok, 0, alpha=.5)(b)

    def test_repeated_annotation_is_deduplicated(self):
        raw = dict(id='dup', context=dict(title=['Paris'], sentences=[['Paris is the capital of France.']]),
                   supporting_facts=dict(title=['Paris', 'Paris'], sent_id=[0, 0]))
        ref = reference_from_raw(raw)
        self.assertEqual(ref['status'], 'valid')
        self.assertEqual(len(ref['supporting_facts']), 1)

    def test_no_answer_loophole_is_detected_and_opt_in_guard_fixes_it(self):
        invalid, wrong = new_trajectory(), new_trajectory()
        record_action(invalid, None, '', 'malformed')
        record_action(wrong, 'search', 'Paris', '<search>Paris</search>', dict(success=True))
        record_action(wrong, 'answer', 'Wrong', '<answer>Wrong</answer>')
        self.assertGreater(reward_components(0, invalid, beta=.02)['total_reward'],
                           reward_components(0, wrong, beta=.02)['total_reward'])
        options = dict(beta=.02, gamma=.05, alpha=.5, delta=.1,
                       require_final_answer=True, max_search_calls=2)
        bad = reward_components(0, invalid, evidence_gain=1., **options)
        good = reward_components(0, wrong, evidence_gain=0., **options)
        self.assertLess(bad['total_reward'], good['total_reward'])
        self.assertGreater(bad['failure_guard_cost'], 0.)
        self.assertEqual(reward_components(0, invalid)['total_reward'], 0.)
        for alpha in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): reward_components(0, wrong, alpha=alpha)


if __name__ == '__main__':
    unittest.main()
