import copy
import unittest
from search_r1.llm_agent.evidence import score_search_state
from search_r1.llm_agent.trajectory import reward_components


def reference():
    return dict(schema_version=1, status='valid', supporting_facts=[
        dict(fact_id='A:0', title='A', sentence='A is here.')])

def view(generation, text='A is here.', decoded=None):
    return dict(generation=generation, chunks=[dict(decoded_text=text if decoded is None else decoded,
        document_bodies=[dict(title='A', text=text)])])

def trace(views=None):
    return dict(visibility_schema_version=1, retrieval_attempts=2, final_answer='yes',
        events=[dict(turn=0, action='search', attempted=True, success=True,
                     visible_views=views if views is not None else [view(1)]),
                dict(turn=1, action='search', attempted=True, success=False, visible_views=[]),
                dict(turn=2, action='search', attempted=False)])


class StateCostTests(unittest.TestCase):
    def test_only_calls_after_prior_visibility_are_charged(self):
        score = score_search_state(reference(), trace())
        self.assertEqual(score['after_full_search_calls'], 1)
        self.assertEqual([s['coverage_before_search'] for s in score['search_state_events']], [0, 1])

    def test_future_visibility_cannot_penalize_earlier_call(self):
        self.assertEqual(score_search_state(reference(), trace([view(2)]))['after_full_search_calls'], 0)

    def test_partial_or_untrusted_generated_text_does_not_establish_full_state(self):
        for views in ([view(1, 'A is')], [view(1, 'A is here.', 'A is')], []):
            self.assertEqual(score_search_state(reference(), trace(views))['after_full_search_calls'], 0)

    def test_bad_timestamps_and_accounting_fail(self):
        for bad in (None, 0, -1):
            with self.assertRaises(ValueError):
                score_search_state(reference(), trace([view(bad)]))
        t = trace()
        t['retrieval_attempts'] = 3
        with self.assertRaises(ValueError):
            score_search_state(reference(), t)

    def test_more_than_two_calls_and_retained_prior_evidence(self):
        t=trace()
        t['retrieval_attempts']=3
        t['events'][2]=dict(turn=2,action='search',attempted=True,success=False)
        self.assertEqual(score_search_state(reference(),t)['after_full_search_calls'],2)

    def test_cost_scale_and_default_compatibility(self):
        t=trace()
        old=reward_components(1,t,beta=.005,alpha=.1,evidence_gain=1)
        new=reward_components(1,t,beta=.005,alpha=.1,evidence_gain=1,
                              after_full_lambda=.01,after_full_search_calls=1)
        self.assertAlmostEqual(old['total_reward'],1.09)
        self.assertAlmostEqual(new['total_reward'],1.08)
        self.assertAlmostEqual(new['after_full_cost'],.01)
        self.assertEqual(reward_components(1,None)['total_reward'],1)

    def test_no_answer_guard_below_every_valid_wrong_answer(self):
        valid=reward_components(0,trace(),beta=.005,alpha=.1,
            after_full_lambda=.01,after_full_search_calls=1,
            require_final_answer=True,max_search_calls=2)['total_reward']
        t=trace();t['final_answer']=None
        missing=reward_components(0,t,beta=.005,alpha=.1,evidence_gain=1,
            after_full_lambda=.01,after_full_search_calls=1,
            require_final_answer=True,max_search_calls=2)
        self.assertLess(missing['total_reward'],valid)
        self.assertAlmostEqual(missing['total_reward'],-1.03)

    def test_missing_or_out_of_range_state_count_rejected(self):
        for count in (None,-1,3,0.5,True):
            with self.assertRaises(ValueError):
                reward_components(0,trace(),after_full_lambda=.01,after_full_search_calls=count)
        for lam in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                reward_components(0,trace(),after_full_lambda=lam,after_full_search_calls=0)
