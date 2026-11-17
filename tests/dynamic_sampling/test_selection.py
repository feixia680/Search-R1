import math
import unittest
from search_r1.dynamic_sampling.selection import BoundedGroupSelector


class SelectionTests(unittest.TestCase):
    def test_all_wrong_with_different_evidence_is_retained(self):
        s = BoundedGroupSelector(1)
        s.offer(0, ['q', 'q'], [.025, .075])
        groups, metrics = s.finish()
        self.assertEqual(groups[0].sample_id, 'q')
        self.assertEqual(metrics['selected_informative_groups'], 1)

    def test_refill_keeps_whole_groups(self):
        s = BoundedGroupSelector(2)
        s.offer(0, ['a', 'a', 'b', 'b'], [0, 1, 0, 0])
        self.assertFalse(s.ready)
        s.offer(1, ['c', 'c', 'd', 'd'], [.025, .075, 1, 1])
        groups, metrics = s.finish()
        self.assertEqual([g.sample_id for g in groups], ['a', 'c'])
        self.assertTrue(all(len(g.row_indices) == 2 for g in groups))
        self.assertEqual(metrics['candidate_trajectories'], 8)
        self.assertEqual(metrics['selected_trajectories'], 4)

    def test_capped_zero_fallback_is_deterministic(self):
        def run():
            s = BoundedGroupSelector(2, max_batches=3, seed=17)
            for i in range(3):
                ids = [f'{i}a']*2 + [f'{i}b']*2
                s.offer(i, ids, [0, 0, 1.1, 1.1])
            return s.finish()
        a, metrics = run()
        self.assertEqual(a, run()[0])
        self.assertEqual(metrics['zero_fallback_groups'], 2)
        self.assertEqual(metrics['candidate_batches'], 3)
        self.assertEqual(metrics['budget_exhausted'], 1)

    def test_uniform_control_does_not_filter_or_refill(self):
        s = BoundedGroupSelector(2, enabled=False)
        s.offer(0, ['a', 'a', 'b', 'b'], [0, 0, 0, 1])
        groups, metrics = s.finish()
        self.assertEqual([g.sample_id for g in groups], ['a', 'b'])
        self.assertEqual(metrics['candidate_batches'], 1)
        self.assertEqual(metrics['zero_fallback_groups'], 0)

    def test_cross_round_duplicate_cannot_merge_into_four_rollouts(self):
        s = BoundedGroupSelector(2, max_batches=2)
        s.offer(0, ['a', 'a', 'b', 'b'], [0, 1, 0, 0])
        s.offer(1, ['a', 'a', 'c', 'c'], [0, 1, 0, 1])
        groups, metrics = s.finish()
        self.assertEqual([g.sample_id for g in groups], ['a', 'c'])
        self.assertEqual(metrics['duplicate_groups'], 1)

    def test_reject_partial_or_duplicate_question_groups(self):
        for ids, rewards in [(['a','a','a','a'],[0,1,0,1]),
                             (['a','b','b','b'],[0,0,1,0]),
                             (['a','a'],[0,1])]:
            with self.assertRaises(ValueError):
                BoundedGroupSelector(2).offer(0, ids, rewards)

    def test_reject_nonfinite_outcomes(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(FloatingPointError):
                BoundedGroupSelector(1).offer(0, ['a','a'], [0,value])

    def test_no_early_fallback_or_extra_batches(self):
        s = BoundedGroupSelector(1, max_batches=2)
        s.offer(0, ['a','a'], [0,0])
        with self.assertRaises(ValueError): s.finish()
        s.offer(1, ['b','b'], [0,0])
        with self.assertRaises(ValueError): s.offer(2, ['c','c'], [0,1])
        s.finish()
        with self.assertRaises(ValueError): s.finish()

    def test_zero_threshold_uses_float32_outcomes(self):
        s = BoundedGroupSelector(1, max_batches=1)
        s.offer(0, ['a','a'], [1,1+1e-10])
        _, metrics = s.finish()
        self.assertEqual(metrics['selected_zero_groups'],1)


if __name__ == '__main__': unittest.main()
