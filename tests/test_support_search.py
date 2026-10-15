"""Verify incidence pruning never substitutes for a complete tensor identity."""

from fractions import Fraction
import unittest

from matrix_multiplication.support_search import diagonal_pairs, prioritized_subsets, search_support
from matrix_multiplication.tensor_restrictions import verify_restriction
from matrix_multiplication.tensors import Tensor, classical, linear_maps


class SupportSearchTests(unittest.TestCase):
    """Check exact cancellations, nonmatrix regular graphs, scopes and schedules."""

    def test_fair_pair_order_and_lazy_subset_deduplication(self):
        """Both input streams advance before the first stream fixes an entire prefix."""
        pairs = list(diagonal_pairs(iter(range(3)), iter(range(4))))
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len(set(pairs)), 12)
        self.assertEqual(pairs[:3], [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(list(diagonal_pairs((), range(4))), [])
        subsets = list(prioritized_subsets((0, 1, 2, 3), 2, ((0, 2), (2, 0), (1, 3))))
        self.assertEqual(subsets[:2], [(0, 2), (1, 3)])
        self.assertEqual(len(subsets), 6)

    def test_scrambled_embedded_matrix_is_fully_recognized(self):
        """Unused source variables and basis permutations must not hide a true product."""
        base = classical((2, 2, 2))
        coordinates = ((4, 1, 3, 0), (5, 2, 4, 1), (6, 0, 3, 2))
        maps = tuple(tuple(((i, 1),) for i in selected) for selected in coordinates)
        source = linear_maps(base, (5, 6, 7), maps)
        witnesses, report = search_support(source)
        self.assertTrue(witnesses)
        self.assertGreater(report['recognized_candidates'], 0)
        for witness in witnesses:
            self.assertEqual(verify_restriction(source, witness).expand(), base.expand())
            self.assertEqual([set(s) for s in witness.selections], [set(s) for s in coordinates])

    def test_regular_support_degrees_are_not_a_matrix_proof(self):
        """A symmetric four-cycle has all the right degrees but the wrong neighborhoods."""
        triples = ((0, 1, 0), (1, 0, 0), (0, 2, 1), (2, 0, 1),
                   (1, 3, 2), (3, 1, 2), (2, 3, 3), (3, 2, 3))
        source = Tensor((4, 4, 4), tuple(tuple(((i, 1),) for i in t) for t in triples))
        witnesses, report = search_support(source, candidate_budget=5)
        self.assertFalse(witnesses)
        self.assertEqual(report['input_pair_screens'], 1)
        self.assertEqual(report['layout_proposals'], 1)
        layout = report['candidates'][1]
        self.assertEqual(layout['left_degrees'], [2]*4)
        self.assertEqual(layout['right_degrees'], [2]*4)
        self.assertEqual(layout['status'], 'not recognized')
        self.assertTrue(report['stream_exhausted'])

    def test_exact_cancellations_and_nonunit_coefficients(self):
        """Pruning expands coefficients exactly, rather than counting decomposition terms."""
        source = classical((2, 2, 2))
        added = (((0, Fraction(1, 3)),), ((0, 1),), ((0, 1),))
        cancelled = (((0, Fraction(-1, 3)),), ((0, 1),), ((0, 1),))
        witnesses, _ = search_support(Tensor(source.dimensions, (*source.terms, added, cancelled)))
        self.assertEqual(len(witnesses), 1)
        witnesses, report = search_support(Tensor(source.dimensions, (*source.terms, added)))
        self.assertFalse(witnesses)
        self.assertEqual(report['candidates'][0]['status'], 'insufficient degree-compatible outputs')

    def test_attempt_units_pool_limits_and_invalid_zero_budget(self):
        """No hidden output-subset loop or unrecorded exhaustion probe exceeds a cap."""
        source = classical((2, 2, 2))
        witnesses, report = search_support(source, candidate_budget=1)
        self.assertFalse(witnesses)
        self.assertEqual(report['input_pair_screens'], 1)
        self.assertTrue(report['exhaustion_unknown_at_budget'])
        witnesses, report = search_support(source, candidate_budget=2)
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(report['attempted_candidates'], report['input_pair_screens']+report['layout_proposals'])
        self.assertTrue(report['budget_reached'])
        self.assertFalse(report['stream_exhausted'])
        witnesses, report = search_support(source, pools=((0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3)))
        self.assertFalse(witnesses)
        self.assertEqual(report['attempted_candidates'], 0)
        self.assertTrue(report['stream_exhausted'])
        with self.assertRaises(ValueError):
            search_support(source, pools=((True, 1, 2, 3),)*3, candidate_budget=0)
        for budget in (True, -1, 1.0):
            with self.assertRaises(ValueError):
                search_support(source, candidate_budget=budget)


if __name__ == '__main__':
    unittest.main()
