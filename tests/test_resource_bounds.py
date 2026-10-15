"""Validate guaranteed resource bounds against independent exact executions."""

from itertools import product
import random
import unittest

from matrix_multiplication import Instrumentation, multiply
from matrix_multiplication.composition import Composition
from matrix_multiplication.resource_bounds import Enclosure, resource_bound
from matrix_multiplication.smirnov import plan as smirnov_plan
from tests.support import requires_coefficients


class ResourceBoundsTests(unittest.TestCase):
    """Check domination of observed widths/word work and exact schedule coverage."""

    def assert_enclosed(self, name, plan, size, cutoff, bound, a, b, policy='fringe'):
        """Compare a bound with ordinary observed arithmetic, never with itself."""
        model = resource_bound(plan, size, cutoff, input_bound=bound, size_policy=policy)
        observer = Instrumentation(profile=False)
        implementation = plan.multiply if name == 'mixed' else None
        if implementation:
            implementation(a, b, cutoff=cutoff, instrumentation=observer, size_policy=policy)
        else:
            multiply(a, b, algorithm=name, cutoff=cutoff, instrumentation=observer, size_policy=policy)
        snapshot = observer.snapshot()
        observed = snapshot['operand_costs']
        for actual, upper in (('observed_max_operand_bits', 'operand_bits_upper_bound'),
                              ('observed_max_result_bits', 'result_bits_upper_bound'),
                              ('observed_max_denominator_bits', 'denominator_bits_upper_bound'),
                              ('equivalent_word_multiplications', 'word_multiplications_upper_bound'),
                              ('equivalent_word_additions_subtractions', 'word_additions_subtractions_upper_bound')):
            self.assertLessEqual(observed[actual], model[upper], (name, size, cutoff, policy, actual))
        self.assertEqual(snapshot['atomic_arithmetic_operations'], sum(model['modelled_atomic_calls'].values()),
                         (name, size, cutoff, policy, 'schedule'))
        self.assertFalse(model['is_python_memory_peak_bound'])

    def test_classical_bounds_are_exact_for_uniform_positive_integer_inputs(self):
        """Positive all-four inputs attain the classical dot-prefix value bound."""
        model = resource_bound(None, 16)
        self.assertEqual(model['operand_bits_upper_bound'], (15*16).bit_length())
        self.assertEqual(model['result_bits_upper_bound'], (16*16).bit_length())
        self.assertEqual(model['word_multiplications_upper_bound'], 16**3)
        self.assertEqual(model['word_additions_subtractions_upper_bound'], 16**3)

    def test_model_neither_allocates_requested_matrices_nor_changes_observation(self):
        """Large abstract dimensions remain unexecuted and invisible to observers."""
        from matrix_multiplication.instrumentation import session
        from unittest.mock import patch
        observer = Instrumentation(profile=False)
        plan = Composition(['strassen-winograd'])
        with session(observer), patch.object(plan, 'multiply', side_effect=AssertionError('matrix execution')):
            bound = resource_bound(plan, 2**40+1)
        self.assertGreater(bound['word_additions_subtractions_upper_bound'], 2**53)
        self.assertEqual(observer.snapshot()['atomic_arithmetic_operations'], 0)

    def test_binary_formulas_cover_all_signed_two_by_two_inputs(self):
        """Exhaust all 256 pairs with entries ±1, including complete cancellations."""
        matrices = [[list(values[:2]), list(values[2:])] for values in product((-1, 1), repeat=4)]
        for name in ('strassen', 'strassen-winograd'):
            plan = Composition([name])
            for a in matrices:
                for b in matrices:
                    self.assert_enclosed(name, plan, 2, 1, 1, a, b)

    def test_fringe_cutoff_padding_and_large_input_words(self):
        """Work is bounded for each size policy and across the 64-bit boundary."""
        rng = random.Random(173)
        for name in ('naive', 'strassen', 'strassen-winograd'):
            plan = None if name == 'naive' else Composition([name])
            for size in (3, 5, 8):
                for cutoff in (1, 3):
                    for policy in ('fringe', 'padded'):
                        bound = 2**70
                        a = [[rng.choice((-bound, 0, bound)) for _ in range(size)] for _ in range(size)]
                        b = [[rng.choice((-bound, 0, bound)) for _ in range(size)] for _ in range(size)]
                        self.assert_enclosed(name, plan, size, cutoff, bound, a, b, policy)

    @requires_coefficients('alphaevolve', 'smirnov')
    def test_gaussian_rational_and_mixed_schedules(self):
        """Bound zero imaginary parts, rational denominators and normalizations."""
        plans = [('alphaevolve', Composition(['alphaevolve'])), ('smirnov', smirnov_plan()),
                 ('mixed', Composition(['strassen', 'alphaevolve']))]
        for name, plan in plans:
            for size in (4, 5, 7, 9):
                a = [[(-1)**(i+j)*4 for j in range(size)] for i in range(size)]
                b = [[4 if (i+j)%3 else -4 for j in range(size)] for i in range(size)]
                self.assert_enclosed(name, plan, size, 1, 4, a, b)

    def test_reduced_denominator_and_zero_invariants(self):
        """LCM enclosures allow reduction without asserting nonexistent cancellation."""
        from fractions import Fraction
        a, b = Enclosure(Fraction(3, 5), 5), Enclosure(Fraction(2, 7), 7)
        self.assertEqual(a.plus(b).denominator, 35)
        self.assertEqual(a.plus(b).numerator_bound, 31)
        self.assertEqual(Enclosure(0, 900).denominator, 1)
        for size, bound in ((0,4),(True,4),(2,-1),(2,True)):
            with self.assertRaises(ValueError):
                resource_bound(None, size, input_bound=bound)

    def test_bound_violation_fails_and_preserves_captured_evidence(self):
        """An invalid upper bound cannot quietly replace or hide a measurement."""
        from matrix_multiplication.work_sweep import collect_work
        from matrix_multiplication.resource_bounds import bound_range
        data = collect_work([2], [1], algorithms=['naive'], projection_cycles=1, measure_memory=False)
        line = data['series'][0]
        sample = line['samples'][0]
        sample['observation']['operand_costs']['observed_max_operand_bits'] = 10**6
        result = bound_range('naive', 1, line['samples'], line['range_projection'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(sample['status'], 'failed')
        self.assertEqual(sample['observation']['operand_costs']['observed_max_operand_bits'], 10**6)


if __name__ == '__main__':
    unittest.main()
