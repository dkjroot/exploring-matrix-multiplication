"""Verify exact values survive bit-cost observation and scope is never invented.

Known integer widths and literal rational expressions give independent word-work
expectations. Gaussian calls check atomic expansion, unavailable float/division
work remains explicit, and huge trace values cannot disrupt exact execution.
"""

from fractions import Fraction
import unittest

from matrix_multiplication.instrumentation import Instrumentation, session, suspended
from matrix_multiplication.operand_costs import word_cost
from matrix_multiplication.scalars import Gaussian, add, divide, multiply, scale, subtract


class OperandCostTests(unittest.TestCase):
    """Check width boundaries, rational growth, complex expansion and exclusions."""

    def test_word_products_follow_both_operand_widths_without_narrowing(self):
        """A 4096-bit pair costs 4096 schoolbook word products, not 64."""
        for a, b, expected in (((1<<64)-1, (1<<64)-1, 1), (1<<64, 1<<128, 6),
                               (1<<4095, 1<<4095, 4096)):
            with self.subTest(expected=expected):
                observer = Instrumentation(profile=False)
                with session(observer):
                    result = multiply(a, b)
                self.assertEqual(result, a*b)
                cost = observer.snapshot()['operand_costs']
                self.assertEqual(cost['equivalent_word_multiplications'], expected)
                self.assertEqual(cost['observed_max_operand_bits'], max(a.bit_length(), b.bit_length()))
                self.assertEqual(cost['observed_max_result_bits'], result.bit_length())
                self.assertEqual(observer.counts['field_multiplications'], 1)
                self.assertFalse(cost['is_hardware_instruction_count'])

    def test_fractions_include_cross_products_and_remain_exact(self):
        """Rational sums cost integer products even though field additions stay one."""
        observer = Instrumentation(profile=False)
        with session(observer):
            total = add(Fraction(3, 5), Fraction(2, 7))
            difference = subtract(Fraction(3, 5), Fraction(2, 7))
        self.assertEqual(total, Fraction(31, 35))
        self.assertEqual(difference, Fraction(11, 35))
        cost = observer.snapshot()['operand_costs']
        self.assertEqual(cost['equivalent_word_multiplications'], 6)
        self.assertEqual(cost['equivalent_word_additions_subtractions'], 2)
        self.assertEqual(observer.counts['field_multiplications'], 0)
        self.assertEqual(cost['primitive_word_costs']['real.add']['word_multiplications'], 3)
        # Unit denominators are copies, not artificial integer products.
        self.assertEqual(word_cost('add', (3, 1), (2, 7)), (1, 1))

    def test_cancellation_model_is_declared_and_not_claimed_as_python_work(self):
        """Observe stored widths even when Fraction cancels a reciprocal product."""
        a = Fraction((1<<128)+1, (1<<129)+1)
        b = 1/a
        observer = Instrumentation(profile=False)
        with session(observer):
            result = multiply(a, b)
        self.assertEqual(result, 1)
        cost = observer.snapshot()['operand_costs']
        self.assertEqual(cost['equivalent_word_multiplications'], 18)
        self.assertIn('unreduced rational', cost['model'])
        self.assertIn('GCD and cancellation', cost['excluded_work'])

    def test_gaussian_and_coefficient_roles_expand_into_atomic_word_work(self):
        """Field rank, coefficient scaling and component work stay separate."""
        big = 1<<64
        observer = Instrumentation(profile=False)
        with session(observer):
            result = multiply(Gaussian(big, big), Gaussian(big, big))
            scaled = scale(big, big)
        self.assertEqual((result.real, result.imag), (0, 2*big*big))
        self.assertEqual(scaled, big*big)
        cost = observer.snapshot()['operand_costs']
        self.assertEqual(cost['word_multiplications_by_role']['data'], 16)
        self.assertEqual(cost['word_multiplications_by_role']['coefficient'], 4)
        self.assertEqual(observer.counts['field_multiplications'], 1)
        self.assertEqual(observer.counts['coefficient_multiplications'], 1)

    def test_unsupported_calls_and_suspended_setup_are_explicit(self):
        """No conversion of floating samples or a hidden model for exact division."""
        observer = Instrumentation(profile=False)
        with session(observer):
            with suspended():
                multiply(1<<4095, 1<<4095)
            multiply(0.1, 0.2)
            self.assertEqual(divide(10, 2), 5)
        cost = observer.snapshot()['operand_costs']
        self.assertEqual(cost['status'], 'partial')
        self.assertEqual(cost['equivalent_word_multiplications'], 0)
        self.assertEqual(cost['unmodelled_calls'], {'real.multiply': 1, 'integer.divmod': 1})
        self.assertEqual(cost['observed_max_operand_bits'], 4)

    def test_huge_trace_values_use_explicit_summaries_and_keep_exact_results(self):
        """Decimal rendering limits cannot make an instrumented primitive fail."""
        value = 1<<65535
        observer = Instrumentation(profile=False, trace=True, capture_values=True)
        with session(observer):
            result = multiply(value, value)
        self.assertEqual(result, 1<<131070)
        event = next(event for event in observer.events if event['type'] == 'arithmetic')
        self.assertIn('exact value summary', event['result'])
        self.assertEqual(observer.snapshot()['operand_costs']['equivalent_word_multiplications'], 1024**2)


if __name__ == '__main__':
    unittest.main()
