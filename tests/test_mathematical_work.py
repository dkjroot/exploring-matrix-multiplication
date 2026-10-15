"""Check independent arithmetic budgets, complete coverage and honest comparisons.

Small hand-calculated schedules test the cost definition. Exhaustive signed
rational examples check replay against Fraction without changing scalar formulas.
End-to-end tests bind scores to the actual input and keep older evidence readable.
"""

from fractions import Fraction
import json
from random import Random
import unittest

from matrix_multiplication import Instrumentation, multiply
from matrix_multiplication.common import crop, padded, split, join
from matrix_multiplication.data_movement import payload_words
from matrix_multiplication.instrumentation import session, suspended
from matrix_multiplication.mathematical_work import MathematicalWork, ReferenceArithmetic
from matrix_multiplication.scalars import Gaussian, add, subtract, scale, divide, multiply as product
from matrix_multiplication.work_comparison import classical_reference, attach_comparison
from matrix_multiplication.work_sweep import collect_work
from matrix_multiplication.reporting import render_html


class MathematicalWorkTests(unittest.TestCase):
    """Protect work units, rational reduction and noninterference with algorithms."""

    def test_literal_integer_and_division_budgets(self):
        """Six limb pairs cost six products, twelve accumulations and twelve carries."""
        ref = ReferenceArithmetic()
        self.assertEqual(ref.product(1<<64, 1<<128), 1<<192)
        self.assertEqual(ref.counts['word_products'], 6)
        self.assertEqual(ref.counts['word_additions'], 12)
        self.assertEqual(ref.counts['carry_borrow_steps'], 12)
        division = ReferenceArithmetic()
        self.assertEqual(division.division(10, 2, exact=True), (5, 0))
        self.assertEqual(division.counts['division_bit_steps'], 4)
        self.assertEqual(division.counts['word_shifts'], 8)
        self.assertEqual(division.counts['word_comparisons'], 4)
        self.assertEqual(division.counts['word_subtractions'], 2)
        self.assertEqual(division.counts['carry_borrow_steps'], 2)
        with self.assertRaises(ArithmeticError):
            division.division(3, 2, exact=True)

    def test_integer_normalization_and_opaque_division_have_explicit_coverage(self):
        """Checked signed integer quotients are charged; floating results are not exact scores."""
        observer = Instrumentation(profile=False)
        with session(observer):
            self.assertEqual(divide(-10, 2), -5)
        work = observer.snapshot()['mathematical_work']
        self.assertEqual(work['schedule_statistics']['exact_divisions'], 1)
        self.assertEqual(work['schedule_statistics']['normalizations'], 1)
        self.assertEqual(work['schedule_statistics']['gcd_calls'], 0)
        self.assertEqual(work['status'], 'modelled')
        self.assertEqual(observer.snapshot()['operand_costs']['status'], 'partial')
        ref = MathematicalWork()
        ref.observe('real.divide', (1, 2), 0.5, 'normalization', 'ok')
        self.assertEqual(ref.snapshot()['status'], 'partial')

    def test_subtraction_and_coefficients_remain_separate(self):
        """Magnitude subtraction, field subtraction and fixed scaling have distinct scopes."""
        observer = Instrumentation(profile=False)
        with session(observer):
            self.assertEqual(subtract(7, 2), 5)
            self.assertEqual(add(7, -2), 5)
            self.assertEqual(scale(7, -2), -14)
        snapshot = observer.snapshot()
        self.assertEqual(snapshot['counts']['field_subtractions'], 1)
        self.assertEqual(snapshot['counts']['field_additions'], 1)
        self.assertEqual(snapshot['mathematical_work']['work_categories']['word_subtractions'], 2)
        self.assertEqual(snapshot['mathematical_work']['work_by_role']['coefficient'], 6)

    def test_all_signed_rational_operations_replay_exactly(self):
        """Negative divisors, zero sums, unit denominators and shared factors survive."""
        rng = Random(43)
        observer = Instrumentation(profile=False)
        with session(observer):
            for _ in range(80):
                a = Fraction(rng.randrange(-20, 21), rng.randrange(1, 20))
                b = Fraction(rng.randrange(-20, 21), rng.randrange(1, 20))
                self.assertEqual(add(a, b), a+b)
                self.assertEqual(subtract(a, b), a-b)
                self.assertEqual(product(a, b), a*b)
                if b:
                    self.assertEqual(divide(a, b), a/b)
        work = observer.snapshot()['mathematical_work']
        self.assertEqual(work['status'], 'modelled', work['unmodelled_calls'])
        self.assertGreater(work['schedule_statistics']['gcd_calls'], 0)
        self.assertGreater(work['schedule_statistics']['exact_divisions'], 0)
        self.assertGreater(work['schedule_statistics']['normalization_reductions'], 0)
        self.assertEqual(work['total_work_units'], sum(work['work_categories'].values()))

    def test_cross_cancellation_charges_gcd_and_saves_wide_products(self):
        """A reciprocal pair pays reduction costs before multiplying one-word residues."""
        a = Fraction((1<<128)+1, (1<<129)+1)
        observer = Instrumentation(profile=False)
        with session(observer):
            self.assertEqual(product(a, 1/a), 1)
        stats = observer.snapshot()['mathematical_work']['schedule_statistics']
        self.assertEqual(stats['cross_cancellations'], 2)
        self.assertEqual(stats['word_products'], 1)
        self.assertEqual(stats['word_products_saved_by_cross_cancellation'], 17)
        self.assertGreater(stats['gcd_iterations'], 0)
        self.assertEqual(stats['exact_divisions'], 4)
        self.assertEqual(observer.snapshot()['operand_costs']['equivalent_word_multiplications'], 18)

    def test_unknown_failed_and_mismatched_calls_cannot_get_a_score(self):
        """Never equate covered work with complete work, or a failed replay with proof."""
        observer = Instrumentation(profile=False)
        with session(observer):
            product(1, 2)
            product(0.1, 0.2)
        self.assertIsNone(observer.snapshot()['mathematical_work']['total_work_units'])
        self.assertEqual(observer.snapshot()['mathematical_work']['status'], 'partial')
        work = MathematicalWork()
        work.observe('real.add', (1, 2), 999, None, 'ok')
        self.assertEqual(work.snapshot()['covered_work_units'], 0)
        self.assertIn('real.add:reference_mismatch', work.snapshot()['unmodelled_calls'])
        reference = classical_reference([[1]], [[2]])
        target = {'status': 'passed', **observer.snapshot()}
        attach_comparison(target, reference, input_hash=reference['input_repr_sha256'])
        self.assertEqual(target['work_comparison']['status'], 'unavailable')
        exact = {'status': 'passed', **Instrumentation().snapshot()}
        attach_comparison(exact, reference, input_hash='another input')
        self.assertEqual(exact['work_comparison']['status'], 'unavailable')

    def test_gaussian_expansion_and_observer_suspension_preserve_values(self):
        """Four component products are charged once; replay never enters field counters."""
        observer = Instrumentation(profile=False)
        with session(observer):
            result = product(Gaussian(Fraction(1, 3), 2), Gaussian(3, Fraction(2, 7)))
            before = observer.snapshot()
            with suspended():
                product(1<<4095, 1<<4095)
            self.assertEqual(observer.snapshot(), before)
        self.assertEqual((result.real, result.imag), (Fraction(3, 7), Fraction(128, 21)))
        self.assertEqual(observer.counts['field_multiplications'], 1)
        self.assertEqual(observer.counts['real_multiplications'], 4)
        self.assertEqual(observer.snapshot()['mathematical_work']['status'], 'modelled')


class MovementAndComparisonTests(unittest.TestCase):
    """Check value traffic and scores against tiny, independently countable products."""

    def test_one_cell_classical_work_and_traffic(self):
        """One product and accumulation cost ten units and ten one-word transfers."""
        observer = Instrumentation(profile=False)
        self.assertEqual(multiply([[2]], [[3]], instrumentation=observer), [[6]])
        snapshot = observer.snapshot()
        self.assertEqual(snapshot['mathematical_work']['total_work_units'], 10)
        movement = snapshot['data_movement']
        # Scalar ports: 4 reads + 2 writes. Structure: column read/write,
        # one zero initialization, and one output store. Inputs are not copied.
        self.assertEqual(movement['logical_word_transfers'], 10)
        self.assertEqual(movement['by_category']['arithmetic_ports']['word_reads'], 4)
        self.assertEqual(movement['by_category']['structural']['word_writes'], 3)
        self.assertFalse(movement['is_hardware_memory_traffic'])

    def test_payload_sizes_and_explicit_copies(self):
        """Word size follows values, while copied Python references stay modelled payloads."""
        self.assertEqual(payload_words(1<<4095), 64)
        self.assertEqual(payload_words(Fraction(1<<64, 3)), 3)
        self.assertEqual(payload_words(Gaussian(1<<64, Fraction(1, 3))), 4)
        self.assertIsNone(payload_words(1.5))
        observer = Instrumentation(profile=False)
        a = [[1<<4095, 2], [3, 4]]
        with session(observer):
            self.assertEqual(join(split(a)), a)
            self.assertEqual(crop(padded(a, 3), 2, 2), a)
        movement = observer.snapshot()['data_movement']
        self.assertEqual(movement['counts']['word_reads'], 4*(64+3))
        self.assertEqual(movement['counts']['word_writes'], 4*(64+3)+9)

    def test_scores_reuse_baseline_and_keep_cutoffs_and_raw_evidence(self):
        """Classical is exactly one; new-model totals never replace old recurrence counts."""
        data = collect_work([1, 2, 3], [1, 2], algorithms=['naive', 'strassen'], measure_memory=False, projection_cycles=1)
        self.assertEqual(data['execution_summary']['classical_reference_executions'], 0)
        for line in data['series']:
            for sample in line['samples']:
                score = sample['work_comparison']
                self.assertEqual(score['status'], 'compared')
                self.assertEqual(score['input_repr_sha256'], sample['input_repr_sha256'])
                self.assertTrue(sample['category_prediction_matches_observed'])
                if line['algorithm'] == 'naive':
                    self.assertEqual(score['relative_efficiency'], {'numerator': 1, 'denominator': 1})
        rendered = render_html(data, title='Work', source_name='test.json', source_hash='abc', generated='now')
        for phrase in ('Mathematical work efficiency', 'Measured Python memory growth', 'Logical data movement',
                       'carry/borrow', 'Euclidean GCD', 'not a speedup prediction', 'Unknown payload accesses'):
            self.assertIn(phrase, rendered)
        roundtrip = json.loads(json.dumps(data, allow_nan=False))
        self.assertEqual(roundtrip['series'][0]['samples'][0]['work_comparison']['relative_efficiency']['denominator'], 1)

    def test_missing_classical_selection_still_gets_a_bound_reference(self):
        """Selecting a single candidate cannot silently remove the score denominator."""
        data = collect_work([2], [1], algorithms=['strassen'], measure_memory=False, projection_cycles=1)
        self.assertEqual(data['execution_summary']['classical_reference_executions'], 1)
        score = data['series'][0]['samples'][0]['work_comparison']
        # Classical: 8 products at 6 units, 8 sums at 4 units. Strassen additionally
        # pays 18 block sums/differences; signs do not change the four-unit scan.
        self.assertEqual(score['baseline_work_units'], 80)
        self.assertEqual(score['candidate_work_units'], 142)
        self.assertEqual(score['relative_efficiency'], {'numerator': 40, 'denominator': 71})


if __name__ == '__main__':
    unittest.main()
