"""Verify exact unpadded execution and independently derived boundary budgets."""

from copy import deepcopy
from fractions import Fraction
import unittest

from matrix_multiplication import Instrumentation, multiply
from matrix_multiplication.composition import Composition
from matrix_multiplication.rectangular import RectangularCycle
from matrix_multiplication.tensors import classical, to_scheme
from matrix_multiplication.verification import check_product
from matrix_multiplication.work_sweep import collect_work
from tests.support import available_algorithms


class FringeTests(unittest.TestCase):
    """Exercise each boundary axis, recursion, exact domains, and the old baseline."""

    def test_strassen_counts_remove_whole_input_plateaus(self):
        """Core products plus complementary cubic terms give explicit small counts."""
        for size, products in ((3, 26), (5, 110), (6, 182), (7, 309), (8, 343), (9, 560)):
            a = [[i-j for j in range(size)] for i in range(size)]
            b = [[i+j for j in range(size)] for i in range(size)]
            observer = Instrumentation(profile=False)
            result = multiply(a, b, algorithm='strassen', instrumentation=observer)
            self.assertTrue(check_product(a, b, result).passed)
            self.assertEqual(observer.counts['field_multiplications'], products)
            self.assertEqual(observer.counts['padded_input_cells'], 0)
            self.assertEqual(observer.primitive_calls['matrix.pad'], 0)
            self.assertEqual(observer.primitive_calls['matrix.crop'], 0)

    def test_all_algorithms_preserve_exact_rectangles_and_inputs(self):
        """Every orientation of the boundary and skinny fallbacks stays exact."""
        for name, implementation in available_algorithms().items():
            for rows, inner, columns in ((5, 5, 5), (5, 7, 9), (7, 9, 5), (9, 5, 7),
                                         (1, 7, 5), (5, 1, 7), (7, 5, 1), (6, 7, 8)):
                a = [[Fraction((1 << 90)+i-j, j+1) for j in range(inner)] for i in range(rows)]
                b = [[Fraction(i+j-3, i+2) for j in range(columns)] for i in range(inner)]
                original = deepcopy((a, b))
                with self.subTest(name=name, shape=(rows, inner, columns)):
                    observer = Instrumentation(profile=False)
                    result = implementation(a, b, instrumentation=observer)
                    self.assertTrue(check_product(a, b, result).passed)
                    self.assertEqual((a, b), original)
                    self.assertEqual(observer.counts['padded_input_cells'], 0)

    def test_cutoff_is_checked_on_actual_dimensions(self):
        """Size three at cutoff three stays classical without enlarging to four."""
        observer = Instrumentation(profile=False)
        a = [[1, 2, 3]]*3
        multiply(a, a, algorithm='strassen', cutoff=3, instrumentation=observer)
        self.assertEqual(observer.counts['field_multiplications'], 27)
        self.assertEqual(observer.counts['recursive_nodes'], 1)
        self.assertEqual(observer.counts['fringe_nodes'], 0)

    def test_boundary_products_keep_their_actual_recursion_depth(self):
        """A 7×7 call has boundary work both at the root and its seven children."""
        observer = Instrumentation(profile=False, trace=True, trace_limit=20)
        a = [[0]*7 for _ in range(7)]
        multiply(a, a, algorithm='strassen', instrumentation=observer)
        self.assertEqual(observer.products_by_depth, {0: 127, 1: 133, 2: 49})
        self.assertEqual(observer.counts['fringe_data_products'], 260)
        self.assertEqual(observer.counts['fringe_nodes'], 8)
        self.assertEqual(observer.counts['field_multiplications'], 309)

    def test_padded_baseline_retains_its_schedule(self):
        """Historical arithmetic and before-cutoff padding remain reproducible."""
        observer = Instrumentation(profile=False)
        a = [[1, 2, 3]]*3
        result = multiply(a, a, algorithm='strassen', cutoff=3,
                          size_policy='padded', instrumentation=observer)
        self.assertTrue(check_product(a, a, result).passed)
        self.assertEqual(observer.counts['field_multiplications'], 56)
        self.assertEqual(observer.counts['padded_input_cells'], 32)

    def test_mixed_and_rectangular_cycles_match_all_budget_layers(self):
        """No power-of-two or square-area assumption enters the boundary budgets."""
        shapes = ((2, 2, 3), (2, 3, 2), (3, 2, 2))
        rectangular = RectangularCycle(tuple(to_scheme(classical(s), s, 'test') for s in shapes))
        four = to_scheme(classical((4, 4, 4)), (4, 4, 4), 'four')
        for plan in (Composition(['strassen', four]), rectangular):
            for shape in ((5, 7, 9), (9, 9, 9), (3, 1, 5), (12, 12, 12)):
                rows, inner, columns = shape
                a = [[i-j for j in range(inner)] for i in range(rows)]
                b = [[i+j+1 for j in range(columns)] for i in range(inner)]
                observer = Instrumentation(profile=False)
                result = plan.multiply(a, b, instrumentation=observer)
                self.assertTrue(check_product(a, b, result).passed)
                budget = plan.budget(*shape)
                snapshot = observer.snapshot()
                for key in ('field_arithmetic_operations', 'atomic_arithmetic_operations'):
                    self.assertEqual(snapshot[key], budget[key])
                for key, value in budget['category_counts'].items():
                    self.assertEqual(observer.counts[key], value)

    def test_sweep_counts_categories_and_policy_are_explicit(self):
        """The graph source carries verified boundary work and zero added cells."""
        data = collect_work([3, 5, 6, 9], [1, 3], algorithms=['strassen', 'strassen-winograd'],
                            measure_memory=False, projection_cycles=2)
        for line in data['series']:
            for sample in line['samples']:
                self.assertEqual(sample['status'], 'passed')
                self.assertTrue(sample['category_prediction_matches_observed'])
                self.assertTrue(sample['prediction_matches_observed_arithmetic_steps'])
                self.assertEqual(sample['size_effects']['extra_zero_input_cells'], 0)
                self.assertEqual(sample['size_effects']['size_policy'], 'fringe')

    def test_invalid_policy_is_rejected(self):
        """A typo must not silently select an execution policy."""
        for name, implementation in available_algorithms().items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                implementation([[1]], [[2]], size_policy='pad')
        with self.assertRaises(ValueError):
            Composition(['strassen']).budget(5, 5, 5, size_policy='pad')


if __name__ == '__main__':
    unittest.main()
