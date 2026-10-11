"""Independent correctness, exactness, input-contract, and coefficient regressions.

Matrix-unit pairs verify entire bilinear base identities; seeded examples exercise
recursion, padding, and numeric domains. Tests deliberately do not assume a naive
floating-point result is an accurate oracle.
"""

import copy
from fractions import Fraction
import json
from pathlib import Path
import random
import unittest

from matrix_multiplication import ALGORITHMS, multiply
from matrix_multiplication import alphaevolve
from matrix_multiplication.verification import check_product
from tests.support import available_algorithms, requires_coefficients


class AlgorithmTests(unittest.TestCase):
    """Exercise the same matrix contract across every registered algorithm."""

    def assert_products(self, a, b, expected=None, cutoffs=(1,)):
        """Compare exact/tolerant results, explicit expectations, and unchanged inputs."""
        original = copy.deepcopy((a, b))
        for name, implementation in available_algorithms().items():
            for cutoff in cutoffs:
                with self.subTest(algorithm=name, cutoff=cutoff, shape=(len(a), len(b), len(b[0]))):
                    result = implementation(a, b, cutoff=cutoff)
                    if expected is not None:
                        self.assertEqual(result, expected)
                    check = check_product(a, b, result)
                    self.assertTrue(check.passed, check)
                    self.assertEqual((a, b), original)

    def test_known_products(self):
        """Check hand-computed scalar, square, and rectangular products."""
        self.assert_products([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[19, 22], [43, 50]])
        self.assert_products([[1, -2, 3], [4, 0, -1]], [[2, 1], [3, -2], [4, 5]], [[8, 20], [4, -1]])
        self.assert_products([[7]], [[-9]], [[-63]])

    def test_zero_and_identity(self):
        """Check zero annihilation and both sides of the identity law."""
        a = [[i * 4 + j - 7 for j in range(4)] for i in range(4)]
        identity = [[int(i == j) for j in range(4)] for i in range(4)]
        zero = [[0] * 4 for _ in range(4)]
        self.assert_products(a, identity, a)
        self.assert_products(identity, a, a)
        self.assert_products(a, zero, zero)

    def test_all_basis_pairs(self):
        """Check every coefficient of each complete bilinear base identity."""
        # Matrix multiplication is bilinear. Testing every E_ij * E_kl checks
        # every coefficient of the full 2x2 and 4x4 base identities.
        for name, implementation in available_algorithms().items():
            if name == "smirnov":
                # Its rectangular identities are checked exhaustively by the
                # independent certificate checker in test_smirnov, not 2x2 units.
                continue
            size = 4 if name == "alphaevolve" else 2
            for left in range(size * size):
                i, j = divmod(left, size)
                a = [[0] * size for _ in range(size)]
                a[i][j] = 1
                for right in range(size * size):
                    k, l = divmod(right, size)
                    b = [[0] * size for _ in range(size)]
                    b[k][l] = 1
                    expected = [[0] * size for _ in range(size)]
                    expected[i][l] = int(j == k)
                    with self.subTest(algorithm=name, left=left, right=right):
                        self.assertEqual(implementation(a, b), expected)

    def test_random_rectangular_odd_and_skinny(self):
        """Exercise padding and leaf choices on reproducible irregular shapes."""
        rng = random.Random(107)
        for rows, inner, columns in [(1, 4, 3), (3, 1, 4), (4, 3, 1), (3, 3, 3),
                                      (3, 5, 2), (5, 2, 3), (4, 4, 4), (7, 3, 5)]:
            a = [[rng.randint(-20, 20) for _ in range(inner)] for _ in range(rows)]
            b = [[rng.randint(-20, 20) for _ in range(columns)] for _ in range(inner)]
            self.assert_products(a, b, cutoffs=(1, 3, 64))

    @requires_coefficients("alphaevolve")
    def test_two_alphaevolve_recursion_levels(self):
        """Verify composition of the rank-48 base scheme, beyond a single level."""
        rng = random.Random(48)
        a = [[rng.randint(-3, 3) for _ in range(16)] for _ in range(16)]
        b = [[rng.randint(-3, 3) for _ in range(16)] for _ in range(16)]
        # This case targets the two-level rank-48 composition specifically.
        # Smirnov's separate bounded tests exercise its own rectangular stages.
        self.assertTrue(check_product(a, b, alphaevolve.multiply(a, b)).passed)

    def test_exact_large_integers(self):
        """Ensure complex intermediates never round large exact real inputs."""
        large = 10 ** 80 + 7
        self.assert_products([[large, -large], [2, 3]], [[large, 1], [large, -1]])
        if "alphaevolve" in available_algorithms():
            self.assertTrue(all(type(x) is int for row in alphaevolve.multiply([[large]], [[large]]) for x in row))

    def test_exact_rationals(self):
        """Preserve Fraction inputs and verify their results by exact equality."""
        a = [[Fraction(1, 3), Fraction(-2, 7)], [Fraction(5, 11), Fraction(4, 9)]]
        b = [[Fraction(7, 13), Fraction(2, 5)], [Fraction(3, 8), Fraction(-1, 17)]]
        self.assert_products(a, b)
        if "alphaevolve" in available_algorithms():
            self.assertTrue(check_product(a, b, alphaevolve.multiply(a, b)).exact)

    def test_floating_and_complex(self):
        """Exercise floating domains with recursion and classical leaf cutoffs."""
        rng = random.Random(2026)
        for complex_values in (False, True):
            a = [[rng.uniform(-3, 3) for _ in range(4)] for _ in range(4)]
            b = [[rng.uniform(-3, 3) for _ in range(4)] for _ in range(4)]
            if complex_values:
                a = [[complex(x, rng.uniform(-3, 3)) for x in row] for row in a]
                b = [[complex(x, rng.uniform(-3, 3)) for x in row] for row in b]
            self.assert_products(a, b, cutoffs=(1, 4))

    def test_complex_basis_has_correct_signs(self):
        """Guard complex signs with a small explicitly interpretable example."""
        a = [[1j, 2], [-3j, 4j]]
        b = [[2j, 3j], [4, -1j]]
        self.assert_products(a, b, [[6, -3 - 2j], [6 + 16j, 13]])

    def test_floating_recursion_with_padding(self):
        """Check floating error tolerances through padded rectangular recursion."""
        rng = random.Random(225)
        a = [[rng.uniform(-1, 1) for _ in range(7)] for _ in range(5)]
        b = [[complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(3)] for _ in range(7)]
        self.assert_products(a, b)

    def test_transpose_product_identity(self):
        """Check that reversing transposed inputs gives the transposed product."""
        a, b = [[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]
        def transpose(matrix):
            """Build the independent transpose used by this algebraic property."""
            return [list(row) for row in zip(*matrix)]
        for name, implementation in available_algorithms().items():
            with self.subTest(algorithm=name):
                self.assertEqual(transpose(implementation(a, b)), implementation(transpose(b), transpose(a)))

    def test_tuple_inputs_and_fresh_output(self):
        """Accept immutable input rows and return independent mutable output rows."""
        a, b = ((1, 2), (3, 4)), ((1, 0), (0, 1))
        self.assert_products(a, b, [[1, 2], [3, 4]])
        for implementation in available_algorithms().values():
            result = implementation(a, b)
            result[0][0] = 999
            self.assertEqual(implementation(a, b)[0][0], 1)

    def test_invalid_inputs_consistently_rejected(self):
        """Enforce a consistent public input and cutoff contract for all algorithms."""
        invalid = [([], [[1]], ValueError), ([[]], [[1]], ValueError),
                   ([[1, 2], [3]], [[1], [2]], ValueError),
                   ([[1, 2]], [[3, 4]], ValueError),
                   ([["1"]], [[1]], TypeError), ([[True]], [[1]], TypeError),
                   ([[float("nan")]], [[1]], ValueError),
                   ([[float("inf")]], [[1]], ValueError),
                   ([[complex(0, float("inf"))]], [[1]], ValueError)]
        for name, implementation in ALGORITHMS.items():
            for a, b, error in invalid:
                with self.subTest(algorithm=name, a=a):
                    with self.assertRaises(error):
                        implementation(a, b)
            for cutoff in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    implementation([[1]], [[2]], cutoff=cutoff)

    def test_dispatch(self):
        """Route every registered algorithm and reject unsupported names."""
        for name in available_algorithms():
            self.assertEqual(multiply([[2]], [[3]], algorithm=name), [[6]])
        with self.assertRaisesRegex(ValueError, "unknown algorithm"):
            multiply([[1]], [[2]], algorithm="openai-2.25")


class CoefficientTests(unittest.TestCase):
    """Guard full coefficient verification and actionable optional-data failures."""
    @requires_coefficients("alphaevolve")
    def test_all_tensor_coefficients(self):
        """Check the full exact identity and retain a hash for data provenance."""
        data = json.loads(alphaevolve.DATA_PATH.read_text())
        alphaevolve.validate_factors(data["factors_doubled"])
        self.assertEqual(len(data["source_sha256"]), 64)

    @requires_coefficients("alphaevolve")
    def test_corrupted_decomposition_rejected(self):
        """Ensure a changed coefficient cannot pass whole-identity verification."""
        data = json.loads(alphaevolve.DATA_PATH.read_text())
        factors = data["factors_doubled"]
        factors[0][0][0][0] += 2
        with self.assertRaisesRegex(ValueError, "incorrect tensor coefficient"):
            alphaevolve.validate_factors(factors)

    def test_missing_data_has_actionable_error(self):
        """Clear caches and confirm a missing source has an actionable diagnostic."""
        from unittest.mock import patch
        alphaevolve._factors.cache_clear()
        try:
            with patch.object(alphaevolve, "DATA_PATH", Path("does-not-exist-rank48.json")):
                with self.assertRaisesRegex(FileNotFoundError, "import_alphaevolve"):
                    alphaevolve.multiply([[1]], [[2]])
        finally:
            alphaevolve._factors.cache_clear()


class VerificationTests(unittest.TestCase):
    """Check that the independent oracle rejects inaccuracies and invalid output."""
    def test_false_results_are_detected(self):
        """Reject wrong values, shapes, and nonfinite or nonnumeric output."""
        self.assertFalse(check_product([[2]], [[3]], [[7]]).passed)
        self.assertFalse(check_product([[2]], [[3]], []).passed)
        self.assertFalse(check_product([[2]], [[3]], [[6, 0]]).passed)
        self.assertFalse(check_product([[2.0]], [[3.0]], [[float("nan")]]).passed)
        self.assertFalse(check_product([[1]], [[1]], [["1"]]).passed)

    def test_integer_checks_have_no_tolerance(self):
        """An integer error of one remains a failure even on enormous results."""
        huge = 10 ** 100
        self.assertFalse(check_product([[huge]], [[1]], [[huge + 1]], rtol=1).passed)

    def test_tolerance_is_applied(self):
        """Distinguish small floating deviations and reject invalid tolerances."""
        self.assertTrue(check_product([[2.0]], [[3.0]], [[6.000000001]]).passed)
        self.assertFalse(check_product([[2.0]], [[3.0]], [[6.001]]).passed)
        with self.assertRaises(ValueError):
            check_product([[1]], [[1]], [[1]], atol=-1)

    def test_cancellation_is_detected_by_exact_reference(self):
        """The oracle must detect cancellation even when classical arithmetic errs."""
        # Even classical floating-point multiplication can lose a small term.
        # Comparing only against naive would silently accept the same error.
        a, b = [[1e16, 1.0, -1e16]], [[1.0], [1.0], [1.0]]
        self.assertFalse(check_product(a, b, ALGORITHMS["naive"](a, b)).passed)

    def test_complex_tolerance(self):
        """Apply a norm-based tolerance to real and imaginary output errors."""
        self.assertTrue(check_product([[1j]], [[1]], [[1j + 1e-10]]).passed)
        self.assertFalse(check_product([[1j]], [[1]], [[1j + 1e-3]]).passed)

    def test_large_float_norm_does_not_overflow_comparison(self):
        """Compare large complex norms exactly without overflowing diagnostics."""
        a, b = [[complex(1e308, 1e308)]], [[1.0]]
        self.assertTrue(check_product(a, b, a).passed)
        self.assertFalse(check_product(a, b, [[complex(0, 0)]]).passed)


class ImportTests(unittest.TestCase):
    """Ensure coefficient ingestion treats upstream notebooks exclusively as data."""
    @requires_coefficients("alphaevolve")
    def test_import_notebook_without_executing_it(self):
        """Reextract literal notebook data and match the validated local factors."""
        from tools.import_alphaevolve import NOTEBOOK, extract
        if not NOTEBOOK.is_file():
            self.skipTest("optional source notebook missing; run python tools/import_alphaevolve.py --download")
        notebook = json.loads(NOTEBOOK.read_text())
        factors = extract(notebook)
        data = json.loads(alphaevolve.DATA_PATH.read_text())
        self.assertEqual(factors, data["factors_doubled"])

    def test_executable_array_content_is_rejected(self):
        """Never execute a function embedded in a coefficient expression."""
        from tools.import_alphaevolve import extract
        notebook = {"cells": [{"cell_type": "code", "source": [
            "decomposition_444 = (np.array(dangerous()), np.array([]), np.array([]))"
        ]}]}
        with self.assertRaises(ValueError):
            extract(notebook)

    def test_missing_assignment_is_rejected(self):
        """Require exactly the expected decomposition assignment in source data."""
        from tools.import_alphaevolve import extract
        with self.assertRaises(ValueError):
            extract({"cells": []})


if __name__ == "__main__":
    unittest.main()
