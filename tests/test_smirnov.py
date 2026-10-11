"""Bounded exact tests for the lower-exponent Smirnov construction.

No default sweep runs here. Coefficients are optional research data; a fresh
checkout skips data-dependent tests until the documented importer is run.
Three-stage full-size execution is reserved for the explicit smoke-test tool.
"""

from fractions import Fraction
import json
from pathlib import Path
import unittest

from matrix_multiplication import Instrumentation, multiply
from matrix_multiplication import smirnov
from matrix_multiplication.certificate_checker import check_recipe, check_certificate
from matrix_multiplication.certificates import create_certificate
from matrix_multiplication.rectangular import RectangularCycle, cyclic_rotation
from matrix_multiplication.scheme_io import load_scheme
from matrix_multiplication.verification import check_product
from matrix_multiplication.work_sweep import collect_work


@unittest.skipUnless(smirnov.DATA_PATH.exists(), "import optional Smirnov data first")
class SmirnovTests(unittest.TestCase):
    """Check required rectangular machinery, exact identities, and small executions."""

    def test_all_cyclic_identities_and_square_certificate(self):
        """Check all 2916 relationships per stage and a square rank-64000 cycle."""
        p = smirnov.plan()
        self.assertEqual(p.shapes, ((3, 3, 6), (3, 6, 3), (6, 3, 3)))
        for document in p.documents:
            self.assertEqual(check_recipe(document)["coefficient_checks"], 2916)
        verified = check_certificate(create_certificate(p))
        self.assertEqual((verified["cycle"]["radix"], verified["cycle"]["rank"]), (54, 64000))
        self.assertLess(p.model()["square_arithmetic_exponent"], 2.7925)

    def test_small_exact_outputs_and_work_budgets(self):
        """Execute one and two rectangular stages with irregular inputs and cutoffs."""
        p = smirnov.plan()
        for rows, inner, columns, cutoff in ((2, 2, 2, 1), (4, 2, 3, 1), (2, 3, 4, 6)):
            a = [[Fraction(i-j, 3) for j in range(inner)] for i in range(rows)]
            b = [[i+j+1 for j in range(columns)] for i in range(inner)]
            observer = Instrumentation(profile=False)
            result = multiply(a, b, algorithm="smirnov", cutoff=cutoff, instrumentation=observer)
            self.assertTrue(check_product(a, b, result).passed)
            budget = p.budget(rows, inner, columns, cutoff=cutoff)
            snapshot = observer.snapshot()
            self.assertEqual(observer.counts["field_multiplications"], budget["field_multiplications"])
            for key in ("field_arithmetic_operations", "atomic_arithmetic_operations"):
                self.assertEqual(snapshot[key], budget[key])

    def test_rotation_roundtrip_and_unbalanced_cycle_rejected(self):
        """Three rotations recover the same tensor; an unbalanced cycle has no square claim."""
        base = load_scheme(smirnov.DATA_PATH)
        rotated = cyclic_rotation(cyclic_rotation(cyclic_rotation(base)))
        for factor in ("left", "right", "output"):
            self.assertEqual([dict(c) for c in getattr(base, factor)],
                             [dict(c) for c in getattr(rotated, factor)])
        with self.assertRaises(ValueError):
            RectangularCycle([base])

    def test_explicit_small_comparison_and_full_cycle_budget(self):
        """Capture small growth evidence without invoking the edited sweep CLI."""
        p = smirnov.plan()
        data = collect_work([1, 2], [1], algorithms=["naive", "smirnov"])
        for line in data["series"]:
            for sample in line["samples"]:
                self.assertEqual(sample["status"], "passed")
                self.assertTrue(sample["prediction_matches_observed_arithmetic_steps"])
        budget = p.budget(54, 54, 54)
        self.assertEqual(budget["field_multiplications"], 64000)
        self.assertEqual(budget["leaf_dimensions"], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
