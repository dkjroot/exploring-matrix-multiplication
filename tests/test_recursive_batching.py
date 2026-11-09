"""Check both the archived CW control and a recipe with mixed-lane factors."""

import json
from pathlib import Path
import tempfile
import unittest

from matrix_multiplication import Instrumentation
from matrix_multiplication.constructions import ConstructionRequest, run_experiment
from matrix_multiplication.cw_batching import rational_inputs
from matrix_multiplication.pan_batching import correction_evidence, pan_pair, run_pan_batch
from matrix_multiplication.recipe_parts import load_parts
from matrix_multiplication.recursive_batching import RecursiveBatch, verify_batch_tensor
from matrix_multiplication.tensors import Tensor, classical, direct_sum
from matrix_multiplication.verification import check_product


class RecursiveBatchingTests(unittest.TestCase):
    """Protect exact routing, identity admission, arithmetic and report export."""

    def test_cw_control_reports_exact_executions_and_reloadable_parts(self):
        """Exercise the public runner with the real archived CW pair."""
        with tempfile.TemporaryDirectory() as temp:
            directory, report = run_experiment(ConstructionRequest("cw-batch"), Path(temp) / "run")
            batch = report["cw_batching"]
            self.assertEqual(batch["parent_terms_by_child_slot"], tuple((i, i+1) for i in range(0, 14, 2)))
            self.assertEqual(batch["second_level_identity"]["coefficient_checks"], 32768)
            self.assertEqual([e["budget"]["field_multiplications"] for e in batch["executions"]], [14, 98, 686])
            self.assertEqual([e["budget"]["field_additions"] for e in batch["executions"]], [80, 880, 7440])
            for depth, item in enumerate(batch["executions"], start=1):
                self.assertEqual(item["exact_outputs_verified"], [True, True])
                self.assertEqual(item["observation"]["products_by_depth"], {depth: item["budget"]["field_multiplications"]})
            self.assertEqual(json.loads((directory / "report.json").read_text())["command"], "cw-batch")
            self.assertIn("terms 6 and 7", (directory / "index.html").read_text())
            parts, _ = load_parts(directory / "parts.json")
            self.assertEqual([tensor.rank for tensor, _ in parts], [14, 98])

    def test_cross_lane_factors_cancel_and_invalid_schedules_fail(self):
        """Mixed intermediate terms remain legal when the complete target has no cross terms."""
        base = direct_sum(classical((2,)*3), classical((2,)*3))
        mixed = ((0, 1), (4, 1))
        positive = (mixed, mixed, mixed)
        negative = (mixed, mixed, ((0, -1), (4, -1)))
        recipe = Tensor(base.dimensions, base.terms+(positive, negative))
        plan = RecursiveBatch(recipe, 2, 2)
        verify_batch_tensor(plan.second_level_tensor(), 4, 2)
        leaf_observer = Instrumentation(profile=False)
        plan.multiply(*rational_inputs(1, 2), instrumentation=leaf_observer)
        self.assertEqual(leaf_observer.counts["matrix_cells_allocated"], 2)
        left, right = rational_inputs(4, 2)
        observer = Instrumentation(profile=False)
        actual = plan.multiply(left, right, instrumentation=observer)
        self.assertTrue(all(check_product(a, b, c).passed for a, b, c in zip(left, right, actual)))
        self.assertEqual(observer.counts["field_multiplications"], 162)
        with self.assertRaisesRegex(ValueError, "complete independent square-product identity"):
            RecursiveBatch(Tensor(base.dimensions, base.terms+(positive, positive)), 2, 2)
        with self.assertRaisesRegex(ValueError, "evenly into child calls"):
            RecursiveBatch(Tensor(base.dimensions, base.terms+(positive,)), 2, 2)
        with self.assertRaisesRegex(ValueError, "power of the radix"):
            plan.multiply(*rational_inputs(3, 2))

    def test_pan_corrections_are_required(self):
        """The uncorrected aggregate and a missing correction both leave cross terms."""
        recipe = pan_pair(2)
        self.assertEqual(recipe.rank, 20)
        self.assertEqual(correction_evidence(recipe, 2)["cancelled_mixed_coefficients"], 48)
        for terms in (recipe.terms[:8], recipe.terms[:-1]):
            incomplete = Tensor(recipe.dimensions, terms)
            with self.assertRaisesRegex(ValueError, "complete independent square-product identity"):
                verify_batch_tensor(incomplete, 2, 2)

    def test_pan_real_mixed_batch_matches_full_identity_and_all_budgets(self):
        """Exercise the independently generated size-four recipe through two levels."""
        parts, report = run_pan_batch()
        self.assertEqual([tensor.rank for tensor, _ in parts], [112, 6272])
        self.assertEqual(report["base_identity"]["coefficient_checks"], 32768)
        self.assertEqual(report["second_level_identity"]["coefficient_checks"], 134217728)
        self.assertEqual(report["corrections"]["cancelled_mixed_coefficients"], 384)
        for execution, counts in zip(report["executions"], ((112, 864, 98, 128), (6272, 62208, 4802, 8192))):
            data, additions, strassen, classical = counts
            self.assertEqual(execution["exact_outputs_verified"], [True, True])
            self.assertEqual(execution["budget"], {"field_multiplications": data,
                                                   "field_additions": additions,
                                                   "coefficient_multiplications": additions})
            self.assertEqual(execution["separate_strassen"]["observation"]["counts"]["field_multiplications"], strassen)
            self.assertEqual(execution["separate_classical"]["observation"]["counts"]["field_multiplications"], classical)


if __name__ == "__main__":
    unittest.main()
