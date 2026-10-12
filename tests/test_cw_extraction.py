"""Check CW extraction with complete identities and adversarial group selections.

These tests protect the distinction between disjoint groups and an induced
direct sum. They also exercise every small sector's row-major permutation,
including unit-size matrix axes, rather than relying on one numerical example.
Portable scheme checks and actual primitive counters validate the bridge from a
balanced component to the existing recursive multiplier.
"""

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from itertools import product
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matrix_multiplication import Instrumentation, certificate_checker
from matrix_multiplication.composition import Composition
from matrix_multiplication.constructions import ConstructionRequest, main, run_experiment
from matrix_multiplication.cw_extraction import (
    SECTORS, choose_independent_words, component, extract_cw,
    sector_word, selection_conflicts)
from matrix_multiplication.degenerations import polynomial_degree_bound, recover_cw
from matrix_multiplication.scheme_io import load_scheme_document, scheme_from_document
from matrix_multiplication.tensors import (
    Tensor, classical, coppersmith_winograd, restrict_coordinates, tensor_power)
from matrix_multiplication.verification import check_product
from matrix_multiplication.tensor_execution import evaluate_tensor


class CWExtractionTests(unittest.TestCase):
    """Require exact algebra, observable execution, and truthful finite costs."""

    def test_every_small_sector_has_the_claimed_matrix_identity(self):
        """Check all 258 words across powers 1/2/3, not just balanced examples."""
        for q, power in ((3, 1), (3, 2), (2, 3)):
            source = tensor_power(coppersmith_winograd(q), power)
            for word in product(SECTORS, repeat=power):
                with self.subTest(q=q, word=word):
                    item = component(q, word)
                    actual = restrict_coordinates(source, item.selections)
                    self.assertEqual(actual.expand(), classical(item.shape).expand())
                    self.assertEqual(len(set(item.selections[0])), len(item.selections[0]))

    def test_disjoint_groups_can_still_leave_a_cross_sector(self):
        """Reject a three-component matching whose retained groups induce a fourth."""
        words = (((0, 0, 2), (0, 0, 2)),
                 ((0, 0, 2), (1, 1, 0)),
                 ((0, 1, 1), (2, 0, 0)))
        conflicts = selection_conflicts(words)
        self.assertEqual(conflicts["shared_legs"], [])
        self.assertEqual(conflicts["unexpected_words"], [((0, 0, 2), (2, 0, 0))])
        # Fail before allocating the polynomial family, not after discovering
        # that an execution happens to agree on an uninformative input pair.
        with patch("matrix_multiplication.cw_extraction.cw_family") as family:
            with self.assertRaisesRegex(ValueError, "independent induced sum"):
                extract_cw(2, 2, words)
            family.assert_not_called()

    def test_shared_groups_and_duplicate_components_are_rejected(self):
        """A component may not borrow an input/output block from another product."""
        for words in ((((0, 1, 1),), ((1, 0, 1),)),
                      (((0, 1, 1),), ((0, 1, 1),))):
            with self.assertRaises(ValueError):
                extract_cw(2, 1, words)

    def test_greedy_selections_recover_and_execute_independent_products(self):
        """Exercise q=1/2 and powers 1/2 with full paid recovery and exact outputs."""
        for q, power in product((1, 2), repeat=2):
            tensor, evidence = extract_cw(q, power)
            chosen = choose_independent_words(q, power)
            self.assertEqual(chosen, choose_independent_words(q, power))
            self.assertEqual(selection_conflicts(chosen), {"shared_legs": [], "unexpected_words": []})
            self.assertTrue(evidence["literal_direct_sum_identity_verified"])
            self.assertTrue(evidence["recovered_direct_sum_identity_verified"])
            self.assertTrue(evidence["execution"]["exact_outputs_verified"])
            self.assertEqual(evidence["execution"]["observation"]["counts"]["field_multiplications"], tensor.rank)
            self.assertLessEqual(tensor.rank, evidence["finite_rank_bound"])
            self.assertFalse(evidence["dense_matrix_packing_implemented"])
            self.assertIsNone(evidence["exponent"])
        self.assertEqual(evidence["selected_components"], 4)
        self.assertEqual(evidence["finite_recipe_products"], 176)
        self.assertEqual(evidence["classical_direct_sum_products"], 10)
        self.assertEqual(evidence["original_degree_bound"], 20)
        self.assertEqual(evidence["degree_bound"], 11)

    def test_restricting_before_recovery_preserves_the_recovered_coefficients(self):
        """Compare two construction orders to catch accidental polynomial deletion."""
        extracted, evidence = extract_cw(2, 2)
        recovered, _ = recover_cw(2, 2)
        restricted = restrict_coordinates(recovered, evidence["retained_source_coordinates"])
        self.assertEqual(extracted.expand(), restricted.expand())
        self.assertLess(extracted.rank, restricted.rank)

    def test_degree_bound_counts_nonzero_coefficients_and_validates_exactness(self):
        """Trailing zeros and cancellation do not justify an unsafe degree guess."""
        term = (((0, (1, 0, 0)),), ((0, (0, 1, 0)),), ((0, (0, 0, 2)),))
        self.assertEqual(polynomial_degree_bound((term,)), 3)
        # Equal and opposite terms cancel globally, but the factor bound remains
        # three. That deliberate conservatism keeps interpolation exact.
        opposite = (term[0], term[1], ((0, (0, 0, -2)),))
        self.assertEqual(polynomial_degree_bound((term, opposite)), 3)
        with self.assertRaises(ValueError):
            polynomial_degree_bound(((((0, (1.0,)),),)*3,))
        with self.assertRaises(ValueError):
            polynomial_degree_bound(((((True, (1,)),),)*3,))

    def test_tensor_kernel_counts_cancellations_and_coefficient_work(self):
        """Input zeros cannot hide products; fixed scaling is counted separately."""
        tensor = Tensor((1, 1, 1), (
            (((0, 1),), ((0, 1),), ((0, Fraction(1, 2)),)),
            (((0, 1),), ((0, 1),), ((0, Fraction(-1, 2)),))))
        observer = Instrumentation(profile=False)
        self.assertEqual(evaluate_tensor(tensor, [0], [3], instrumentation=observer), (0,))
        self.assertEqual(observer.counts["field_multiplications"], 2)
        self.assertEqual(observer.counts["coefficient_multiplications"], 6)
        self.assertEqual(observer.counts["field_additions"], 6)
        with self.assertRaises(ValueError):
            evaluate_tensor(tensor, [0.0], [3])
        with self.assertRaises(ValueError):
            evaluate_tensor(tensor, [1, 2], [3])

    def test_balanced_component_exports_and_executes_a_verified_square_recipe(self):
        """Bridge tensor extraction to the existing checker and measured engine.

        This recovered recipe is expensive (324 products versus classical eight).
        The test establishes an executable identity, not an improved exponent.
        Recursing it on a 4-by-4 input would need 324**2 products, so one base
        call is sufficient here; recursive traversal has its own existing tests.
        """
        word = (((1, 0, 1), (1, 1, 0), (0, 1, 1)),)
        with tempfile.TemporaryDirectory() as temp:
            folder, report = run_experiment(ConstructionRequest("cw-extract", q=2, power=3,
                                                                sector_words=word, certificate=True),
                                            Path(temp) / "balanced")
            document = load_scheme_document(folder / "scheme.json")
            scheme = scheme_from_document(document)
            certificate_checker.check_recipe(document)
            self.assertEqual(scheme.shape, (2, 2, 2))
            self.assertEqual(scheme.rank, 324)
            self.assertFalse(report["square_recursion"]["strictly_below_2_75"])
            self.assertTrue((folder / "certificate" / "certificate.json").is_file())
            plan = Composition.from_spec(str(folder / "scheme.json"))
            a, b = [[1, -2], [3, 4]], [[5, 6], [-7, 8]]
            observer = Instrumentation(profile=False)
            result = plan.multiply(a, b, instrumentation=observer)
            self.assertTrue(check_product(a, b, result).passed)
            self.assertEqual(observer.counts["field_multiplications"], 324)
            self.assertIn("Verified independent matrix products", (folder / "index.html").read_text())

    def test_invalid_selection_parameters_fail_before_output_creation(self):
        """Reject wrong powers, empty selections, and unsupported certificates."""
        for q, power in ((True, 1), (2, 0), (2, 1.0)):
            with self.assertRaises(ValueError):
                extract_cw(q, power)
        for words in ((), (((0, 1, 1), (1, 0, 1)),)):
            with self.assertRaises(ValueError):
                extract_cw(2, 1, words)
        with self.assertRaises(ValueError):
            sector_word(((False, 1, 1),))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "unsupported"
            with self.assertRaisesRegex(ValueError, "single supported"):
                run_experiment(ConstructionRequest("cw-extract", certificate=True), folder)
            self.assertFalse(folder.exists())

    def test_cli_runs_extraction_and_rejects_invalid_sector_text(self):
        """The focused command shares the suite's implementation and output format."""
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--output-root", temp, "cw-extract", "--q", "2", "--power", "2"]), 0)
            folder = next(Path(temp).iterdir())
            report = json.loads((folder / "report.json").read_text())
            self.assertEqual(report["cw_extraction"]["selected_components"], 4)
            self.assertTrue((folder / "tensor.json").is_file())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["cw-extract", "--sector-word", "101,111,011"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
