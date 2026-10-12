"""Protect finite CW recovery and dense packing with independent exact evidence.

Coverage tests deliberately introduce holes, duplicate inner contributions, and
out-of-range tiles. Algebra tests compare full tensors, replay both reduction
chains, run recursive exported schemes, and compare observed primitive counts
with the execution engine's separately derived budget. No numerical example is
used in place of a complete matrix multiplication identity.
"""

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
import unittest

from matrix_multiplication import Instrumentation, certificate_checker
from matrix_multiplication.certificates import formula_recipe
from matrix_multiplication.composition import Composition
from matrix_multiplication.constructions import ConstructionRequest, main, run_experiment
from matrix_multiplication.cw_extraction import optimize_cw_extraction
from matrix_multiplication.decomposition_search import replay
from matrix_multiplication.degenerations import cw_family
from matrix_multiplication.dense_packing import (
    Tile, batch_maps, pack_cw, pack_tensor, packing_schedule, verify_schedule)
from matrix_multiplication.cw_extraction import restrict_polynomial
from matrix_multiplication.scheme_io import scheme_from_document, scheme_document
from matrix_multiplication.tensors import (
    Tensor, classical, coefficient_expansion, direct_sum, from_scheme, linear_maps,
    polynomial_coefficient, to_scheme)
from matrix_multiplication.verification import check_product


class DensePackingTests(unittest.TestCase):
    """Require executable dense identities and explicitly accounted finite costs."""

    def test_coefficient_expansion_matches_independent_interpolation(self):
        """Exercise all coefficients, duplicate coordinates, and exact cancellation."""
        terms = ((((0, (1, 2)), (0, (-1, 0)), (1, (3,))),
                  ((0, (3, 4)),), ((0, (5, 6)),)),)
        for coefficient in range(4):
            expanded = coefficient_expansion((2, 1, 1), terms, coefficient)
            interpolated = polynomial_coefficient((2, 1, 1), terms, 3, coefficient)
            self.assertEqual(expanded.expand(), interpolated.expand())
        self.assertEqual(coefficient_expansion((2, 1, 1), terms, 4).rank, 0)
        # Validate even unused coordinates and high-degree coefficients; filtering
        # by the requested degree must not mask malformed source input.
        for terms in (((((2, (0,)),), ((0, (1,)),), ((0, (1,)),)),),
                      (((((0, (1, 0.0)),),)*3),)):
            with self.assertRaises(ValueError):
                coefficient_expansion((1, 1, 1), terms, 0)
        with self.assertRaises(ValueError):
            coefficient_expansion((1, 1, 1), (), True)

    def test_recovery_reductions_are_reproducible_without_a_classical_replacement(self):
        """Regenerate the coefficient planes, replay logs, and check every entry."""
        source, evidence = optimize_cw_extraction(2, 2)
        recovery = evidence["optimized_recovery"]
        self.assertEqual(recovery["interpolation_products"], 176)
        self.assertEqual(recovery["coefficient_expansion_products"], 42)
        self.assertEqual(recovery["selected_products"], 10)
        _, terms, _, leading = cw_family(2, 2)
        expanded = coefficient_expansion(source.dimensions, restrict_polynomial(
            terms, evidence["retained_source_coordinates"]), leading)
        self.assertEqual(replay(expanded, recovery["reduction_events"]), source)
        self.assertTrue(recovery["complete_identity_verified"])
        self.assertEqual(recovery["execution"]["observation"]["counts"]["field_multiplications"], 10)
        self.assertFalse(recovery["minimum_rank_claim"])

    def test_mixed_partial_slots_cover_rectangular_dense_targets(self):
        """Partial inner slots and overlapping output entries still form exact products."""
        shapes = ((2, 1, 2), (2, 2, 1), (1, 1, 1), (1, 1, 1))
        source = classical(shapes[0])
        for shape in shapes[1:]:
            source = direct_sum(source, classical(shape))
        for target in ((2, 2, 2), (3, 3, 3), (3, 2, 4), (1, 5, 2)):
            schedule = packing_schedule(shapes, target)
            self.assertTrue(verify_schedule(shapes, target, schedule))
            packed, evidence = pack_tensor(source, shapes, target, batches=schedule)
            self.assertEqual(packed.expand(), classical(target).expand())
            self.assertLessEqual(packed.rank, evidence["batch_product_bound"])
            self.assertEqual(evidence["selection_method"], "explicit supplied schedule")
        # For size two the four mixed slots cover all eight triples in one batch,
        # including zero-padding of a component that has a longer inner axis.
        self.assertEqual(len(packing_schedule(shapes, (2, 2, 2))), 1)

    def test_bad_schedules_fail_independently_of_the_planner(self):
        """Detect double counting, missing summands, bounds, and undeclared slots."""
        shapes, target = ((1, 1, 1),), (1, 2, 1)
        origin = Tile((0, 0, 0), (1, 1, 1))
        second = Tile((0, 1, 0), (1, 1, 1))
        self.assertTrue(verify_schedule(shapes, target, ((origin,), (second,))))
        for schedule in (((origin,),), ((origin,), (origin,)), ((None,),),
                         ((origin, second),), ((Tile((0, 2, 0), (1, 1, 1)),),),
                         ((Tile((0, 0, 0), (1, 2, 1)),),), (("not a tile",),)):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                verify_schedule(shapes, target, schedule)
        with self.assertRaises(ValueError):
            Tile((False, 0, 0), (1, 1, 1))
        with self.assertRaises(ValueError):
            Tile((0, 0, 0), (1, 0, 1))

    def test_packing_rejects_a_false_source_identity(self):
        """Matching vector-space dimensions alone cannot establish independence."""
        source = classical((2, 2, 2))
        corrupt = Tensor(source.dimensions, source.terms[:-1])
        with self.assertRaisesRegex(ValueError, "independent matrix-product sum"):
            pack_tensor(corrupt, ((2, 2, 2),), (2, 2, 2))

    def test_packing_accepts_a_genuinely_fast_source_recipe(self):
        """Keep Strassen's seven products when its one slot covers the whole target."""
        source = from_scheme(scheme_from_document(formula_recipe("strassen")))
        packed, evidence = pack_tensor(source, ((2, 2, 2),), (2, 2, 2))
        self.assertEqual(packed.rank, 7)
        self.assertTrue(evidence["dense_matrix_identity_verified"])
        certificate_checker.check_recipe(scheme_document(to_scheme(packed, (2, 2, 2))))

    def test_cw_packing_replays_the_mapped_recipe_and_counts_actual_dense_work(self):
        """A power-two packing reduces ten mapped products to eight, exactly."""
        packed, evidence = pack_cw(2, 2, 2)
        self.assertEqual(evidence["batch_count"], 1)
        self.assertEqual(evidence["source_products_per_batch"], 10)
        self.assertEqual(evidence["mapped_products_before_reductions"], 10)
        self.assertEqual(evidence["packed_products"], 8)
        self.assertEqual(evidence["execution"]["observation"]["counts"]["field_multiplications"], 8)
        source, _ = optimize_cw_extraction(2, 2)
        shapes = tuple(tuple(shape) for shape in evidence["component_shapes"])
        batches = tuple(tuple(Tile(slot["origin"], slot["extent"]) if slot else None for slot in batch)
                        for batch in evidence["batches"])
        mapped = linear_maps(source, packed.dimensions, batch_maps(shapes, (2, 2, 2), batches[0]))
        self.assertEqual(replay(mapped, evidence["packing_reduction_events"]), packed)
        self.assertFalse(evidence["strictly_fewer_products_than_classical"])

    def test_exported_packing_recurses_with_matching_arithmetic_predictions(self):
        """Exercise real growth, partial sizes, cutoffs, floats, and certificates."""
        with tempfile.TemporaryDirectory() as temp:
            folder, report = run_experiment(ConstructionRequest("cw-pack", q=2, power=2, certificate=True),
                                            Path(temp) / "packing")
            plan = Composition.from_spec(str(folder / "scheme.json"))
            self.assertEqual(report["square_recursion"]["exponent"], 3)
            self.assertTrue((folder / "certificate" / "certificate.json").is_file())
            for size, expected in ((2, 8), (4, 64), (8, 512)):
                a = [[Fraction(i-2*j, 3) for j in range(size)] for i in range(size)]
                b = [[Fraction(2*i+j-1, 5) for j in range(size)] for i in range(size)]
                observer = Instrumentation(profile=False)
                result = plan.multiply(a, b, instrumentation=observer)
                self.assertTrue(check_product(a, b, result).passed)
                self.assertEqual(observer.counts["field_multiplications"], expected)
                self.assertEqual(observer.snapshot()["field_arithmetic_operations"],
                                 plan.budget(size, size, size)["field_arithmetic_operations"])
            for cutoff in (1, 2, 4):
                a, b = [[0.1, -0.2, 0.3]], [[0.4], [0.5], [-0.6]]
                self.assertTrue(check_product(a, b, plan.multiply(a, b, cutoff=cutoff)).passed)
            self.assertIn("Exact recovery and dense packing", (folder / "index.html").read_text())

    def test_cli_and_preflight_controls(self):
        """Create the same verified artifacts from the focused CLI without a sweep."""
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--output-root", temp, "cw-pack", "--size", "2", "--certificate"]), 0)
            report = json.loads((next(Path(temp).iterdir()) / "report.json").read_text())
            self.assertEqual(report["rank"], 8)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["cw-pack", "--size", "1"])


if __name__ == "__main__":
    unittest.main()
