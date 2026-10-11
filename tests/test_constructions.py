"""Verify constructive algebra, discovered recipes, and evidence interoperability.

Small complete tensors catch index-convention errors and exact cancellation.
The discovery fixture starts from classical multiplication, not a known rank-7
recipe. We check the found scheme through a separate basis checker, actual
recursive execution, independent products, and measured-versus-budget counts.
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
from matrix_multiplication.constructions import main, target_rank
from matrix_multiplication.decomposition_search import (
    beats_exponent, dependence, flip, flip_choices, reduce_all, reduce_relation, replay, search, split)
from matrix_multiplication.degenerations import cw_family, recover_cw
from matrix_multiplication.scheme_io import scheme_document, scheme_from_document, save_scheme, load_scheme
from matrix_multiplication.tensors import (
    Tensor, classical, coppersmith_winograd, direct_sum, from_scheme, linear_maps,
    matrix_product, polynomial_coefficient, restrict_coordinates, tensor_power,
    tensor_product, to_scheme)
from matrix_multiplication.verification import check_product


class ConstructionTests(unittest.TestCase):
    """Protect exact identities and distinctions between tensor and matrix rank."""

    @classmethod
    def setUpClass(cls):
        """Discover a rational seven-product scheme once for execution regressions."""
        cls.start = classical((2, 2, 2))
        cls.found, cls.evidence = search(cls.start, (2, 2, 2), target_rank=7,
                                         steps=1000, restarts=1, seed=0, max_height=1)

    def test_actual_discovery_and_replay(self):
        """Find rank seven from eight elementary terms, with reproducible evidence."""
        self.assertEqual(self.found.rank, 7)
        self.assertTrue(self.evidence["target_reached"])
        self.assertEqual(self.found.expand(), self.start.expand())
        self.assertEqual(replay(self.start, self.evidence["events"]), self.found)
        repeated, evidence = search(self.start, (2, 2, 2), target_rank=7,
                                    steps=1000, restarts=1, seed=0, max_height=1)
        self.assertEqual((repeated, evidence), (self.found, self.evidence))
        # Independent verifier does not import the search or tensor constructor.
        certificate_checker.check_recipe(scheme_document(to_scheme(self.found, (2, 2, 2))))

    def test_discovered_recursive_execution_and_arithmetic_counts(self):
        """Run real recursive products; count seven-power leaves, not just base rank."""
        plan = Composition([to_scheme(self.found, (2, 2, 2))])
        for size, expected in ((2, 7), (4, 49), (8, 343)):
            a = [[Fraction(i-2*j, 3) for j in range(size)] for i in range(size)]
            b = [[Fraction(2*i+j-4, 5) for j in range(size)] for i in range(size)]
            observer = Instrumentation(profile=False)
            result = plan.multiply(a, b, instrumentation=observer)
            self.assertTrue(check_product(a, b, result).passed)
            self.assertEqual(observer.counts["field_multiplications"], expected)
            budget = plan.budget(size, size, size)
            self.assertEqual(observer.snapshot()["field_arithmetic_operations"], budget["field_arithmetic_operations"])
        # Non-power size and cutoff exercise padding and classical leaf recovery.
        a, b = [[i+j for j in range(5)] for i in range(3)], [[i-j for j in range(2)] for i in range(5)]
        for cutoff in (1, 2, 8):
            self.assertTrue(check_product(a, b, plan.multiply(a, b, cutoff=cutoff)).passed)

    def test_every_flip_direction_preserves_the_tensor(self):
        """Check all classical neighbors and both signs in all three common legs."""
        expected = self.start.expand()
        for leg, i, j in flip_choices(self.start):
            for amount in (-1, 1, Fraction(1, 2)):
                self.assertEqual(flip(self.start, leg, i, j, amount).expand(), expected)
        with self.assertRaises(ValueError):
            flip(self.start, 0, 0, 7, 1)

    def test_general_dependence_reduction_and_adversarial_log(self):
        """Reduce a true linear relation that is not a repeated pair of vectors."""
        # Pair vectors e0, e1, e0+e1 are dependent; all three free factors differ.
        tensor = Tensor((1, 2, 2), (
            (((0, 1),), ((0, 1),), ((0, 1),)),
            (((0, 1),), ((1, 1),), ((1, 1),)),
            (((0, 1),), ((0, 1), (1, 1)), ((0, 2), (1, 3)))))
        reduced, events = reduce_all(tensor)
        self.assertLess(reduced.rank, tensor.rank)
        self.assertEqual(reduced.expand(), tensor.expand())
        self.assertEqual(replay(tensor, events), reduced)
        with self.assertRaises(ValueError):
            reduce_relation(tensor, 2, 2, ((2, 1), (0, -1)))
        self.assertIsNone(dependence(self.start, 0))

    def test_tensor_product_and_matrix_reindexing(self):
        """Verify nontrivial rectangular flattening against independent classical targets."""
        for first, second in (((2, 2, 2), (2, 3, 2)), ((2, 3, 4), (3, 2, 2))):
            a, b = classical(first), classical(second)
            shape = tuple(x*y for x, y in zip(first, second))
            result = matrix_product(a, first, b, second)
            self.assertEqual(result.expand(), classical(shape).expand())
            to_scheme(result, shape)
        squared = matrix_product(self.found, (2, 2, 2), self.found, (2, 2, 2))
        self.assertEqual(squared.rank, 49)
        self.assertEqual(squared.expand(), classical((4, 4, 4)).expand())
        certificate_checker.check_recipe(scheme_document(to_scheme(squared, (4, 4, 4))))
        self.assertEqual(tensor_power(self.start, 0).expand(), {(0, 0, 0): 1})
        self.assertEqual(tensor_power(self.start, 2), tensor_product(self.start, self.start))

    def test_restriction_direct_sum_and_non_matrix_rejection(self):
        """Extract one disjoint summand and reject mislabelled matrix algorithms."""
        summed = direct_sum(self.start, self.start)
        extracted = restrict_coordinates(summed, (tuple(range(4, 8)),)*3)
        self.assertEqual(extracted.expand(), self.start.expand())
        with self.assertRaises(ValueError):
            to_scheme(coppersmith_winograd(2), (2, 2, 2))
        with self.assertRaises(ValueError):
            restrict_coordinates(self.start, ((0, 0), (0,), (0,)))
        # A rational map can merge coordinates, with exact coefficient addition.
        maps = (tuple((((0, Fraction(1, 2)),)) for _ in range(4)),)*3
        merged = linear_maps(self.start, (1, 1, 1), maps)
        self.assertEqual(merged.expand(), {(0, 0, 0): 1})

    def test_polynomial_recovery_counts_overhead(self):
        """Recover a nonconstant coefficient that needs cancellation of high powers."""
        terms = ((((0, (1, 2)),), ((0, (3, 4)),), ((0, (5, 6)),)),)
        # (1+2e)(3+4e)(5+6e) has e^2 coefficient 40+36+24 = 100.
        recovered = polynomial_coefficient((1, 1, 1), terms, 3, 2)
        self.assertEqual(recovered.expand(), {(0, 0, 0): 100})
        self.assertEqual(recovered.rank, 4)
        with self.assertRaises(ValueError):
            polynomial_coefficient((1, 1, 1), terms, 2, 1)
        with self.assertRaises(ValueError):
            Tensor((1, 1, 1), ((((0, 1.0),), ((0, 1),), ((0, 1),)),))

    def test_invalid_polynomial_coordinates_cannot_disappear_at_evaluation(self):
        """Even a zero polynomial must have a valid source coordinate and space."""
        for coordinate in (True, -1, 1):
            terms = ((((coordinate, (0,)),), ((0, (1,)),), ((0, (1,)),)),)
            with self.subTest(coordinate=coordinate), self.assertRaises(ValueError):
                polynomial_coefficient((1, 1, 1), terms, 0, 0)
        with self.assertRaises(ValueError):
            linear_maps(self.start, (1, 1), ((), (), ()))

    def test_cw_degenerations_and_shared_parameter_powers(self):
        """Check complete recovered CW powers and all lower coefficients cancel."""
        for q, power in ((1, 1), (2, 1), (2, 2)):
            tensor, evidence = recover_cw(q, power)
            self.assertEqual(tensor.expand(), tensor_power(coppersmith_winograd(q), power).expand())
            self.assertEqual(evidence["polynomial_rank"], (q+2)**power)
            self.assertEqual(evidence["finite_rank_bound"], (10*power+1)*(q+2)**power)
            self.assertGreater(evidence["finite_recipe_products"], evidence["polynomial_rank"])
            self.assertFalse(evidence["matrix_multiplication_recipe"])
        dimensions, terms, degree, _ = cw_family(1)
        for coefficient in range(3):
            self.assertEqual(polynomial_coefficient(dimensions, terms, degree, coefficient).expand(), {})

    def test_exact_exponent_targets_and_unsuccessful_search(self):
        """Do not turn an unsuccessful bounded walk into a new exponent claim."""
        self.assertFalse(beats_exponent(7, 2, 11, 4))
        self.assertTrue(beats_exponent(6, 2, 11, 4))
        self.assertFalse(beats_exponent(16, 4, 2, 1))  # Strict inequality.
        self.assertEqual(target_rank(2, Fraction(11, 4)), 6)
        self.assertEqual(target_rank(2, Fraction(4)), 15)
        for exponent in (True, 2.75, Fraction(0), Fraction(-1)):
            with self.subTest(exponent=exponent), self.assertRaises(ValueError):
                target_rank(2, exponent)
        best, evidence = search(self.start, (2, 2, 2), target_rank=6, steps=0, restarts=1)
        self.assertEqual(best.rank, 8)
        self.assertFalse(evidence["target_reached"])
        for settings in ({"steps": -1}, {"restarts": 0}, {"max_height": 0}):
            with self.assertRaises(ValueError):
                search(self.start, (2, 2, 2), target_rank=7, **settings)

    def test_search_height_ceiling_covers_start_and_reduction_results(self):
        """Exact identity preservation alone must not bypass the coefficient budget."""
        from unittest.mock import patch
        # Scaling U by two and W by one half preserves every outer product,
        # producing a genuine rank-seven identity whose height is two.
        scaled = Tensor(self.found.dimensions, tuple(
            (tuple((i, 2*c) for i, c in u), v, tuple((i, c/2) for i, c in w))
            for u, v, w in self.found.terms))
        self.assertEqual(scaled.expand(), self.start.expand())
        with self.assertRaisesRegex(ValueError, "starting recipe exceeds"):
            search(scaled, (2, 2, 2), target_rank=6, max_height=1)
        # Inject a mathematically valid but out-of-budget reduction result.
        # The search must retain its original state, even though the rank fell.
        with patch("matrix_multiplication.decomposition_search.reduce_all", return_value=(scaled, [])):
            best, evidence = search(self.start, (2, 2, 2), target_rank=7,
                                    steps=4, restarts=1, max_height=1)
        self.assertEqual(best, self.start)
        self.assertFalse(evidence["target_reached"])
        self.assertEqual(evidence["accepted_flips"], 0)

    def test_optional_splits_preserve_identity_and_replay(self):
        """Permit bounded temporary rank increases without labelling them improvements."""
        piece = ((1, Fraction(1)),)
        expanded = split(self.start, 0, 0, piece)
        self.assertEqual(expanded.rank, 9)
        self.assertEqual(expanded.expand(), self.start.expand())
        event = {"kind": "split", "term": 0, "leg": 0, "piece": [[1, "1"]]}
        self.assertEqual(replay(self.start, [event]), expanded)
        best, evidence = search(self.start, (2, 2, 2), target_rank=7, steps=25,
                                restarts=1, seed=4, extra_rank=2, split_every=5, max_height=4)
        self.assertGreater(evidence["accepted_splits"], 0)
        self.assertEqual(best.expand(), self.start.expand())
        self.assertEqual(replay(self.start, evidence["events"]), best)

    def test_cli_artifacts_replay_certificate_and_provenance(self):
        """Persist usable recipes, offline summaries, and upstream data attribution."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output-root", temp, "search", "--target-rank", "7", "--certificate"]), 0)
            directory = next(root.glob("construction_*"))
            self.assertEqual(load_scheme(directory / "scheme.json").rank, 7)
            self.assertTrue((directory / "index.html").is_file())
            self.assertTrue((directory / "certificate" / "certificate.json").is_file())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output-root", temp, "replay", str(directory / "report.json")]), 0)
            input_path = root / "input.json"
            notices = {"license": "test notice", "author": "original data author"}
            save_scheme(to_scheme(self.start, (2, 2, 2)), input_path, provenance=notices)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output-root", temp, "product", "--left", str(input_path), "--right", "strassen", "--certificate"]), 0)
            reports = [json.loads((p / "report.json").read_text()) for p in root.glob("construction_*")]
            product_report = next(r for r in reports if r["command"] == "product")
            self.assertEqual(product_report["provenance"]["input_provenance"][0], notices)
            self.assertEqual(product_report["rank"], 56)
            product_directory = next(p for p in root.glob("construction_*")
                                     if json.loads((p / "report.json").read_text())["command"] == "product")
            certificate = json.loads((product_directory / "certificate" / "certificate.json").read_text())
            self.assertEqual(certificate["stages"][0]["recipe"]["provenance"]["input_provenance"][0], notices)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output-root", temp, "cw", "--q", "1"]), 0)
            bad = json.loads((directory / "report.json").read_text())
            bad["search"]["best_rank"] = 6
            bad_path = root / "bad.json"
            bad_path.write_text(json.dumps(bad))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--output-root", temp, "replay", str(bad_path)])


if __name__ == "__main__":
    unittest.main()
