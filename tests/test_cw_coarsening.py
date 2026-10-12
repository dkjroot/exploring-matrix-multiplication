"""Check merged partitions, conservative matrix classification and full packing.

These tests use complete rational coefficient identities, independently rebuilt
source maps, and the standalone matrix certificate checker. Actual recursive
calls additionally check count budgets. No mathematical improvement is inferred
from classification success or from merely matching vector-space dimensions.
"""

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from matrix_multiplication import certificate_checker
from matrix_multiplication.composition import Composition
from matrix_multiplication.constructions import ConstructionRequest, main, run_experiment
from matrix_multiplication.cw_coarsening import (coarse_catalog, coarse_conflicts,
                                                extract_coarse_cw, weight_groups)
from matrix_multiplication.cw_extraction import choose_independent_words, component, restrict_polynomial
from matrix_multiplication.decomposition_search import replay
from matrix_multiplication.degenerations import cw_family
from matrix_multiplication.dense_packing import pack_coarse_cw
from matrix_multiplication.instrumentation import Instrumentation
from matrix_multiplication.tensor_classification import classify_matrix_tensor, dimension_shape
from matrix_multiplication.tensors import (Tensor, classical, coefficient_expansion,
                                          linear_maps, restrict_coordinates)
from matrix_multiplication.verification import check_product


class CoarseCWTests(unittest.TestCase):
    """Protect all catalog components and every acceptance boundary downstream."""

    def test_matrix_recognizer_handles_scrambled_rectangular_and_unit_bases(self):
        """Recover a real matrix identity from unrelated permutations of each leg."""
        rng = random.Random(71)
        for shape in ((1, 1, 6), (6, 1, 1), (1, 6, 1), (1, 1, 1), (2, 3, 4), (3, 2, 2)):
            with self.subTest(shape=shape):
                source = classical(shape)
                maps = []
                for dimension in source.dimensions:
                    permutation = list(range(dimension))
                    rng.shuffle(permutation)
                    maps.append(tuple(((index, 1),) for index in permutation))
                scrambled = linear_maps(source, source.dimensions, maps)
                layout, evidence = classify_matrix_tensor(scrambled)
                self.assertIsNotNone(layout, evidence)
                self.assertEqual(layout.shape, shape)
                self.assertTrue(evidence["complete_identity_verified"])
                self.assertEqual(restrict_coordinates(scrambled, layout.selections).expand(), source.expand())

    def test_dimension_and_support_obstructions_do_not_make_global_claims(self):
        """Matching dimensions cannot admit a missing product or unrelated tensor."""
        self.assertIsNone(dimension_shape((4, 4, 6)))
        self.assertEqual(dimension_shape((6, 12, 8)), (2, 3, 4))
        source = classical((2, 2, 2))
        latin = Tensor((4, 4, 4), tuple((((i, 1),), ((j, 1),), (((i+j) % 4, 1),))
                                       for i in range(4) for j in range(4)))
        scaled = linear_maps(source, source.dimensions,
                             tuple(tuple(((i, 2 if leg == 0 else 1),) for i in range(d))
                                   for leg, d in enumerate(source.dimensions)))
        for false in (Tensor(source.dimensions, source.terms[:-1]), latin, scaled):
            with self.subTest(false=false):
                layout, evidence = classify_matrix_tensor(false)
                self.assertIsNone(layout)
                self.assertFalse(evidence["recognized"])
                self.assertFalse(evidence["general_linear_isomorphism_searched"])
                self.assertEqual(evidence["status"], "not recognized")

    def test_catalog_partition_and_each_matrix_witness_are_complete(self):
        """Inspect all sectors at several powers, including the unrecognized ones."""
        for q in (1, 2):
            for power in (1, 2, 3):
                with self.subTest(q=q, power=power):
                    catalog, groups = coarse_catalog(q, power)
                    self.assertEqual(len(catalog), (2*power+1)*(2*power+2)//2)
                    flattened = [i for group in groups for i in group]
                    self.assertEqual(sorted(flattened), list(range((q+2)**power)))
                    self.assertEqual(len(flattened), len(set(flattened)))
                    for item in catalog:
                        document = item.document()
                        # Rebuild from saved support, independently of the literal
                        # decomposition used to generate the catalog.
                        rebuilt = Tensor(item.tensor.dimensions, tuple(
                            (((i, Fraction(*c)),), ((j, 1),), ((k, 1),))
                            for i, j, k, c in document["support_coefficients"]))
                        self.assertEqual(rebuilt.expand(), item.tensor.expand())
                        if item.layout:
                            self.assertEqual(restrict_coordinates(rebuilt, item.layout.selections).expand(),
                                             classical(item.shape).expand())
                        else:
                            with self.assertRaises(ValueError):
                                _ = item.shape
                            with self.assertRaises(ValueError):
                                _ = item.matrix_selections

    def test_power_two_retains_interior_dimension_obstructions(self):
        """The three interior tensors are evidence for subsequent extraction work."""
        catalog, groups = coarse_catalog(2, 2)
        self.assertEqual([len(group) for group in groups], [1, 4, 6, 4, 1])
        interior = [item for item in catalog if item.layout is None]
        self.assertEqual({item.sector for item in interior}, {(1, 1, 2), (1, 2, 1), (2, 1, 1)})
        self.assertTrue(all(item.classification["status"] == "dimension obstruction" for item in interior))
        large = next(item for item in catalog if item.sector == (0, 2, 2))
        self.assertEqual(large.shape, (1, 1, 6))

    def test_cross_components_and_invalid_weights_are_rejected(self):
        """Pairwise disjoint weights can still preserve unintended supported sums."""
        conflicts = coarse_conflicts(((0, 1, 3), (1, 3, 0), (3, 0, 1)), 2)
        self.assertEqual(conflicts["shared_legs"], [])
        self.assertEqual(len(conflicts["unexpected_sectors"]), 3)
        self.assertTrue(coarse_conflicts(((0, 1, 3), (0, 3, 1)), 2)["shared_legs"])
        for sectors in ((), ((0, 0, 3),), ((False, 1, 3),), ((-1, 2, 3),)):
            with self.subTest(sectors=sectors), self.assertRaises(ValueError):
                coarse_conflicts(sectors, 2)
        for q, power in ((True, 2), (2, 0), (0, 1)):
            with self.assertRaises(ValueError):
                weight_groups(q, power)

    def test_coarse_recovery_replays_and_exposes_larger_components(self):
        """Regenerate coefficient planes from recorded maps; never insert an oracle."""
        for q in (1, 2):
            with self.subTest(q=q):
                recovered, report = extract_coarse_cw(q, 2)
                _, terms, _, leading = cw_family(q, 2)
                expanded = coefficient_expansion(recovered.dimensions,
                                                restrict_polynomial(terms, report["retained_source_coordinates"]), leading)
                recovery = report["optimized_recovery"]
                self.assertEqual(replay(expanded, recovery["reduction_events"]), recovered)
                self.assertEqual(recovered.rank, recovery["selected_products"])
                self.assertEqual(recovery["execution"]["observation"]["counts"]["field_multiplications"], recovered.rank)
                self.assertTrue(report["partition_reconstruction_verified"])
                fine_words = choose_independent_words(q, 2)
                fine_volumes = [classical(component(q, word).shape).rank for word in fine_words]
                self.assertGreater(max(classical(shape).rank for shape in report["selected_component_shapes"]),
                                   max(fine_volumes))
                if q == 1:
                    self.assertGreater(report["classical_direct_sum_products"], sum(fine_volumes))

    def test_packed_coarse_recipe_executes_recursively_with_exact_budgets(self):
        """Export a genuine square recipe and independently certify its coefficients."""
        packed, evidence = pack_coarse_cw(2, 2, 2)
        self.assertEqual(packed.expand(), classical((2, 2, 2)).expand())
        self.assertEqual(evidence["packing"]["execution"]["observation"]["counts"]["field_multiplications"], packed.rank)
        with tempfile.TemporaryDirectory() as temp:
            folder, report = run_experiment(ConstructionRequest("cw-coarse", q=2, power=2, certificate=True),
                                            Path(temp) / "coarse")
            self.assertEqual(report["rank"], 8)
            self.assertEqual(report["square_recursion"]["exponent"], 3)
            certificate_checker.check_recipe(json.loads((folder / "scheme.json").read_text()))
            self.assertTrue((folder / "certificate" / "certificate.json").is_file())
            plan = Composition.from_spec(str(folder / "scheme.json"))
            for size in (2, 3, 4):
                a = [[Fraction(i-j, 3) for j in range(size)] for i in range(size)]
                b = [[Fraction(2*i+j, 5) for j in range(size)] for i in range(size)]
                observer = Instrumentation(profile=False)
                result = plan.multiply(a, b, instrumentation=observer)
                self.assertTrue(check_product(a, b, result).passed)
                self.assertEqual(observer.snapshot()["field_arithmetic_operations"],
                                 plan.budget(size, size, size)["field_arithmetic_operations"])
            html = (folder / "index.html").read_text()
            for phrase in ("Merged CW components", "Needs further extraction", "dimension obstruction",
                           "Verified finite recovery and packing costs"):
                self.assertIn(phrase, html)

    def test_coarse_cli_and_preflight(self):
        """The focused command retains catalogs and refuses invalid requests early."""
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--output-root", temp, "cw-coarse", "--certificate"]), 0)
            report = json.loads((next(Path(temp).iterdir()) / "report.json").read_text())
            self.assertEqual(report["cw_coarse"]["components_needing_further_work"], 3)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["cw-coarse", "--size", "1"])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "invalid"
            with self.assertRaises(ValueError):
                run_experiment(ConstructionRequest("cw-coarse", sector_words=()), folder)
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
