"""Verify scheme interchange, mixed recursion, exact identities, and work budgets.

Independent classical answers and declared ranks catch traversal errors. Mutated
coefficient tensors must fail before execution; mixed radices and Gaussian stages
exercise the domain and padding cases absent from homogeneous recursion tests.
"""

from contextlib import redirect_stdout, redirect_stderr
from fractions import Fraction
import io
import json
from math import log
from pathlib import Path
import random
import tempfile
import unittest

import explore
from matrix_multiplication import Instrumentation, strassen, alphaevolve
from matrix_multiplication.composition import Composition
from matrix_multiplication.scheme_io import save_scheme, load_scheme, scheme_document, scheme_from_document
from matrix_multiplication.schemes import BilinearScheme
from matrix_multiplication.scalars import Gaussian
from matrix_multiplication.verification import check_product
from matrix_multiplication.work_sweep import collect_work
from matrix_multiplication.reporting import render_html
from tests.support import requires_coefficients


def classical_scheme():
    """Construct our own row-by-column radix-two specification with eight products."""
    left, right, output = [], [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                left.append(((2*i+j, 1),))
                right.append(((2*j+k, 1),))
                output.append(((2*i+k, 1),))
    return BilinearScheme("classical-2", 2, tuple(left), tuple(right), tuple(output))


class CompositionTests(unittest.TestCase):
    """Test reusable construction machinery without borrowing algorithm implementations."""

    def assert_budget(self, plan, a, b, cutoff=1):
        """Compare exact numerical results and independent predicted/captured counts."""
        observer = Instrumentation(profile=False)
        result = plan.multiply(a, b, cutoff=cutoff, instrumentation=observer)
        self.assertTrue(check_product(a, b, result).passed)
        budget = plan.budget(len(a), len(b), len(b[0]), cutoff=cutoff)
        snapshot = observer.snapshot()
        self.assertEqual(observer.counts["field_multiplications"], budget["field_multiplications"])
        for key in ("field_arithmetic_operations", "atomic_arithmetic_operations"):
            self.assertEqual(snapshot[key], budget[key])
        return result, snapshot

    def test_exact_interchange_and_provenance(self):
        """Round-trip exact Gaussian rationals and retain attribution in the run model."""
        base = classical_scheme()
        # Multiply U by i/2 and W by -2i; their product is one.
        left = tuple(tuple((i, Gaussian(0, Fraction(1, 2))) for i, _ in c) for c in base.left)
        output = tuple(tuple((i, Gaussian(0, -2)) for i, _ in c) for c in base.output)
        scheme = BilinearScheme("rational-complex-2", 2, left, base.right, output)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheme.json"
            provenance = {"author": "repository test", "license": "GPL-3.0", "source_url": "local construction"}
            save_scheme(scheme, path, provenance=provenance)
            restored = load_scheme(path)
            self.assertEqual(scheme_document(restored)["left"], scheme_document(scheme)["left"])
            plan = Composition.from_spec(str(path))
            self.assertEqual(plan.model()["scheme_documents"][0]["provenance"], provenance)
            self.assert_budget(plan, [[Fraction(1, 3), 2], [3, 4]], [[5, 6], [7, 8]])

    def test_corrupt_and_malformed_schemes_rejected(self):
        """Reject altered identities, wrong conventions, duplicates, and inexact data."""
        for mutation in ("coefficient", "output", "index", "duplicate", "float", "denominator", "version", "rank"):
            document = scheme_document(classical_scheme())
            if mutation == "coefficient":
                document["left"][0][0][1] = 2
            elif mutation == "output":
                document["output"][0][0][0] = 1
            elif mutation == "index":
                document["left"][0][0][0] = 4
            elif mutation == "duplicate":
                document["left"][0].append([0, 1])
            elif mutation == "float":
                document["left"][0][0][1] = 1.0
            elif mutation == "denominator":
                document["left"][0][0][1] = {"real": [1, 0], "imag": [0, 1]}
            elif mutation == "version":
                document["schema_version"] = True
            else:
                document["left"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                scheme_from_document(document)

    def test_verified_scheme_snapshots_mutable_tables_and_coefficients(self):
        """Source edits cannot silently change an identity after it was verified."""
        from dataclasses import FrozenInstanceError
        base = classical_scheme()
        source = [[list(term) for term in column] for column in base.left]
        scheme = BilinearScheme("snapshot", 2, source, base.right, base.output)
        scheme_from_document(scheme_document(scheme))
        source[0][0][1] = 999
        self.assertEqual(scheme.left[0][0][1], 1)
        scheme_from_document(scheme_document(scheme))
        coefficient = Gaussian(1, Fraction(1, 2))
        with self.assertRaises(FrozenInstanceError):
            coefficient.imag = 0

    def test_import_rejects_ambiguous_json_and_nonfinite_metadata(self):
        """Reject duplicate keys throughout a recipe, not merely invalid coefficients."""
        document = json.dumps(scheme_document(classical_scheme()))
        malformed = (document.replace('"radix": 2', '"radix": 99, "radix": 2'),
                     document.replace('"provenance": {}', '"provenance": {"license": "a", "license": "b"}'),
                     document.replace('"provenance": {}', '"provenance": {"diagnostic": NaN}'),
                     document.replace('"provenance": {}', '"provenance": {"diagnostic": 1e999}'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            for raw in malformed:
                path.write_text(raw, encoding="utf-8")
                for reader in (load_scheme, Composition.from_spec):
                    with self.subTest(raw=raw, reader=reader), self.assertRaises(ValueError):
                        reader(str(path))

    @requires_coefficients("alphaevolve")
    def test_homogeneous_builtin_counts_preserved(self):
        """Composition of a single built-in retains all its arithmetic counts."""
        a = [[1, -2], [3, 4]]
        for name, implementation in (("strassen", strassen.multiply), ("alphaevolve", alphaevolve.multiply)):
            observer = Instrumentation(profile=False)
            expected = implementation(a, a, instrumentation=observer)
            actual, snapshot = self.assert_budget(Composition([name]), a, a)
            self.assertEqual(actual, expected)
            self.assertEqual(snapshot["counts"], observer.snapshot()["counts"])

    @requires_coefficients("alphaevolve")
    def test_mixed_radices_rectangles_cutoffs_and_domains(self):
        """Exercise both stage orders, signed padding, multiple cycles, and exact fractions."""
        rng = random.Random(2026)
        for stages in (["strassen", "alphaevolve"], ["alphaevolve", "strassen-winograd"],
                       [classical_scheme(), "strassen"]):
            plan = Composition(stages)
            shapes = [(3, 5, 2), (8, 8, 8)]
            if stages[0] == "strassen":
                shapes.append((9, 2, 3))  # Three levels, beginning a second cycle.
            for rows, inner, columns in shapes:
                a = [[rng.randrange(-3, 4) for _ in range(inner)] for _ in range(rows)]
                b = [[rng.randrange(-3, 4) for _ in range(columns)] for _ in range(inner)]
                for cutoff in (1, 2, 4, 64):
                    with self.subTest(stages=stages, shape=(rows, inner, columns), cutoff=cutoff):
                        self.assert_budget(plan, a, b, cutoff)
            self.assert_budget(plan, [[Fraction(1, 7), 10**50]], [[3], [Fraction(-2, 9)]])
            self.assert_budget(plan, [[1j, 2]], [[3], [4j]])

    @requires_coefficients("alphaevolve")
    def test_cycle_rank_and_domain_counts_are_independent_of_inputs(self):
        """A radix-eight rank-336 cycle predicts 336 leaf products for 8x8 inputs."""
        plan = Composition(["strassen", "alphaevolve"])
        model = plan.model()
        self.assertEqual((model["base_dimension"], model["base_bilinear_rank"]), (8, 336))
        self.assertAlmostEqual(model["square_arithmetic_exponent"], log(336)/log(8))
        budget = plan.budget(8, 8, 8)
        self.assertEqual(budget["field_multiplications"], 336)
        zero = [[0]*8 for _ in range(8)]
        one = [[1]*8 for _ in range(8)]
        self.assertEqual(self.assert_budget(plan, zero, zero)[1]["counts"],
                         self.assert_budget(plan, one, one)[1]["counts"])

    def test_normalized_integer_scheme_and_contract(self):
        """Support common coefficient scale and reject invalid inputs/plans."""
        base = classical_scheme()
        doubled = tuple(tuple((i, 2*v) for i, v in c) for c in base.output)
        plan = Composition([BilinearScheme("scaled", 2, base.left, base.right, doubled, divisor=2)])
        self.assert_budget(plan, [[1, 2], [3, 4]], [[5, 6], [7, 8]])
        wrapped = tuple(tuple((i, Gaussian(v, 0)) for i, v in c) for c in base.left)
        real_plan = Composition([BilinearScheme("wrapped-real", 2, wrapped, base.right, base.output)])
        self.assertFalse(real_plan.gaussian)
        self.assert_budget(real_plan, [[1, 2], [3, 4]], [[5, 6], [7, 8]])
        for stages in ([], ["naive"], ["unknown"]):
            with self.assertRaises(ValueError):
                Composition(stages)
        for cutoff in (0, True, 1.5):
            with self.assertRaises(ValueError):
                plan.multiply([[1]], [[2]], cutoff=cutoff)
        with self.assertRaises(ValueError):
            plan.multiply([[1, 2]], [[3]])

    @requires_coefficients("alphaevolve")
    def test_explorer_generates_report_with_schedule_and_budget(self):
        """Run the public CLI and confirm automatic HTML retains composition evidence."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = explore.main(["--composition", "strassen,alphaevolve", "--report", str(path),
                                       "--output-root", directory, "--counts-only"])
            self.assertEqual(status, 0)
            report = json.loads(path.read_text())["composition"]
            self.assertTrue(all(report["composition_budget_matches"].values()))
            page = next(Path(directory).glob("*/index.html")).read_text(encoding="utf-8")
            self.assertIn("Repeating composition schedule", page)
            self.assertIn("Predicted", page)
        diagnostics = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(diagnostics):
            self.assertEqual(explore.main(["--composition", ""]), 1)
        self.assertIn("stage names must not be empty", diagnostics.getvalue())

    @requires_coefficients("alphaevolve")
    def test_sweep_compares_composition_to_builtin(self):
        """Execute growth samples and render discrete predictions without a false smooth model."""
        data = collect_work([1, 2, 3, 8], [1, 2], algorithms=["naive"],
                            compositions={"mixed": Composition(["strassen", "alphaevolve"])})
        for line in data["series"]:
            for sample in line["samples"]:
                self.assertEqual(sample["status"], "passed")
                self.assertTrue(sample["prediction_matches_observed_data_multiplications"])
                self.assertTrue(sample["prediction_matches_observed_arithmetic_steps"])
            if line["algorithm"] == "mixed":
                self.assertEqual(line["ideal_curve"], [])
        page = render_html(data, title="Composition", source_name="test.json", source_hash="test", generated="test")
        self.assertIn("mixed", page)
        self.assertIn("Composition schedules", page)
        self.assertIn("stopping at different points in that order can give different finite costs", page)


if __name__ == "__main__":
    unittest.main()
