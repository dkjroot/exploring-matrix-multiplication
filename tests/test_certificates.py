"""Check exact certificates, adversarial alterations, and standalone portability.

Tests exercise independent algebra checks rather than trusting saved successes.
Tiny recipes keep mutation tests inexpensive; a published complex recipe covers
the domain needed by our strongest current construction.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from matrix_multiplication import certificate_checker as checker
from matrix_multiplication.certificates import create_certificate, formula_recipe, write_bundle, main
from matrix_multiplication.composition import Composition
from matrix_multiplication.scheme_io import scheme_from_document
from tests.support import requires_coefficients


class CertificateTests(unittest.TestCase):
    """Protect the distinction between a full exact check and claimed evidence."""

    def fixture(self):
        """Build a small genuine certificate for mutation tests."""
        return create_certificate(Composition(["strassen"]))

    def test_formula_extraction_and_full_identity_checks(self):
        """Extract both maintained formulas and check all 64 coefficient relationships."""
        for name in ("strassen", "strassen-winograd"):
            document = formula_recipe(name)
            scheme_from_document(document)
            evidence = checker.check_recipe(document)
            self.assertEqual((evidence["rank"], evidence["radix"], evidence["coefficient_checks"]), (7, 2, 64))
            self.assertEqual(evidence["basis_pairs"], 16)

    def test_formula_setup_is_excluded_from_an_enclosing_observer(self):
        """Symbolic extraction must not pollute an existing scalar experiment."""
        from matrix_multiplication import Instrumentation
        from matrix_multiplication.instrumentation import session
        from matrix_multiplication.scalars import multiply
        observer = Instrumentation(trace=True)
        with session(observer):
            formula_recipe("strassen")
            self.assertEqual(observer.snapshot()["field_arithmetic_operations"], 0)
            self.assertEqual(observer.events, [])
            self.assertEqual(multiply(2, 3), 6)
        self.assertEqual(observer.counts["field_multiplications"], 1)

    @requires_coefficients("alphaevolve")
    def test_complex_identity_and_cycle(self):
        """Check AlphaEvolve's 4096 relationships and derive a mixed cycle from exact integers."""
        certificate = create_certificate(Composition(["strassen", "alphaevolve"]))
        verified = checker.check_certificate(certificate)
        self.assertEqual(verified["stages"][1]["coefficient_checks"], 4096)
        self.assertEqual((verified["cycle"]["radix"], verified["cycle"]["rank"]), (8, 336))
        self.assertEqual(verified["cycle"]["exponent_expression"], "log_8(336)")

    def test_altered_coefficient_rejected_even_with_new_hash(self):
        """A forged matching fingerprint cannot rescue a mathematically false recipe."""
        certificate = self.fixture()
        stage = certificate["stages"][0]
        stage["recipe"]["left"][0][0][1] += 1
        stage["recipe_sha256"] = checker.fingerprint(stage["recipe"])
        stage["evidence"]["recipe_sha256"] = stage["recipe_sha256"]
        with self.assertRaisesRegex(ValueError, "basis identity failed"):
            checker.check_certificate(certificate)

    def test_hash_evidence_cycle_and_conventions_rejected(self):
        """Reject changed data binding, saved claims, field rules, and cycle parameters."""
        original = self.fixture()
        for mutation in ("hash", "evidence", "rank", "repeat", "field", "convention", "empty"):
            certificate = deepcopy(original)
            if mutation == "hash":
                certificate["stages"][0]["recipe_sha256"] = "0"*64
            elif mutation == "evidence":
                certificate["stages"][0]["evidence"]["coefficient_checks"] = 63
            elif mutation == "rank":
                certificate["cycle"]["rank"] = 6
            elif mutation == "repeat":
                certificate["cycle"]["repeats"] = 1
            elif mutation == "field":
                certificate["field"] = "floating point"
            elif mutation == "convention":
                certificate["conventions"] = "transposed output"
            else:
                certificate["stages"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                checker.check_certificate(certificate)

    def test_independent_checker_rejects_bad_recipe_without_library(self):
        """The second checker still rejects corruption when the first checker is bypassed."""
        plan = Composition(["strassen"])
        recipe = formula_recipe("strassen")
        recipe["output"][0][0][1] += 1
        with patch("matrix_multiplication.certificates.scheme_from_document", return_value=None), \
                patch("matrix_multiplication.certificates.formula_recipe", return_value=recipe):
            with self.assertRaisesRegex(ValueError, "basis identity failed"):
                create_certificate(plan)

    def test_imported_example_and_reproducibility(self):
        """Retain imported notices and reproduce identical certificates without timestamps."""
        path = Path(__file__).resolve().parents[1] / "schemes" / "classical-2.json"
        plan = Composition.from_spec(str(path))
        first, second = create_certificate(plan), create_certificate(plan)
        self.assertEqual(first, second)
        self.assertEqual(first["stages"][0]["recipe"]["provenance"], json.loads(path.read_text())["provenance"])
        self.assertEqual(checker.fingerprint(first), checker.fingerprint(second))

    def test_standalone_package_and_corruption_exit_status(self):
        """Run the copied checker in isolated Python, without the repository on sys.path."""
        with tempfile.TemporaryDirectory() as directory:
            package = write_bundle(Composition(["strassen"]), Path(directory) / "package")
            checksums = json.loads((package / "file-sha256.json").read_text())
            self.assertIn("LICENSE", checksums)
            for name, digest in checksums.items():
                self.assertEqual(sha256((package / name).read_bytes()).hexdigest(), digest)
            command = [sys.executable, "-I", str(package / "check_certificate.py")]
            result = subprocess.run(command, cwd=package, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS", result.stdout)
            certificate = json.loads((package / "certificate.json").read_text())
            certificate["cycle"]["rank"] = 1
            (package / "certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
            result = subprocess.run(command, cwd=package, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL", result.stdout)
            with self.assertRaises(FileExistsError):
                write_bundle(Composition(["strassen"]), package)

    def test_strict_json_and_malformed_coefficients(self):
        """Reject ambiguous JSON and unsupported exact coefficient encodings."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"schema":1,"schema":2}', '{"value":NaN}'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    checker.read_json(path)
        for value in (True, 1.0, {"real": [1, 0], "imag": [0, 1]}):
            with self.assertRaises(ValueError):
                checker.exact_pair(value)

    def test_cli_verify_status(self):
        """The public recheck command reports a failure for absent or altered packages."""
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["verify", "nonexistent-certificate.json"]), 1)


if __name__ == "__main__":
    unittest.main()
