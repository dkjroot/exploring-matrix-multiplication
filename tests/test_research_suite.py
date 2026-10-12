"""Verify single-command report orchestration without an expensive default sweep.

Tiny actual executions establish artifact interoperability, exact candidate
measurement, trace/memory scopes, and stage failure retention. Tests do not
replace identities with mocked successes. Only the failure-under-test is
injected; the remaining generators and arithmetic paths execute normally.
"""

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sweep
from matrix_multiplication import ALGORITHMS, naive
from matrix_multiplication.research_suite import (
    SuiteRun, SuiteSettings, pair_evidence, run_suite, target_evidence)


class ResearchSuiteTests(unittest.TestCase):
    """Protect complete report families, checked discoveries, and failure semantics."""

    def settings(self, **overrides):
        """Keep constructor/recovery instances small while using a real discovery."""
        values = {"cw_parameters": (1,), "cw_powers": (1,), "trace_limit": 20}
        return SuiteSettings(**{**values, **overrides})

    def test_checkpoint_retries_temporary_access_but_preserves_persistent_errors(self):
        """An atomic replace retry must not erase the previous completed manifest."""
        original_replace = Path.replace
        attempts = []
        def temporarily_locked(source, destination):
            """Emulate one busy destination before allowing the real atomic replace."""
            attempts.append(source)
            if len(attempts) == 1:
                raise PermissionError("temporary reader lock")
            return original_replace(source, destination)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(Path, "replace", temporarily_locked):
                run = SuiteRun(Path(temp) / "run", {"fixture": True})
            self.assertEqual(len(attempts), 2)
            manifest = run.directory / "suite-manifest.json"
            before = manifest.read_bytes()
            run.manifest["status"] = "changed in memory"
            with patch.object(Path, "replace", side_effect=PermissionError("persistent access denial")), patch(
                    "matrix_multiplication.research_suite.sleep"), self.assertRaises(PermissionError):
                run.checkpoint()
            self.assertEqual(manifest.read_bytes(), before)
            self.assertTrue((run.directory / "suite-manifest.pending.json").exists())

    def test_suite_preserves_all_report_families_and_measures_candidates(self):
        """Produce one dated folder, with exact replay/certificates and real counts."""
        with tempfile.TemporaryDirectory() as temp:
            index, status = run_suite([2, 4], [1, 2], algorithms=["naive", "strassen"],
                                      settings=self.settings(), output_root=temp)
            self.assertEqual(status, 0)
            root = index.parent
            self.assertEqual(list(Path(temp).iterdir()), [root])
            manifest = json.loads((root / "suite-manifest.json").read_text())
            self.assertEqual(manifest["status"], "passed")
            self.assertTrue(all(s["status"] == "passed" for s in manifest["stages"]))
            setup = next(s for s in manifest['stages'] if s['name'] == 'search-1')['setup_measurements']
            self.assertEqual(setup['python_traced_peak_bytes']['status'], 'measured')
            self.assertGreater(setup['elapsed_seconds'], 0)
            self.assertEqual(setup['coefficient_arithmetic_operations']['status'], 'not_counted')
            families = {s["family"] for s in manifest["stages"]}
            self.assertTrue({"discovery", "replay", "construction", "polynomial tensor recovery", "exact certificate",
                             "verified independent product extraction", "exact recovery optimization and dense packing",
                             "coarse CW classification and verified packing",
                             "profiles, memory, accuracy, traces",
                             "measured arithmetic and theoretical comparison"} <= families)
            extraction = json.loads((root / "cw-extraction-q1-power1" / "report.json").read_text())["cw_extraction"]
            self.assertTrue(extraction["recovered_direct_sum_identity_verified"])
            self.assertTrue(extraction["execution"]["exact_outputs_verified"])
            self.assertIsNone(extraction["exponent"])
            coarse = json.loads((root / "cw-coarse-q1-power1" / "report.json").read_text())["cw_coarse"]
            self.assertTrue(coarse["partition_reconstruction_verified"])
            self.assertTrue(coarse["packing"]["dense_matrix_identity_verified"])
            integration = next(s for s in manifest["stages"] if s["name"] == "register-generated-cw-packed")
            self.assertTrue(any(c["construction"] == "cw-coarse-q1-power1" for c in integration["candidates"]))
            for stage in manifest["stages"]:
                for key in ("data", "report", "scheme", "argument"):
                    if key in stage:
                        self.assertTrue((root / stage[key].split("#")[0]).is_file(), stage)
            for relative, digest in manifest["artifact_sha256"].items():
                self.assertEqual(sha256((root / relative).read_bytes()).hexdigest(), digest)
            data = json.loads((root / "measurements.json").read_text())
            self.assertEqual(data["configuration"]["sizes"], [2, 4])
            self.assertEqual(len(data["reference_reports"]), 2)
            generated = next(line for line in data["series"] if line["algorithm"] == "generated-search-1" and line["cutoff"] == 1)
            self.assertEqual([s["observation"]["counts"]["field_multiplications"] for s in generated["samples"]], [7, 49])
            self.assertTrue(all(s["prediction_matches_observed_arithmetic_steps"] for s in generated["samples"]))
            packed = next(line for line in data["series"] if line["algorithm"] == "generated-cw-packed" and line["cutoff"] == 1)
            self.assertEqual([s["observation"]["counts"]["field_multiplications"] for s in packed["samples"]], [8, 64])
            self.assertTrue(all(s["prediction_matches_observed_arithmetic_steps"] for s in packed["samples"]))
            pair = json.loads((root / "pair-exact" / "measurements.json").read_text())
            for name, report in pair["reports"].items():
                self.assertEqual(report["status"], "passed", report)
                self.assertEqual(report["execution_measurements"]["python_traced_peak_bytes"]["status"], "measured")
                self.assertGreater(report["execution_measurements"]["elapsed_algorithm_call_seconds"], 0)
                self.assertLessEqual(len(pair["events"][name]), 20)
            html = index.read_text(encoding="utf-8")
            for phrase in ("Complete research suite", "Arithmetic work breakdown", "Square arithmetic exponents",
                           "Data products by recursion depth", "Primitive profiles", "Accuracy and verification evidence",
                           "All recorded operation and structural counters", "Original earlier JSON snapshot",
                           "Finite product-count targets", "pair-floating/index.html", "certificate-naive"):
                self.assertIn(phrase, html)
            self.assertNotIn("<script", html)

    def test_search_miss_is_evidence_but_a_failed_stage_is_not_success(self):
        """Retain an unsuccessful finite search; preserve other stages on recovery error."""
        with tempfile.TemporaryDirectory() as temp, patch("matrix_multiplication.constructions.recover_cw", side_effect=ValueError("injected CW check failure")):
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                      settings=self.settings(search_target_rank=6, search_steps=0, search_restarts=1))
            self.assertEqual(status, 1)
            manifest = json.loads((index.parent / "suite-manifest.json").read_text())
            discovery = next(s for s in manifest["stages"] if s["name"] == "search-1")
            self.assertEqual(discovery["status"], "passed")
            self.assertFalse(discovery["target_reached"])
            cw = next(s for s in manifest["stages"] if s["name"].startswith("cw-"))
            self.assertEqual(cw["status"], "error")
            self.assertIn("CW check failure", cw["error"])
            self.assertEqual(next(s for s in manifest["stages"] if s["name"] == "arithmetic-growth")["status"], "passed")
            self.assertTrue((index.parent / "certificates" / "naive" / "certificate.json").exists())
            self.assertIn("Suite status: <strong>failed", index.read_text())

    def test_diagnostic_pair_failure_keeps_counters_and_verification(self):
        """A corrupted numerical answer cannot appear as a healthy diagnostic run."""
        def wrong(a, b, **kwargs):
            """Execute real observed arithmetic, then return an incorrect first entry."""
            result = naive.multiply(a, b, **kwargs)
            result[0][0] += 1
            return result

        envelope = pair_evidence({"naive": wrong}, {}, trace_limit=3)
        report = envelope["reports"]["naive"]
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["execution_measurements"]["accuracy"]["passed"])
        self.assertEqual(report["counts"]["field_multiplications"], 8)
        self.assertEqual(len(envelope["events"]["naive"]), 3)

    def test_packing_identity_failure_retains_extraction_and_other_reports(self):
        """A failed new construction must not hide its error or discard other work."""
        with tempfile.TemporaryDirectory() as temp, patch("matrix_multiplication.constructions.pack_cw",
                                                        side_effect=AssertionError("injected invalid packing")):
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                      settings=self.settings(search_steps=0))
            manifest = json.loads((index.parent / "suite-manifest.json").read_text())
            stages = {stage["name"]: stage for stage in manifest["stages"]}
            self.assertEqual(status, 1)
            self.assertEqual(stages["cw-packing-q1-power1"]["status"], "error")
            self.assertIn("invalid packing", stages["cw-packing-q1-power1"]["error"])
            self.assertEqual(stages["cw-extraction-q1-power1"]["status"], "passed")
            self.assertEqual(stages["arithmetic-growth"]["status"], "passed")
            self.assertEqual(stages["cw-coarse-q1-power1"]["status"], "passed")
            self.assertEqual(stages["register-generated-cw-packed"]["selected_construction"], "cw-coarse-q1-power1")

    def test_coarse_classification_failure_keeps_fine_packing_and_diagnostics(self):
        """A broken coarse identity fails its stage while fine evidence survives."""
        with tempfile.TemporaryDirectory() as temp, patch("matrix_multiplication.cw_coarsening.classify_matrix_tensor",
                                                        side_effect=AssertionError("injected classifier failure")):
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                      settings=self.settings(search_steps=0))
            stages = {s["name"]: s for s in json.loads((index.parent / "suite-manifest.json").read_text())["stages"]}
            self.assertEqual(status, 1)
            self.assertEqual(stages["cw-coarse-q1-power1"]["status"], "error")
            self.assertIn("classifier failure", stages["cw-coarse-q1-power1"]["error"])
            self.assertEqual(stages["cw-packing-q1-power1"]["status"], "passed")
            self.assertEqual(stages["register-generated-cw-packed"]["selected_construction"], "cw-packing-q1-power1")
            self.assertEqual(stages["arithmetic-growth"]["status"], "passed")

    def test_candidate_integration_failure_keeps_independent_stages(self):
        """A generated backend failure must produce a failed, navigable suite."""
        for error in (ValueError("unsupported candidate"), AssertionError("identity assertion"), RuntimeError("backend bug")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp, patch(
                    "matrix_multiplication.research_suite.generated_plan", side_effect=error):
                index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                          settings=self.settings(search_steps=0))
                self.assertEqual(status, 1)
                manifest = json.loads((index.parent / "suite-manifest.json").read_text())
                integration = [s for s in manifest["stages"] if s["family"] == "candidate integration"]
                self.assertEqual(len(integration), 3)
                self.assertTrue(all(s["status"] == "error" for s in integration))
                self.assertTrue(all(s["error_type"] == type(error).__name__ and "Traceback" in s["traceback"] for s in integration))
                self.assertEqual(next(s for s in manifest["stages"] if s["name"] == "arithmetic-growth")["status"], "passed")
                measured = json.loads((index.parent / "measurements.json").read_text())
                self.assertEqual({line["algorithm"] for line in measured["series"]}, {"naive"})
                self.assertTrue((index.parent / "replay-1" / "report.json").exists())
                self.assertIn(str(error), index.read_text())

    def test_missing_reference_preserves_completed_measurements(self):
        """Retain primary JSON and render it even if optional report enrichment fails."""
        with tempfile.TemporaryDirectory() as temp:
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                      reference_reports=[Path(temp) / "missing.json"],
                                      settings=self.settings(search_steps=0))
            self.assertEqual(status, 1)
            self.assertIn("Arithmetic work breakdown", index.read_text())
            data = json.loads((index.parent / "measurements.json").read_text())
            self.assertTrue(all(s["status"] == "passed" for line in data["series"] for s in line["samples"]))
            manifest = json.loads((index.parent / "suite-manifest.json").read_text())
            self.assertEqual(next(s for s in manifest["stages"] if s["name"] == "arithmetic-growth")["status"], "error")

    def test_root_renderer_failure_has_readable_fallback_and_raw_evidence(self):
        """Record presentation failure without losing successful mathematical evidence."""
        from matrix_multiplication.reporting import render_html
        def failing_root(*args, **kwargs):
            """Fail only the combined page; leave diagnostic pages normally rendered."""
            if kwargs.get("title") == "Matrix multiplication research suite":
                raise ValueError("injected root renderer failure")
            return render_html(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, patch(
                "matrix_multiplication.research_suite.render_html", side_effect=failing_root):
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                      settings=self.settings(search_steps=0))
            self.assertEqual(status, 1)
            page = index.read_text()
            self.assertIn("injected root renderer failure", page)
            self.assertIn("<style>", page)
            self.assertIn("could not be produced", page)
            self.assertTrue((index.parent / "measurements.json").is_file())
            self.assertTrue((index.parent / "pair-exact" / "index.html").is_file())

    def test_all_registered_pair_domains_have_matching_budgets(self):
        """Exercise complex coefficient and rectangular plans in both diagnostic domains."""
        from matrix_multiplication.composition import Composition
        from tests.support import available_algorithms
        available = available_algorithms()
        plans = {name: Composition.from_spec(name) for name in available if name != "naive"}
        for floating in (False, True):
            envelope = pair_evidence(available, plans, floating=floating, trace_limit=2)
            for name, report in envelope["reports"].items():
                self.assertEqual(report["status"], "passed", (name, report.get("error")))
                if name != "naive":
                    self.assertTrue(all(report["composition_budget_matches"].values()))

    def test_targets_settings_and_preflight_validation(self):
        """Keep exact thresholds separate from achievable ranks; reject invalid requests early."""
        table = target_evidence()
        self.assertEqual(next(r for r in table["rows"] if r["dimension"] == 4)["11/4"], 45)
        self.assertEqual(next(r for r in table["rows"] if r["dimension"] == 32)["11/4"], 13777)
        self.assertIn("no existence claim", table["meaning"])
        self.assertEqual(self.settings(search_target_exponent=Fraction(11, 4)).document()["search_target_exponent"], "11/4")
        for overrides in ({"search_steps": -1}, {"cw_parameters": ()}, {"cw_powers": (0,)}, {"cw_packing_size": 1},
                          {"search_seeds": ()}, {"search_target_rank": 0},
                          {"search_split_every": 0}, {"search_start": ""},
                          {"search_target_rank": 7, "search_target_exponent": Fraction(11, 4)}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.settings(**overrides).validate()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_suite([0], [1], output_root=temp)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_existing_recipe_start_and_source_fingerprints(self):
        """Search a maintained explicit recipe through the suite rather than classical terms."""
        with tempfile.TemporaryDirectory() as temp:
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=temp,
                                      settings=self.settings(search_start="strassen", search_target_rank=7,
                                                             search_steps=0, search_split_every=7))
            self.assertEqual(status, 0)
            manifest = json.loads((index.parent / "suite-manifest.json").read_text())
            self.assertEqual(manifest["configuration"]["research"]["search_start"], "strassen")
            self.assertEqual(manifest["configuration"]["research"]["search_split_every"], 7)
            self.assertIn("sweep.py", manifest["configuration"]["source_sha256"])
            self.assertIn("matrix_multiplication/decomposition_search.py", manifest["configuration"]["source_sha256"])
            report = json.loads((index.parent / "search-1" / "report.json").read_text())
            self.assertEqual(report["search"]["initial_rank"], 7)
            self.assertEqual(report["search"]["attempted_flips"], 0)

    def test_rectangular_discovery_joins_square_measurement_and_certificates(self):
        """Balance a generated rectangular base through three verified rotations."""
        from matrix_multiplication.scheme_io import save_scheme
        from matrix_multiplication.tensors import classical, to_scheme
        with tempfile.TemporaryDirectory() as temp:
            recipe_path = Path(temp) / "rectangular.json"
            save_scheme(to_scheme(classical((2, 2, 3)), (2, 2, 3)), recipe_path,
                        provenance={"author": "test construction", "license": "GPL-3.0"})
            index, status = run_suite([2], [1], algorithms=["naive"], output_root=Path(temp) / "runs",
                                      settings=self.settings(search_start=str(recipe_path), search_target_rank=12,
                                                             search_steps=0))
            self.assertEqual(status, 0)
            data = json.loads((index.parent / "measurements.json").read_text())
            line = next(line for line in data["series"] if line["algorithm"] == "generated-search-1")
            self.assertEqual(line["mathematical_model"]["base_dimension"], 12)
            self.assertEqual(line["mathematical_model"]["base_bilinear_rank"], 12**3)
            self.assertEqual(line["mathematical_model"]["stage_dimensions"], [[2, 2, 3], [2, 3, 2], [3, 2, 2]])
            self.assertTrue(line["samples"][0]["prediction_matches_observed_arithmetic_steps"])
            certificate = json.loads((index.parent / "certificates" / "generated-1" / "certificate.json").read_text())
            self.assertEqual(certificate["cycle"]["radix"], 12)
            self.assertEqual(len(certificate["stages"]), 3)

    def test_sweep_defaults_to_suite_and_keeps_original_mode(self):
        """Expose all families by default while retaining the original count-only command."""
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            status = sweep.main(["--sizes", "2", "--cutoffs", "1", "--algorithms", "naive",
                                 "--cw-parameters", "1", "--cw-powers", "1", "--output-root", temp])
            self.assertEqual(status, 0)
            folder = next(Path(temp).iterdir())
            self.assertTrue((folder / "suite-manifest.json").is_file())
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            status = sweep.main(["--sizes", "2", "--cutoffs", "1", "--algorithms", "naive", "--measurements-only", "--output-root", temp])
            self.assertEqual(status, 0)
            folder = next(Path(temp).iterdir())
            self.assertFalse((folder / "suite-manifest.json").exists())
            self.assertEqual({line["algorithm"] for line in json.loads((folder / "measurements.json").read_text())["series"]}, {"naive"})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sweep.main(["--search-target-rank", "7", "--search-target-exponent", "11/4"])


if __name__ == "__main__":
    unittest.main()
