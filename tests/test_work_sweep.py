"""Verify executed-work sweeps, independent recurrence predictions, and growth plots.

Small exact experiments exercise padding/cutoff effects without using timings as
evidence of arithmetic complexity. Power-law fits are checked against known data.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matrix_multiplication import ALGORITHMS, naive, smirnov
from matrix_multiplication.work_sweep import collect_work, log_log_fit, prediction
from matrix_multiplication.composition import Composition
import sweep
from matrix_multiplication import reporting as visual
from tests.support import requires_coefficients


class WorkSweepTests(unittest.TestCase):
    """Check that predictions do not substitute for observed work or accuracy checks."""

    def test_boolean_grid_entries_are_rejected_before_deduplication(self):
        """A valid 1 must not conceal True, which compares equal in a Python set."""
        for sizes, cutoffs in (([1, True], [1]), ([1], [1, True]), ([1, 1.0], [1])):
            with self.subTest(sizes=sizes, cutoffs=cutoffs), self.assertRaises(ValueError):
                collect_work(sizes, cutoffs, algorithms=["naive"])

    def test_naive_executes_once_per_size_and_reports_reused_evidence(self):
        """Count real kernel calls, retain cutoff rows, and isolate copied snapshots."""
        with patch("matrix_multiplication.naive.classical", wraps=naive.classical) as kernel:
            data = collect_work([2, 3], [1, 2, 4], algorithms=["naive"])
        self.assertEqual(kernel.call_count, 2)
        self.assertEqual(data["execution_summary"], {"requested_configurations": 6,
                         "actual_executions": 2, "reused_measurements": 4, "setup_errors": 0,
                         "classical_reference_executions": 0})
        source, reused = data["series"][0]["samples"][0], data["series"][1]["samples"][0]
        self.assertEqual(reused["measurement"]["kind"], "reused")
        self.assertEqual(reused["measurement"]["source_cutoff"], 1)
        self.assertEqual(source["observation"], reused["observation"])
        html = visual.render_html(data, title="Reuse", source_name="test.json", source_hash="abc", generated="today")
        for phrase in ("2 actual executions", "4 reused measurements", "Measurement source", "Reused from cutoff 1"):
            self.assertIn(phrase, html)
        reused["observation"]["counts"]["field_multiplications"] = -1
        reused["verification"]["failures"] = ("changed copy",)
        self.assertEqual(source["observation"]["counts"]["field_multiplications"], 8)
        self.assertEqual(source["verification"]["failures"], ())

    def test_recursive_cutoffs_with_different_leaves_remain_distinct(self):
        """Reuse Strassen cutoff 3 from 2 while executing leaf sizes 1, 2, and 4."""
        data = collect_work([4], [1, 2, 3, 4], algorithms=["strassen"])
        samples = [line["samples"][0] for line in data["series"]]
        self.assertEqual([s["measurement"]["kind"] for s in samples],
                         ["executed", "executed", "reused", "executed"])
        self.assertEqual(samples[2]["measurement"]["source_cutoff"], 2)
        self.assertEqual([s["observation"]["counts"]["field_multiplications"] for s in samples], [49, 56, 56, 64])

    def test_composition_uses_ordered_stages_and_keeps_each_prediction(self):
        """Known mixed-radix plans reuse equal leaves without dropping stage evidence."""
        plan = Composition(["strassen", "strassen-winograd"])
        data = collect_work([4], [1, 2, 3, 4], algorithms=["naive"], compositions={"mixed": plan})
        lines = [line for line in data["series"] if line["algorithm"] == "mixed"]
        self.assertEqual([line["samples"][0]["measurement"]["kind"] for line in lines],
                         ["executed", "executed", "reused", "executed"])
        self.assertEqual(lines[2]["samples"][0]["prediction"], lines[1]["samples"][0]["prediction"])

    def test_replaced_callbacks_are_never_assumed_equivalent(self):
        """A new callback can depend on cutoff even when its advertised model cannot."""
        calls = []
        def replacement(a, b, **kwargs):
            """Keep the output correct while exposing actual public dispatch calls."""
            calls.append(kwargs["cutoff"])
            return naive.multiply(a, b, **kwargs)
        with patch.dict(ALGORITHMS, {"naive": replacement}):
            data = collect_work([2], [1, 2, 4], algorithms=["naive"])
        self.assertEqual(calls, [1, 2, 4])
        self.assertEqual(data["execution_summary"]["reused_measurements"], 0)

    def test_failed_outputs_and_budget_mismatches_are_not_reused(self):
        """Only successful observations agreeing with independent budgets enter the cache."""
        from matrix_multiplication.instrumentation import count
        original = naive.classical
        for failure in ("output", "budget"):
            def inconsistent(a, b):
                """Inject an oracle failure or a counter mismatch beneath known dispatch."""
                result = original(a, b)
                if failure == "output":
                    result[0][0] += 1
                else:
                    count("field_multiplications")
                return result
            with self.subTest(failure=failure), patch("matrix_multiplication.naive.classical", side_effect=inconsistent) as kernel:
                data = collect_work([2], [1, 2, 4], algorithms=["naive"])
            self.assertEqual(kernel.call_count, 3)
            self.assertEqual(data["execution_summary"]["reused_measurements"], 0)

    def test_unfamiliar_plan_classes_execute_independently(self):
        """Custom plan classes may add cutoff-sensitive behavior beneath the same budget."""
        class UnfamiliarPlan(Composition):
            """Stand in for an extension whose behavior is not guaranteed by our traversal."""
        data = collect_work([2], [1, 2, 3], algorithms=["naive"],
                            compositions={"custom": UnfamiliarPlan(["strassen"])})
        lines = [line for line in data["series"] if line["algorithm"] == "custom"]
        self.assertTrue(all(line["samples"][0]["measurement"]["kind"] == "executed" for line in lines))

    @unittest.skipUnless(smirnov.DATA_PATH.exists(), "import Smirnov data first")
    @requires_coefficients("alphaevolve")
    def test_reuse_matches_independent_runs_including_rectangular_stages(self):
        """Small executions compare optimized and repeated counts, verification, and fits."""
        options = {"algorithms": ["naive", "alphaevolve", "smirnov"]}
        optimized = collect_work([2, 4], [1, 2, 4, 6], **options)
        independent = collect_work([2, 4], [1, 2, 4, 6], reuse_equivalent_cutoffs=False, **options)
        self.assertLess(optimized["execution_summary"]["actual_executions"], independent["execution_summary"]["actual_executions"])
        for line, baseline in zip(optimized["series"], independent["series"]):
            self.assertEqual(line["empirical_fits"], baseline["empirical_fits"])
            for sample, original in zip(line["samples"], baseline["samples"]):
                for key in ("observation", "verification", "prediction", "status",
                            "prediction_matches_observed_data_multiplications", "prediction_matches_observed_arithmetic_steps"):
                    self.assertEqual(sample[key], original[key], (line["algorithm"], key))
        self.assertEqual(independent["execution_summary"]["reused_measurements"], 0)

    def test_cli_can_repeat_equivalent_cutoffs_explicitly(self):
        """The independent-run option disables reuse without changing report generation."""
        with tempfile.TemporaryDirectory(dir=visual.ROOT) as directory, redirect_stdout(io.StringIO()):
            self.assertEqual(sweep.main(["--measurements-only", "--sizes", "2", "--cutoffs", "1", "2", "4", "--algorithms", "naive",
                                        "--repeat-equivalent-cutoffs", "--output-root", directory]), 0)
            folder = next(Path(directory).iterdir())
            data = json.loads((folder / "measurements.json").read_text(encoding="utf-8"))
        self.assertFalse(data["configuration"]["reuse_equivalent_cutoffs"])
        self.assertEqual(data["execution_summary"]["actual_executions"], 3)
        self.assertEqual(data["execution_summary"]["reused_measurements"], 0)

    @unittest.skipUnless(smirnov.DATA_PATH.exists(), "import Smirnov data first")
    @requires_coefficients("alphaevolve")
    def test_default_suite_includes_smirnov_with_rectangular_predictions(self):
        """The default executes all five, retaining exact stage shapes and counts."""
        data = collect_work([1, 2, 4], [1, 2])
        self.assertEqual({line["algorithm"] for line in data["series"]}, set(ALGORITHMS))
        self.assertIn("smirnov", ALGORITHMS)
        for line in data["series"]:
            for sample in line["samples"]:
                self.assertEqual(sample["status"], "passed")
                self.assertTrue(sample["prediction_matches_observed_data_multiplications"])
                self.assertTrue(sample["prediction_matches_observed_arithmetic_steps"])
            if line["algorithm"] == "smirnov":
                self.assertEqual(line["ideal_curve"], [])
                self.assertEqual(line["mathematical_model"]["base_dimension"], 54)
                self.assertEqual(line["samples"][1]["prediction"]["padded_dimensions"], [2, 2, 2])
                self.assertEqual(line["samples"][1]["size_effects"]["extra_zero_input_cells"], 0)
                self.assertEqual(prediction("smirnov", 4, line["cutoff"]), line["samples"][2]["prediction"])

    def test_missing_smirnov_data_remains_an_explicit_reported_error(self):
        """An unavailable recipe cannot drop an algorithm or break HTML generation."""
        message = "Smirnov data missing: run python tools/import_smirnov.py --download"
        with patch("matrix_multiplication.smirnov.plan", side_effect=FileNotFoundError(message)):
            data = collect_work([2], [1], algorithms=["naive", "smirnov"])
        line = data["series"][1]
        sample = line["samples"][0]
        self.assertEqual(sample["status"], "error")
        self.assertIn(message, sample["error"])
        self.assertEqual(sample["prediction"], {})
        self.assertIsNone(sample["prediction_matches_observed_arithmetic_steps"])
        html = visual.render_html(data, title="Missing data", source_name="test.json", source_hash="abc", generated="today")
        self.assertIn(message, html)
        self.assertIn("2.774", html)

    def test_predictions_include_padding_and_effective_cutoffs(self):
        """Exercise recurrence plateaus and leaf sizes independently of the evaluator."""
        self.assertEqual(prediction("strassen", 3, 1, size_policy="padded")["padded_data_multiplications"], 49)
        self.assertEqual(prediction("strassen", 3, 2, size_policy="padded")["padded_data_multiplications"], 56)
        self.assertEqual(prediction("alphaevolve", 8, 3, size_policy="padded")["effective_leaf_dimension"], 1)
        self.assertEqual(prediction("alphaevolve", 8, 4, size_policy="padded")["padded_data_multiplications"], 3072)
        self.assertEqual(prediction("alphaevolve", 8, 1, size_policy="padded")["padded_data_multiplications"],
                         prediction("alphaevolve", 16, 1, size_policy="padded")["padded_data_multiplications"])
        self.assertAlmostEqual(prediction("strassen", 4, 1)["ideal_data_multiplications"], 49)

    def test_observed_growth_matches_executed_classical_and_strassen_work(self):
        """Captured counters follow known budgets through different actual leaf sizes."""
        data = collect_work([1, 2, 3, 4], [1, 2], algorithms=["naive", "strassen"])
        lines = {(line["algorithm"], line["cutoff"]): line for line in data["series"]}
        samples = lines[("strassen", 1)]["samples"]
        self.assertEqual([sample["observation"]["counts"]["field_multiplications"] for sample in samples], [1, 7, 26, 49])
        self.assertEqual([sample["observation"]["counts"]["field_multiplications"] for sample in lines[("strassen", 2)]["samples"]], [1, 8, 26, 56])
        self.assertAlmostEqual(lines[("naive", 1)]["empirical_fits"]["field_multiplications"]["slope"], 3)
        for line in lines.values():
            for sample in line["samples"]:
                self.assertEqual(sample["status"], "passed")
                self.assertTrue(sample["prediction_matches_observed_data_multiplications"])
                self.assertTrue(sample["prediction_matches_observed_arithmetic_steps"])
                self.assertTrue(all(profile["inclusive_seconds"] == 0 for profile in sample["observation"]["profiles"].values()))
                counts = sample["observation"]["counts"]
                expected = sum(counts.get(key, 0) for key in ("field_multiplications", "coefficient_multiplications", "field_additions", "field_subtractions", "field_normalizations"))
                self.assertEqual(sample["observation"]["field_arithmetic_operations"], expected)
        self.assertEqual(samples[2]["input_repr_sha256"], lines[("naive", 1)]["samples"][2]["input_repr_sha256"])

    @requires_coefficients("alphaevolve")
    def test_full_recurrences_match_gaussian_and_winograd_execution(self):
        """Verify coefficient/normalization costs and complex component expansion."""
        data = collect_work([1, 4], [1, 4], algorithms=["alphaevolve", "strassen-winograd"])
        for line in data["series"]:
            for sample in line["samples"]:
                self.assertEqual(sample["status"], "passed")
                self.assertTrue(sample["prediction_matches_observed_arithmetic_steps"])
                for layer in ("field", "atomic"):
                    self.assertAlmostEqual(sample["prediction"][f"ideal_{layer}_arithmetic_operations"],
                                           sample["prediction"][f"padded_{layer}_arithmetic_operations"])
        sample = data["series"][0]["samples"][1]
        self.assertEqual(sample["observation"]["field_arithmetic_operations"], 2864)
        self.assertEqual(sample["observation"]["atomic_arithmetic_operations"], 11424)

    def test_fits_are_measured_range_estimates(self):
        """Recover known synthetic slopes and handle one-point/constant workloads."""
        fit = log_log_fit([(2, 24), (4, 192), (8, 1536)])
        self.assertAlmostEqual(fit["slope"], 3)
        self.assertAlmostEqual(fit["r_squared"], 1)
        self.assertEqual(log_log_fit([(2, 7)])["status"], "insufficient_points")
        plateau = log_log_fit([(2, 48), (4, 48)])
        self.assertEqual(plateau["slope"], 0)
        self.assertIsNone(plateau["r_squared"])

    def test_inaccurate_executions_are_retained_and_excluded_from_fits(self):
        """A failed result cannot appear as successful growth evidence."""
        def wrong(a, b, **kwargs):
            """Execute real counted work, then corrupt one output for the oracle."""
            result = naive.multiply(a, b, **kwargs)
            result[0][0] += 1
            return result

        with patch.dict("matrix_multiplication.work_sweep.ALGORITHMS", {"naive": wrong}):
            data = collect_work([1, 2], [1], algorithms=["naive"])
        line = data["series"][0]
        self.assertTrue(all(sample["status"] == "failed" for sample in line["samples"]))
        self.assertEqual(line["empirical_fits"]["field_multiplications"]["status"], "insufficient_points")
        self.assertEqual(line["samples"][1]["observation"]["counts"]["field_multiplications"], 8)

    def test_sweep_visualization_labels_measured_and_predicted_counts(self):
        """The plotted report exposes curve meanings and exact sampled evidence."""
        data = collect_work([1, 2, 4], [1, 2], algorithms=["strassen"])
        decoded = visual.read_reports(json.dumps(data))
        html = visual.render_html(decoded, title="Sweep", source_name="test.json", source_hash="abc", generated="today")
        for phrase in ("Wall-clock time is not plotted", "ideal model", "execution-policy prediction", "Log-log view",
                       "Measured multiplication slope", "Data multiplications", "Coefficient multiplications",
                       "Exact sampled counts", "not proofs of an asymptotic exponent",
                       "Square arithmetic exponents", "Arithmetic work breakdown", "Subtractions", "Normalizations",
                       "Data products by recursion depth", "Primitive call counts", "Primitive profiles",
                       "Accuracy and verification evidence", "All recorded operation and structural counters"):
            self.assertIn(phrase, html)
        self.assertEqual(html.count('<script>'), 1)
        self.assertIn('data-range-chart=', html)
        report = visual.sweep_sample_report(data["series"][0], data["series"][0]["samples"][1])
        self.assertEqual(report["counts"]["field_multiplications"], 7)
        self.assertEqual(report["execution_measurements"]["accuracy"]["maximum_absolute_error"], 0)
        self.assertEqual(report["execution_measurements"]["elapsed_algorithm_call_seconds"],
                         data["series"][0]["samples"][1]["execution_measurements"]["elapsed_algorithm_call_seconds"])
        self.assertEqual(report["execution_measurements"]["python_traced_peak_bytes"]["status"], "measured")

    def test_reference_report_is_preserved_and_separated_from_sweep_measurements(self):
        """Earlier diagnostics retain exact source text, scope, and measured values."""
        with tempfile.TemporaryDirectory(dir=visual.ROOT) as directory:
            root = Path(directory)
            source = root / "earlier.json"
            earlier = {"reports": {"naive": {"status": "passed", "counts": {"field_multiplications": 8},
                       "execution_measurements": {"elapsed_algorithm_call_seconds": 123.456,
                           "python_traced_peak_bytes": {"status": "measured", "value": 54321}},
                       "experiment": {"input_shapes": [[2, 2], [2, 2]]}}},
                       "events": {"naive": [{"value": "<script>unsafe</script>"}]}}
            raw = json.dumps(earlier, indent=4) + "\n"
            source.write_bytes(raw.encode("utf-8"))
            with redirect_stdout(io.StringIO()):
                status = sweep.main(["--sizes", "1", "--cutoffs", "1", "--algorithms", "naive",
                                     "--reference-report", str(source), "--output-root", str(root / "outputs")])
            self.assertEqual(status, 0)
            folder = next((root / "outputs").iterdir())
            combined = visual.read_reports((folder / "measurements.json").read_bytes())
            self.assertEqual(combined["reference_reports"][0]["source_json"], raw)
            self.assertEqual(source.read_bytes(), raw.encode("utf-8"))
            html = (folder / "index.html").read_text(encoding="utf-8")
            for phrase in ("Earlier single-pair experiment", "separate input pair", "Secondary · elapsed time",
                           "Secondary · Python allocation peak", "123.456", "54,321", "Original earlier JSON snapshot"):
                self.assertIn(phrase, html)
            self.assertNotIn("<script>unsafe", html)
            self.assertIn("&lt;script&gt;unsafe", html)
            self.assertNotIn("execution_measurements", combined["series"][0]["samples"][0]["observation"])
            nested = root / "nested.json"
            nested.write_text(json.dumps(collect_work([1], [1], algorithms=["naive"])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not nested sweeps"):
                visual.attach_reference_reports(json.dumps(combined), [nested])

    def test_cli_writes_one_dated_report_folder(self):
        """A complete small experiment preserves its JSON beside its offline graphs."""
        with tempfile.TemporaryDirectory(dir=visual.ROOT) as directory, redirect_stdout(io.StringIO()):
            status = sweep.main(["--sizes", "1", "2", "--cutoffs", "1", "--algorithms", "naive",
                                 "--output-root", directory])
            self.assertEqual(status, 0)
            folders = list(Path(directory).iterdir())
            self.assertEqual(len(folders), 1)
            self.assertTrue((folders[0] / "index.html").exists())
            data = json.loads((folders[0] / "measurements.json").read_text(encoding="utf-8"))
            self.assertEqual(data["configuration"]["sizes"], [1, 2])
        with self.assertRaises(ValueError):
            collect_work([0], [1])


if __name__ == "__main__":
    unittest.main()
