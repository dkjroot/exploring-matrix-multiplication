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

from matrix_multiplication import naive
from matrix_multiplication.work_sweep import collect_work, log_log_fit, prediction
import sweep
from matrix_multiplication import reporting as visual


class WorkSweepTests(unittest.TestCase):
    """Check that predictions do not substitute for observed work or accuracy checks."""

    def test_predictions_include_padding_and_effective_cutoffs(self):
        """Exercise recurrence plateaus and leaf sizes independently of the evaluator."""
        self.assertEqual(prediction("strassen", 3, 1)["padded_data_multiplications"], 49)
        self.assertEqual(prediction("strassen", 3, 2)["padded_data_multiplications"], 56)
        self.assertEqual(prediction("alphaevolve", 8, 3)["effective_leaf_dimension"], 1)
        self.assertEqual(prediction("alphaevolve", 8, 4)["padded_data_multiplications"], 3072)
        self.assertEqual(prediction("alphaevolve", 8, 1)["padded_data_multiplications"],
                         prediction("alphaevolve", 16, 1)["padded_data_multiplications"])
        self.assertAlmostEqual(prediction("strassen", 4, 1)["ideal_data_multiplications"], 49)

    def test_observed_growth_matches_executed_classical_and_strassen_work(self):
        """Captured counters follow known budgets through different actual leaf sizes."""
        data = collect_work([1, 2, 3, 4], [1, 2], algorithms=["naive", "strassen"])
        lines = {(line["algorithm"], line["cutoff"]): line for line in data["series"]}
        samples = lines[("strassen", 1)]["samples"]
        self.assertEqual([sample["observation"]["counts"]["field_multiplications"] for sample in samples], [1, 7, 49, 49])
        self.assertEqual([sample["observation"]["counts"]["field_multiplications"] for sample in lines[("strassen", 2)]["samples"]], [1, 8, 56, 56])
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
        for phrase in ("Wall-clock time is not plotted", "ideal model", "padded prediction", "Log-log view",
                       "Measured multiplication slope", "Data multiplications", "Coefficient multiplications",
                       "Exact sampled counts", "not proofs of an asymptotic exponent",
                       "Square arithmetic exponents", "Arithmetic work breakdown", "Subtractions", "Normalizations",
                       "Data products by recursion depth", "Primitive call counts", "Primitive profiles",
                       "Accuracy and verification evidence", "All recorded operation and structural counters"):
            self.assertIn(phrase, html)
        self.assertNotIn("<script", html)
        report = visual.sweep_sample_report(data["series"][0], data["series"][0]["samples"][1])
        self.assertEqual(report["counts"]["field_multiplications"], 7)
        self.assertEqual(report["execution_measurements"]["accuracy"]["maximum_absolute_error"], 0)
        self.assertNotIn("elapsed_algorithm_call_seconds", report["execution_measurements"])

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
