"""Check report semantics and isolation of secondary execution diagnostics.

Metrics must not change arithmetic counts, count the correctness oracle as part
of execution, borrow an existing tracer's peak, or invent hardware observations.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest

import explore
from matrix_multiplication.experiments import ExecutionMeasurement


class ExperimentMeasurementsTests(unittest.TestCase):
    """Exercise successful, inaccurate, and failed experiments through public flows."""

    def run_report(self, *arguments):
        """Run a CLI experiment in a checkout-local temporary directory."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "experiment.json"
            output_root = Path(directory) / "outputs"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = explore.main([*arguments, "--report", str(path), "--output-root", str(output_root)])
            folders = list(output_root.iterdir())
            self.assertEqual(len(folders), 1)
            self.assertTrue((folders[0] / "index.html").exists())
            self.assertEqual((folders[0] / "measurements.json").read_bytes(), path.read_bytes())
            return status, json.loads(path.read_text(encoding="utf-8"))

    def test_reports_separate_exponent_counts_and_execution(self):
        """Optional diagnostics retain product counts and expose their actual scope."""
        status, reports = self.run_report("--algorithm", "strassen", "--counts-only", "--measure-memory")
        self.assertEqual(status, 0)
        report = reports["strassen"]
        self.assertEqual(report["counts"]["field_multiplications"], 7)
        model = report["mathematical_model"]
        self.assertEqual(model["base_dimension"], 2)
        self.assertEqual(model["base_bilinear_rank"], 7)
        self.assertAlmostEqual(model["square_arithmetic_exponent"], 2.807354922057604)
        self.assertFalse(report["experiment"]["primitive_timing_enabled"])
        self.assertTrue(all(entry["inclusive_seconds"] == 0 for entry in report["profiles"].values()))
        execution = report["execution_measurements"]
        self.assertGreater(execution["elapsed_algorithm_call_seconds"], 0)
        self.assertEqual(execution["accuracy"]["maximum_absolute_error"], 0)
        self.assertEqual(execution["python_traced_peak_bytes"]["status"], "measured")
        self.assertGreater(execution["python_traced_peak_bytes"]["value"], 0)
        self.assertFalse(execution["equivalent_rate_is_hardware_throughput"])
        self.assertEqual(execution["hardware_memory_traffic_bytes"], {"status": "not_measured", "value": None})
        self.assertFalse(tracemalloc.is_tracing())

    def test_inaccurate_and_invalid_runs_keep_diagnostics(self):
        """Nonfinite output errors stay strict JSON; malformed inputs still time attempts."""
        status, reports = self.run_report("--algorithm", "naive", "--a", "[[1e308]]", "--b", "[[1e308]]")
        self.assertEqual(status, 1)
        self.assertEqual(reports["naive"]["status"], "failed")
        self.assertEqual(reports["naive"]["execution_measurements"]["accuracy"]["maximum_absolute_error"], "inf")
        status, reports = self.run_report("--algorithm", "naive", "--a", "[[1,2]]", "--b", "[[1]]", "--measure-memory")
        self.assertEqual(status, 1)
        execution = reports["naive"]["execution_measurements"]
        self.assertEqual(execution["accuracy"]["status"], "unavailable")
        self.assertIsNotNone(execution["elapsed_algorithm_call_seconds"])
        self.assertFalse(tracemalloc.is_tracing())
        status, reports = self.run_report("--algorithm", "naive", "--atol", "nan")
        self.assertEqual(status, 1)
        self.assertEqual(reports["naive"]["experiment"]["atol"], "nan")

    def test_memory_probe_excludes_later_work_and_restores_on_error(self):
        """Work after the algorithm call cannot change its captured allocation peak."""
        measurement = ExecutionMeasurement(measure_memory=True)
        with measurement:
            allocation = bytearray(100_000)
        before = measurement.snapshot()
        later_allocation = bytearray(1_000_000)
        self.assertEqual(measurement.snapshot(), before)
        self.assertGreaterEqual(before["python_traced_peak_bytes"]["value"], len(allocation))
        self.assertLess(before["python_traced_peak_bytes"]["value"], len(later_allocation))
        with self.assertRaises(ArithmeticError):
            with ExecutionMeasurement(measure_memory=True):
                raise ArithmeticError("failed attempt")
        self.assertFalse(tracemalloc.is_tracing())

    def test_existing_memory_tracer_is_left_untouched(self):
        """Do not reset or stop a caller's probe, or report its historical peak."""
        measurement = ExecutionMeasurement(measure_memory=True)
        with measurement:
            pass
        tracemalloc.start()
        try:
            earlier_allocation = bytearray(100_000)
            with measurement:
                self.assertTrue(tracemalloc.is_tracing())
            self.assertTrue(tracemalloc.is_tracing())
            self.assertEqual(measurement.snapshot()["python_traced_peak_bytes"],
                             {"status": "unavailable_existing_tracer", "value": None})
            self.assertGreaterEqual(tracemalloc.get_traced_memory()[1], len(earlier_allocation))
        finally:
            tracemalloc.stop()


if __name__ == "__main__":
    unittest.main()
