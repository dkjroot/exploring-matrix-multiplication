"""Verify report fidelity, missing-data handling, escaping, and output organization.

These tests use small report fixtures rather than re-running expensive algorithms.
The renderer must preserve evidence and make model/measurement differences visible.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from matrix_multiplication import reporting as visual


class VisualizationTests(unittest.TestCase):
    """Protect the report's mathematical meaning and portable offline output."""

    def fixture(self):
        """Provide one complete report with zero, absent, and measured diagnostics."""
        return {"strassen": {
            "status": "passed", "counts": {"field_multiplications": 7, "field_additions": 19},
            "products_by_depth": {"1": 7}, "atomic_arithmetic_operations": 32,
            "experiment": {"input_repr_sha256": "abc", "input_shapes": [[2, 2], [2, 2]],
                           "cutoff": 1, "atol": 1e-9, "rtol": 1e-9, "primitive_timing_enabled": False},
            "mathematical_model": {"square_arithmetic_exponent": 2.807354922057604,
                                   "recurrence": "T(n) = 7 T(n/2) + O(n^2)"},
            "execution_measurements": {"elapsed_algorithm_call_seconds": 0.01,
                                       "memory_tracking_enabled": False,
                                       "accuracy": {"mode": "exact equality", "maximum_absolute_error": 0},
                                       "python_traced_peak_bytes": {"status": "not_requested", "value": None}},
            "profiles": {"scalar.multiply": {"calls": 7, "inclusive_seconds": 0, "exclusive_seconds": 0}},
        }}

    def render(self, reports):
        """Render a deterministic in-memory document for semantic checks."""
        return visual.render_html(reports, title="Experiment", source_name="data.json",
                                  source_hash="abc", generated="2026-10-11T12:00:00+00:00")

    def test_model_work_and_secondary_scope_are_distinct(self):
        """Required qualifications stay next to plots and counts-only profiles."""
        html = self.render(self.fixture())
        for phrase in ("Square arithmetic exponents", "Observed data multiplications", "Data products by recursion depth",
                       "Secondary · elapsed time", "Secondary · Python allocation peak",
                       "not actual work or hardware FLOP/s", "Primitive timing was disabled",
                       "Do not sum those layers", "Unavailable"):
            self.assertIn(phrase, html)
        self.assertIn('role="img"', html)
        self.assertIn('href="measurements.json"', html)
        self.assertNotIn("<script", html)
        self.assertNotIn('src="http', html)

    def test_missing_values_are_not_zero_and_zero_is_not_missing(self):
        """Distinguish a real zero-width bar from absent or unusable numeric data."""
        svg = visual.bar_chart("Example", [("Zero", 0), ("Missing", None), ("Boolean", True)], "operations")
        self.assertEqual(svg.count("<rect "), 1)
        self.assertEqual(svg.count(">Unavailable</text>"), 2)
        self.assertIn("Zero: 0 operations", svg)
        legacy = {"naive": {"status": "passed", "counts": {"field_multiplications": 8}}}
        html = self.render(legacy)
        self.assertIn("Some experiment metadata is missing", html)
        self.assertNotIn("2.80735", html)
        self.assertIsNone(visual.arithmetic_count(legacy["naive"], "coefficient_multiplications"))
        legacy["naive"]["counter_convention"] = "omitted arithmetic categories have zero observed calls"
        self.assertEqual(visual.arithmetic_count(legacy["naive"], "coefficient_multiplications"), 0)

    def test_different_experiments_warn_without_ranking_a_winner(self):
        """Same-looking bars must not imply comparison of identical input experiments."""
        reports = self.fixture()
        reports["other"] = json.loads(json.dumps(reports["strassen"]))
        reports["other"]["experiment"]["input_repr_sha256"] = "different"
        html = self.render(reports)
        self.assertIn("different inputs, cutoffs, tolerances, or profiling settings", html)
        self.assertIn("do not establish a fair comparison", html)

    def test_failed_reports_and_data_labels_are_escaped(self):
        """Render errors and infinite display errors without active injected HTML."""
        label = '<script>alert("name")</script>'
        reports = {label: {"status": "error", "error": '<img src=x onerror="attack()">',
                           "execution_measurements": {"accuracy": {"maximum_absolute_error": "inf"}}}}
        html = self.render(reports)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)
        self.assertIn("inf", html)

    def test_dated_output_preserves_source_and_accepts_trace_envelopes(self):
        """Keep exact input bytes beside a standalone report, without overwriting runs."""
        with tempfile.TemporaryDirectory(dir=visual.ROOT) as directory:
            root = Path(directory)
            source = root / "trace.json"
            raw = json.dumps({"reports": self.fixture(), "events": {"strassen": [{"type": "enter"}]}}).encode()
            source.write_bytes(raw)
            timestamp = datetime(2026, 10, 11, 12, 34, 56, 123456, tzinfo=timezone.utc)
            destination = visual.write_report(source, output_root=root / "outputs", timestamp=timestamp)
            self.assertEqual(destination.parent.name, "2026-10-11_123456_123456")
            self.assertEqual((destination.parent / "measurements.json").read_bytes(), raw)
            self.assertIn("strassen", destination.read_text(encoding="utf-8"))
            self.assertEqual(source.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                visual.write_report(source, output_root=root / "outputs", timestamp=timestamp)

    def test_invalid_json_and_shapes_fail_before_creating_outputs(self):
        """Invalid report data fails before any dated output directory is created."""
        for raw in ("[]", "{}", '{"naive":{"counts":[]}}', '{"naive":{"status":"passed","counts":{"x":NaN}}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                visual.read_reports(raw)
        with tempfile.TemporaryDirectory(dir=visual.ROOT) as directory:
            source = Path(directory) / "bad.json"
            source.write_text("not json", encoding="utf-8")
            output = Path(directory) / "outputs"
            with self.assertRaises(ValueError):
                visual.write_report(source, output_root=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
