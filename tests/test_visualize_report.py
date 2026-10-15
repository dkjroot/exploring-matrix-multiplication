"""Verify report fidelity, missing-data handling, escaping, and output organization.

These tests use small report fixtures rather than re-running expensive algorithms.
The renderer must preserve evidence and make model/measurement differences visible.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import unittest
from xml.etree import ElementTree

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
                       "not a count of this algorithm’s actual work", "Primitive timing was disabled",
                       "Do not sum those layers", "Unavailable"):
            self.assertIn(phrase, html)
        self.assertIn('role="img"', html)
        self.assertIn('href="measurements.json"', html)
        self.assertNotIn("<script", html)
        self.assertNotIn('src="http', html)

    def test_all_chart_axes_name_the_quantity_and_logarithmic_scale(self):
        """Every main chart labels both axes, including depth and secondary bars."""
        html = self.render(self.fixture())
        charts = [ElementTree.fromstring(svg) for svg in re.findall(r'<svg\b.*?</svg>', html, re.DOTALL)]
        self.assertGreaterEqual(len(charts), 5)
        for chart in charts:
            labels = {node.attrib["data-axis"]: node.text for node in chart.findall("text") if "data-axis" in node.attrib}
            self.assertEqual(set(labels), {"x", "y"})
            self.assertTrue(all(labels.values()))
        self.assertIn("Algorithm and recursion depth", html)
        self.assertIn("Elapsed time (seconds)", html)
        self.assertIn("Python allocations (bytes)", html)
        curves = visual.curve_chart("Growth", [{"label": "Example", "color": 0, "points": [(1, 1), (2, 7)]}],
                                    logarithmic=True, y_label="Data multiplications (count)")
        chart = ElementTree.fromstring(re.search(r'<svg\b.*?</svg>', curves, re.DOTALL).group())
        labels = [node.text for node in chart.findall("text") if "data-axis" in node.attrib]
        self.assertEqual(len(labels), 2)
        self.assertTrue(all("log scale" in label for label in labels))

    def test_legends_group_related_models_and_do_not_rely_on_color_alone(self):
        """Isolation retains associated models; markers and legend text remain distinct."""
        series = [{"label": f"cutoff 1 {kind}", "color": 0, "kind": kind, "points": [(1, 1), (2, 7)]}
                  for kind in ("observed", "ideal", "prediction")]
        series.append({"label": "<unsafe> cutoff 2", "color": 1, "points": [(1, 1), (2, 7)]})
        html = visual.curve_chart("Growth", series)
        svg = ElementTree.fromstring(re.search(r'<svg\b.*?</svg>', html, re.DOTALL).group())
        groups = [node for node in svg.findall("g") if node.attrib.get("class") == "chart-series"]
        self.assertEqual([node.attrib["data-series-group"] for node in groups], ["g0", "g0", "g0", "g1"])
        self.assertEqual({node.attrib["data-series-kind"] for node in groups[:3]}, {"observed", "ideal", "prediction"})
        self.assertTrue(groups[0].findall(".//circle"))
        self.assertTrue(groups[3].findall(".//rect"))
        self.assertIn('stroke-dasharray="8 6"', html)
        self.assertIn(':has(.legend-button[data-series-group="g0"][aria-pressed="true"]:hover)', html)
        self.assertIn(':has(.legend-button[data-series-group="g0"][aria-pressed="true"]:focus-visible)', html)
        self.assertIn('.work-chart:not(:has(.legend-button:hover))', html)
        self.assertIn('visibility:hidden', html)
        self.assertEqual(html.count('class="legend-button"'), 4)
        self.assertIn('Some measured series have identical counts and overlap', html)
        self.assertNotIn('<unsafe>', html)
        self.assertIn('&lt;unsafe&gt;', html)
        self.assertNotIn('<script', html)
        self.assertIn('visibility:visible!important', visual.STYLE)

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
        self.assertIn("different inputs, cutoffs, size policies, tolerances, or profiling settings", html)
        self.assertIn("do not establish a fair comparison", html)

    def test_size_policy_difference_is_not_a_fair_run_comparison(self):
        """Matching inputs and cutoff cannot hide different boundary schedules."""
        reports = self.fixture()
        reports['other'] = json.loads(json.dumps(reports['strassen']))
        reports['strassen']['experiment']['size_policy'] = 'fringe'
        reports['other']['experiment']['size_policy'] = 'padded'
        self.assertIn('do not establish a fair comparison', self.render(reports))

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
