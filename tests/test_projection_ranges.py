"""Check exact discrete range budgets and safe, initially sampled graph controls."""

from html import unescape
import json
import re
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from matrix_multiplication import ALGORITHMS
from matrix_multiplication.composition import Composition
from matrix_multiplication.reporting import curve_chart, render_html
from matrix_multiplication.scaling import project_range
from matrix_multiplication.work_sweep import collect_work


class ProjectionRangeTests(unittest.TestCase):
    """Keep matrix execution, exact budget arithmetic and browser coordinates apart."""

    def test_off_grid_counts_and_boundary_neighbours_need_no_execution(self):
        """Predict fringes and natural powers rather than fitting their sampled slope."""
        with patch.dict(ALGORITHMS, {'strassen': lambda *a, **k: self.fail('matrix executed')}):
            projection = project_range('strassen', 1, [4, 8], extensions=12)
        points = {p['size']: p for p in projection['points']}
        for size, products in ((5, 110), (7, 309), (8, 343), (9, 560), (16, 2401)):
            self.assertEqual(points[size]['counts']['field_multiplications'], products)
        self.assertTrue({1023, 1024, 1025} <= points.keys())
        self.assertEqual(points[1024]['counts']['field_multiplications'], 7**10)
        self.assertEqual(projection['range_limits'][0], 8)
        self.assertEqual(projection['range_limits'][-1], 8*2**12)
        self.assertIn('no time', projection['resource_scope'])

    def test_both_size_policies_and_cutoffs_match_captured_counts(self):
        """The same range machinery covers the historical staircase and current fringes."""
        for policy in ('fringe', 'padded'):
            data = collect_work([3, 5, 8], [1, 3], algorithms=['naive', 'strassen-winograd'],
                                size_policy=policy, measure_memory=False, projection_cycles=2)
            for line in data['series']:
                predicted = {p['size']: p for p in line['range_projection']['points']}
                for sample in line['samples']:
                    point, observed = predicted[sample['size']], sample['observation']
                    for key, value in point['counts'].items():
                        self.assertEqual(value, observed['counts'].get(key, 0))
                    for key in ('field_arithmetic_operations', 'atomic_arithmetic_operations'):
                        self.assertEqual(point[key], observed[key])

    def test_mixed_schedule_uses_its_stage_budgets(self):
        """A repeating plan retains its own cutoff phase in unexecuted dimensions."""
        plan = Composition(['strassen', 'strassen-winograd'])
        projection = project_range('mixed', 3, [4, 9], plan, extensions=3)
        for point in projection['points']:
            budget = plan.budget(*(point['size'],)*3, cutoff=3)
            self.assertEqual(point['counts'], budget['category_counts'])
            self.assertEqual(point['atomic_arithmetic_operations'], budget['atomic_arithmetic_operations'])

    def test_slider_defaults_to_samples_and_preserves_huge_decimal_strings(self):
        """No future point sets initial axes; browser data cannot round exact counts."""
        huge = 7**40
        html = curve_chart('Range <test>', [
            {'label':'Measured', 'color':0, 'kind':'observed', 'points':[(2,7),(4,49)]},
            {'label':'Projection </script>', 'color':0, 'kind':'projected',
             'range_limits':[4,2**40], 'points':[(4,49),(2**40,huge)]}])
        payload = json.loads(unescape(re.search(r'data-range-chart="([^"]+)"', html).group(1)))
        self.assertEqual(payload['series'][1]['points'][-1], [str(2**40), str(huge)])
        self.assertIn('value="0"', html)
        self.assertNotIn('Projection </script>', html)
        svg = ElementTree.fromstring(re.search(r'<svg\b.*?</svg>', html, re.DOTALL).group())
        future = [g for g in svg.findall('g') if g.attrib.get('data-series-kind') == 'projected']
        self.assertEqual(len(future), 1)
        self.assertEqual(list(future[0]), [])

    def test_resource_bounds_extend_but_memory_and_legacy_reports_stay_measured(self):
        """Only declared upper bounds extend resource plots; legacy data is untouched."""
        data = collect_work([2,4], [1], algorithms=['strassen'], measure_memory=True, projection_cycles=2)
        html = render_html(data, title='Range', source_name='test', source_hash='abc', generated='today')
        self.assertEqual(html.count('<script>'), 1)
        self.assertIn('data-measured-boundary', html)
        self.assertIn('BigInt(value)', html)
        for heading in ('64-bit multiplication equivalents', '64-bit addition and subtraction equivalents', 'Observed operand growth'):
            section = next(s for s in html.split('<section class="panel">') if '<h2>'+heading+'</h2>' in s)
            self.assertIn('data-range-chart=', section)
            self.assertIn('proved upper bound', section)
            payload = json.loads(unescape(re.search(r'data-range-chart="([^"]+)"', section).group(1)))
            self.assertTrue(payload['bounded'])
            svg = ElementTree.fromstring(re.search(r'<svg\b.*?</svg>', section, re.DOTALL).group())
            self.assertTrue(all(not list(g) for g in svg.findall('g') if g.attrib.get('data-series-kind') == 'upper_bound'))
        memory = next(s for s in html.split('<section class="panel">') if '<h2>Measured Python memory growth</h2>' in s)
        self.assertNotIn('data-range-chart=', memory)
        self.assertIn('deliberately no extrapolation slider', memory)
        self.assertTrue(all(s['resource_bounds_check_passed'] for s in data['series'][0]['samples']))
        for line in data['series']:
            line.pop('range_projection')
            line.pop('resource_bounds')
        legacy = render_html(data, title='Old', source_name='old', source_hash='abc', generated='today')
        self.assertNotIn('data-range-chart=', legacy)
        self.assertEqual(legacy.count('<script>'), 1)
        self.assertIn('Select none', legacy)

    def test_legend_entries_have_independent_persistent_line_controls(self):
        """Related measured/model curves share a hover group but separate toggles."""
        html = curve_chart('Choices', [
            {'label':'Measured', 'color':0, 'points':[(2,7),(4,49)]},
            {'label':'Bound', 'color':0, 'kind':'upper_bound', 'range_limits':[4,8], 'points':[(4,60),(8,400)]}])
        self.assertIn('Select all', html)
        self.assertIn('Select none', html)
        buttons = re.findall(r'<button[^>]+class="legend-button"[^>]+>', html)
        self.assertEqual(len(buttons), 2)
        self.assertIn('data-series-id="l0"', buttons[0])
        self.assertIn('data-series-id="l1"', buttons[1])
        self.assertTrue(all('data-series-group="g0"' in b and 'aria-pressed="true"' in b for b in buttons))

    def test_invalid_range_controls_fail_before_preparing_a_plan(self):
        """Reject booleans, empty samples and invalid cutoffs without setup work."""
        for sizes, extensions, cutoff in (([],1,1), ([True],1,1), ([2],True,1), ([2],0,1), ([2],1,0)):
            with self.assertRaises(ValueError):
                project_range('strassen', cutoff, sizes, extensions=extensions)


if __name__ == '__main__':
    unittest.main()
