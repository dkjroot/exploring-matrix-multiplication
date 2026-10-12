"""Verify category projections against executions and keep resources scoped.

Exact count projections are stage calculations, not extrapolated timings. Tiny
real matrices check their acceptance boundary; huge dimensions must be handled
without invoking a matrix multiplier. Memory/word plots preserve older reports.
"""

import json
import unittest
from unittest.mock import patch

from matrix_multiplication import ALGORITHMS
from matrix_multiplication.composition import Composition
from matrix_multiplication.experiments import arithmetic_model
from matrix_multiplication.reporting import render_html
from matrix_multiplication.scaling import CATEGORIES, category_budget, project_work
from matrix_multiplication.work_sweep import collect_work
from tests.support import requires_coefficients


class ScalingTests(unittest.TestCase):
    """Keep exact theoretical projections and captured execution evidence distinct."""

    def test_projection_never_executes_the_algorithm(self):
        """Compute enormous category budgets with integer arithmetic and no matrices."""
        with patch.dict(ALGORITHMS, {'strassen': lambda *args, **kwargs: self.fail('projection executed a matrix')}):
            projection = project_work('strassen', 1, arithmetic_model('strassen'), cycles=40)
        last = projection['points'][-1]
        self.assertEqual(last['size'], 2**40)
        self.assertEqual(last['counts']['field_multiplications'], 7**40)
        self.assertEqual(last['counts']['field_subtractions'], 6*(7**40-4**40)//3)
        self.assertEqual(last['counts']['field_additions'], 7**40+12*(7**40-4**40)//3)
        self.assertIn('not derived', projection['peak_memory_projection'])

    def test_categories_match_padding_cutoffs_and_mixed_stage_executions(self):
        """Separate add/sub counts verify shared formulas and repeated schedules."""
        mixed = Composition(['strassen', 'strassen-winograd'])
        data = collect_work([2, 3, 4], [1, 2], algorithms=['naive', 'strassen', 'strassen-winograd'],
                            compositions={'mixed': mixed}, projection_cycles=4)
        for line in data['series']:
            for sample in line['samples']:
                self.assertTrue(sample['category_prediction_matches_observed'])
                self.assertEqual(sample['execution_measurements']['python_traced_peak_bytes']['status'], 'measured')
                self.assertTrue(all(type(value) is int for value in sample['category_prediction'].values()))
        self.assertEqual(category_budget('strassen', 2, 1)['field_subtractions'], 6)
        self.assertEqual(category_budget('strassen-winograd', 2, 1)['field_subtractions'], 8)
        self.assertEqual(category_budget('naive', 2, 100)['field_subtractions'], 0)

    @requires_coefficients('alphaevolve', 'smirnov')
    def test_sparse_and_rectangular_categories_match_actual_work(self):
        """Count rectangular areas and exact complex normalization separately."""
        data = collect_work([2, 4], [1], algorithms=['alphaevolve', 'smirnov'], measure_memory=False,
                            projection_cycles=2)
        for line in data['series']:
            for sample in line['samples']:
                self.assertEqual(sample['status'], 'passed')
                self.assertTrue(sample['category_prediction_matches_observed'])
                self.assertEqual(sum(sample['category_prediction'].values()), sample['observation']['field_arithmetic_operations'])
                self.assertEqual(sample['execution_measurements']['python_traced_peak_bytes']['status'], 'not_requested')

    def test_reports_expose_models_memory_and_bits_without_losing_older_evidence(self):
        """New data renders readable scopes; legacy JSON can omit every new field."""
        data = collect_work([2, 4], [1, 2], algorithms=['naive', 'strassen'], projection_cycles=3)
        html = render_html(data, title='Scope', source_name='test', source_hash='abc', generated='today')
        for phrase in ('Measured additions', 'Measured subtractions', 'Observed operand growth',
                       '64-bit multiplication equivalents', 'Measured Python memory growth',
                       'Exact recurrence projections', 'These matrices were not executed', 'omega (ω)',
                       'Arithmetic work breakdown', 'All recorded operation and structural counters',
                       'Requested square dimension n', 'Projected field calls (count)'):
            self.assertIn(phrase, html)
        for line in data['series']:
            line.pop('scaling_projection')
            for sample in line['samples']:
                sample['observation'].pop('operand_costs')
                sample.pop('execution_measurements')
        legacy = render_html(json.loads(json.dumps(data)), title='Old', source_name='old', source_hash='abc', generated='today')
        self.assertIn('Arithmetic work breakdown', legacy)
        self.assertNotIn('Operand-size and word-model evidence', legacy)

    def test_invalid_projection_controls_are_rejected_before_execution(self):
        """Do not let bool or a nonpositive cycle count become a silent setting."""
        for cycles in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                collect_work([2], [1], algorithms=['naive'], projection_cycles=cycles)
        for size, cutoff in ((True, 1), (1, 0), (0, 1)):
            with self.assertRaises(ValueError):
                category_budget('naive', size, cutoff)


if __name__ == '__main__':
    unittest.main()
