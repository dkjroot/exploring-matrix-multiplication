"""Check exact evidence behind educational explanations of graph steps.

Actual off-grid calls test plateaus and boundary changes. Independent descriptor
budgets exercise mixed/rectangular paths without pretending every stage has a
binary split. Unknown callbacks, failed models and legacy JSON remain explicit.
"""

import json
import unittest
from unittest.mock import patch

from matrix_multiplication import ALGORITHMS, naive
from matrix_multiplication.composition import Composition
from matrix_multiplication.rectangular import RectangularCycle
from matrix_multiplication.reporting import render_html
from matrix_multiplication.size_effects import describe_sample
from matrix_multiplication.tensors import classical, to_scheme
from matrix_multiplication.work_sweep import collect_work, schedule_prediction


class SizeEffectsTests(unittest.TestCase):
    """Protect the distinction between formula constraints and padding choices."""

    def test_strassen_plateau_and_next_boundary_follow_padded_geometry(self):
        """Real calls at 4,5,8,9 expose the off-grid policy without time fitting."""
        data = collect_work([4, 5, 8, 9], [1], algorithms=['strassen'], measure_memory=False, projection_cycles=3,
                            size_policy='padded')
        samples = data['series'][0]['samples']
        self.assertEqual([s['observation']['counts']['field_multiplications'] for s in samples], [49, 343, 343, 2401])
        self.assertEqual([s['size_effects']['padded_dimensions'] for s in samples], [[4]*3, [8]*3, [8]*3, [16]*3])
        plateau = samples[2]['size_effects']['transition_from_previous_sample']
        self.assertEqual(plateau['kind'], 'padding_plateau')
        self.assertTrue(plateau['observed_data_products_equal'])
        boundary = samples[3]['size_effects']['transition_from_previous_sample']
        self.assertEqual(boundary['kind'], 'padded_geometry_change')
        self.assertFalse(boundary['observed_data_products_equal'])
        effect = samples[3]['size_effects']
        self.assertEqual([level['child_dimensions'] for level in effect['levels']], [[8]*3, [4]*3, [2]*3, [1]*3])
        self.assertFalse(effect['padding_is_universal_mathematical_requirement'])
        self.assertFalse(effect['volume_ratio_is_multiplication_overhead'])
        self.assertEqual(effect['padded_volume_ratio'], [4096, 729])

    def test_padding_before_cutoff_is_explicit_not_called_mathematically_necessary(self):
        """A request fitting cutoff three still splits after rounding up first."""
        data = collect_work([3, 6], [3], algorithms=['strassen'], measure_memory=False, projection_cycles=2,
                            size_policy='padded')
        first, second = data['series'][0]['samples']
        effect = first['size_effects']
        self.assertTrue(effect['requested_fits_cutoff'])
        self.assertFalse(effect['padded_fits_cutoff'])
        self.assertTrue(effect['padding_triggered_recursion'])
        self.assertEqual(effect['extra_zero_input_cells'], 14)
        self.assertEqual(first['observation']['counts']['field_multiplications'], 56)
        self.assertEqual(second['observation']['counts']['field_multiplications'], 392)
        self.assertFalse(second['size_effects']['padding_triggered_recursion'])
        html = render_html(data, title='Cutoff', source_name='test', source_hash='abc', generated='today')
        self.assertIn('already fits the cutoff', html)
        self.assertIn('not a mathematical requirement', html)

    def test_mixed_schedule_uses_prefixes_not_whole_cycle_padding(self):
        """A binary then radix-four cycle pads size three to eight, not two or four."""
        scheme = to_scheme(classical((4, 4, 4)), (4, 4, 4), 'classical-four')
        plan = Composition(['strassen', scheme])
        prediction = schedule_prediction(plan, 3, 1, size_policy='padded')
        effect = describe_sample('mixed', 3, 1, prediction, plan.model(), known_policy=True)
        self.assertEqual(effect['padded_dimensions'], [8]*3)
        self.assertEqual([stage['block_counts'] for stage in effect['levels']], [[2]*3, [4]*3])
        self.assertEqual([stage['child_dimensions'] for stage in effect['levels']], [[4]*3, [1]*3])
        self.assertEqual(effect['leaf_dimensions'], [1]*3)

    def test_rectangular_schedule_preserves_each_axis_and_partial_cycle(self):
        """A balanced cycle can stop before its square full-cycle radix is reached."""
        shapes = ((2, 2, 3), (2, 3, 2), (3, 2, 2))
        plan = RectangularCycle(tuple(to_scheme(classical(shape), shape, 'rectangular') for shape in shapes))
        prediction = schedule_prediction(plan, 3, 1, size_policy='padded')
        effect = describe_sample('rectangular', 3, 1, prediction, plan.model(), known_policy=True)
        self.assertEqual(plan.radix, 12)
        self.assertEqual(effect['padded_dimensions'], [4, 6, 6])
        self.assertEqual([stage['child_dimensions'] for stage in effect['levels']], [[2, 3, 2], [1, 1, 1]])
        self.assertEqual(effect['recursion_levels'], 2)
        self.assertEqual(effect['padded_volume_ratio'], [16, 3])

    def test_naive_does_not_attribute_steps_to_recursion_or_cutoff(self):
        """Schoolbook work follows requested dimensions and has no padded plateau."""
        data = collect_work([2, 3], [1, 4], algorithms=['naive'], measure_memory=False, projection_cycles=2)
        for line in data['series']:
            for sample in line['samples']:
                effect = sample['size_effects']
                self.assertEqual(effect['padding_policy'], 'none')
                self.assertEqual(effect['cutoff_policy'], 'ignored')
                self.assertEqual(effect['levels'], [])
                self.assertFalse(effect['padding_added'])
            self.assertEqual(line['samples'][1]['size_effects']['transition_from_previous_sample']['kind'], 'schoolbook_growth')

    def test_unknown_callback_gets_only_model_evidence_even_if_counts_match(self):
        """Matching totals alone cannot establish an unfamiliar traversal policy."""
        def wrapper(a, b, **kwargs):
            """Supply a distinct callback with the same numerical/count behavior."""
            return naive.multiply(a, b, **kwargs)

        with patch.dict(ALGORITHMS, {'naive': wrapper}):
            data = collect_work([2, 3], [1], algorithms=['naive'], measure_memory=False, projection_cycles=2)
        samples = data['series'][0]['samples']
        self.assertEqual(samples[0]['size_effects']['status'], 'declared_model_only')
        self.assertEqual(samples[1]['size_effects']['transition_from_previous_sample']['kind'], 'unattributed')
        self.assertIsNone(samples[0]['size_effects']['padding_and_cropping_are_exact'])
        self.assertEqual(describe_sample('missing', 2, 1, {}, {}),
                         {'status': 'unavailable', 'reason': 'no discrete budget was produced'})

    def test_report_keeps_educational_evidence_and_reads_legacy_json(self):
        """Current reports show maps/causes; old saved measurements are not rewritten."""
        data = collect_work([3, 4], [1], algorithms=['strassen'], measure_memory=False, projection_cycles=2)
        html = render_html(data, title='Steps', source_name='test', source_hash='abc', generated='today')
        for phrase in ('Why can the graphs have steps?', 'Recursion alone does not force a staircase',
                       'Where this algorithm’s steps come from', 'Extra zero input cells',
                       'Fringe data products', 'Dimension path', 'Stage path', 'not a Python/GPU/home-computer restriction'):
            self.assertIn(phrase, html)
        legacy = json.loads(json.dumps(data))
        for sample in legacy['series'][0]['samples']:
            sample.pop('size_effects')
        previous = render_html(legacy, title='Legacy', source_name='old', source_hash='abc', generated='today')
        self.assertIn('Measured field arithmetic steps', previous)
        self.assertNotIn('<summary>Where this algorithm’s steps come from</summary>', previous)


if __name__ == '__main__':
    unittest.main()
