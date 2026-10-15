"""Exercise the actual three-step extraction, rank reduction and recombination."""

from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matrix_multiplication import Instrumentation, certificate_checker
from matrix_multiplication.composition import Composition
from matrix_multiplication.constructions import ConstructionRequest, main, run_experiment
from matrix_multiplication.cw_support import explore_support
from matrix_multiplication.decomposition_search import replay
from matrix_multiplication.recipe_parts import load_parts
from matrix_multiplication.tensors import classical
from matrix_multiplication.verification import check_product


class CWSupportTests(unittest.TestCase):
    """Protect actual scientific outcomes and distinguish known progress from a record."""

    @classmethod
    def setUpClass(cls):
        """Run the real finite pipeline once, including a failed seed and exact success."""
        cls.parts, cls.report = explore_support()

    def test_three_steps_retain_caps_and_new_coordinates(self):
        """A pooled miss does not suppress a hit using previously omitted outputs."""
        r = self.report
        self.assertEqual(len(r['step_one']), 3)
        self.assertTrue(all(e['search']['recognized_candidates'] == 0 for e in r['step_one']))
        self.assertEqual(len(r['step_two']), 12)
        hits = [e for e in r['step_two'] if e['search']['recognized_candidates']]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['sector'], [1, 1, 2])
        self.assertEqual(hits[0]['shape'], [2, 2, 2])
        self.assertEqual(hits[0]['search']['recognized_candidates'], 8)
        for e in (*r['step_one'], *r['step_two']):
            search = e['search']
            self.assertEqual(search['attempted_candidates'], 4096)
            self.assertEqual(search['input_pair_screens']+search['layout_proposals'], 4096)
            self.assertTrue(search['exhaustion_unknown_at_budget'])
        first = self.parts[0][1]
        self.assertEqual(len(first['coordinates_outside_seed_pool'][2]), 4)
        self.assertEqual(first['recovery']['interpolation_products'], 432)
        self.assertEqual(first['products'], 8)

    def test_rank_walk_miss_success_and_recombination_are_exact(self):
        """Replay the retained successful path from its actual recovered eight products."""
        campaign = self.report['step_three']['campaigns'][0]
        self.assertEqual([run['seed'] for run in campaign['runs']], [0, 1])
        self.assertFalse(campaign['runs'][0]['target_reached'])
        self.assertEqual(campaign['runs'][0]['attempted_flips'], 10000)
        self.assertTrue(campaign['runs'][1]['target_reached'])
        self.assertEqual(campaign['runs'][1]['attempted_flips'], 1779)
        parent = self.parts[0][0]
        replayed = replay(parent, campaign['runs'][campaign['best_run_index']]['events'])
        self.assertEqual(replayed, self.parts[1][0])
        self.assertEqual(replayed.rank, 7)
        for tensor, evidence in self.parts:
            self.assertEqual(tensor.expand(), classical(evidence['shape']).expand())
            self.assertEqual(evidence['execution']['observation']['counts']['field_multiplications'], tensor.rank)
        self.assertEqual((self.parts[-1][0].rank, self.parts[-1][1]['shape']), (49, [4, 4, 4]))

    def test_actual_certificate_library_and_recursive_execution(self):
        """The exported recombination survives independent checks and multiple levels."""
        with tempfile.TemporaryDirectory() as temp:
            with patch('matrix_multiplication.constructions.explore_support', return_value=(self.parts, self.report)):
                folder, report = run_experiment(ConstructionRequest('cw-support', power=2, certificate=True), Path(temp) / 'run')
            self.assertFalse(report['square_recursion']['strictly_below_2_75'])
            self.assertEqual(report['square_recursion']['exponent_expression'], 'log_4(49)')
            loaded, _ = load_parts(folder / 'parts.json')
            self.assertEqual([t for t, _ in loaded], [t for t, _ in self.parts])
            certificate_checker.check_recipe(json.loads((folder / 'scheme.json').read_text(encoding='utf-8')))
            plan = Composition.from_spec(str(folder / 'scheme.json'))
            for size in (4, 5, 16):
                a = [[Fraction(i-j, 3) for j in range(size)] for i in range(size)]
                b = [[Fraction(i+j, 5) for j in range(size)] for i in range(size)]
                observer = Instrumentation(profile=False)
                result = plan.multiply(a, b, instrumentation=observer)
                self.assertTrue(check_product(a, b, result).passed)
                self.assertEqual(observer.snapshot()['field_arithmetic_operations'], plan.budget(size, size, size)['field_arithmetic_operations'])
            html = (folder / 'index.html').read_text(encoding='utf-8')
            for phrase in ('Three sequential support experiments', 'necessary conditions', 'not a new bound', 'not an exclusion'):
                self.assertTrue(phrase in html, phrase)
            self.assertNotIn('&quot;eligible_outputs&quot;', html)

    def test_no_walk_and_preflight_leave_honest_outcomes(self):
        """Zero rank steps preserve eight-product parts rather than inventing success."""
        parts, report = explore_support(steps=0)
        self.assertEqual([t.rank for t, _ in parts], [8, 8])
        self.assertFalse(report['step_three']['campaigns'][0]['improved'])
        self.assertEqual(report['step_three']['recombinations'], [])
        with patch('matrix_multiplication.cw_support.coarse_catalog') as catalog:
            with self.assertRaises(ValueError):
                explore_support(power=10)
            catalog.assert_not_called()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--output-root', temp, 'cw-support', '--q', '1', '--power', '1']), 0)
            self.assertTrue((next(Path(temp).iterdir()) / 'parts.json').exists())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_experiment(ConstructionRequest('cw-support', support_budget=-1), Path(temp) / 'bad')
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
