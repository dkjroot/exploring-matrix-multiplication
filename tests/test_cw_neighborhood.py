"""Check exact seeded moves, independent sums, joint recovery and bounded misses."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matrix_multiplication.constructions import ConstructionRequest, main, run_experiment, tensor_document
from matrix_multiplication.cw_extraction import execution_evidence
from matrix_multiplication.cw_neighborhood import explore_neighborhoods
from matrix_multiplication.recipe_parts import load_parts
from matrix_multiplication.restriction_neighborhoods import (
    JointRestriction, check_joint, search_joints, search_neighborhood, subset_key, recover_joint)
from matrix_multiplication.tensor_restrictions import RestrictionWitness, recover_tensor_restriction
from matrix_multiplication.tensors import Tensor, classical, direct_sum


def constant_polynomial(tensor):
    """Lift actual rational source factors to constant polynomials for recovery tests."""
    return tuple(tuple(tuple((i, (c,)) for i, c in factor) for factor in term) for term in tensor.terms)


def two_independent_parts():
    """Build two dot-row products in disjoint spaces, with explicit original maps."""
    source = direct_sum(classical((1, 1, 2)), classical((1, 1, 2)))
    first = RestrictionWitness(source.dimensions, (1, 1, 2), ((0,), (0, 1), (0, 1)))
    second = RestrictionWitness(source.dimensions, (1, 1, 2), ((1,), (2, 3), (2, 3)))
    return source, first, second


class SeededMapTests(unittest.TestCase):
    """Protect complete identities and honest graph/pair budget boundaries."""

    def test_actual_neighbor_is_accepted_and_frontier_avoids_duplicates(self):
        """Two interchangeable input variables give a genuine successful edit."""
        source = Tensor((2, 2, 2), ((((0, 1), (1, 1)), ((0, 1),), ((0, 1),)),
                                     (((0, 1), (1, 1)), ((1, 1),), ((1, 1),))))
        seed = RestrictionWitness(source.dimensions, (1, 1, 2), ((0,), (0, 1), (0, 1)))
        witnesses, report = search_neighborhood(source, (seed,), candidate_budget=5)
        self.assertEqual(len(witnesses), 2)
        self.assertEqual(report['attempted_candidates'], 1)
        self.assertEqual(report['recognized_new_candidates'], 1)
        self.assertTrue(report['retained_frontier_exhausted'])
        self.assertNotEqual(subset_key(witnesses[0].selections), subset_key(witnesses[1].selections))
        witnesses, report = search_neighborhood(source, (seed,), candidate_budget=5, max_witnesses=1)
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(report['accepted_maps_not_retained'], 1)
        self.assertIn('not_retained_due_to_pool_cap', report['candidates'][0])
        witnesses, report = search_neighborhood(source, (seed,), candidate_budget=0)
        self.assertEqual(report['attempted_candidates'], 0)
        self.assertTrue(report['budget_exhausted'])

    def test_disjoint_spaces_still_reject_unwanted_cross_coefficients(self):
        """Each separate identity can pass while the union includes an extra term."""
        source, first, second = two_independent_parts()
        cross = (((0, Fraction(1, 3)),), ((2, 1),), ((2, 1),))
        contaminated = Tensor(source.dimensions, (*source.terms, cross))
        joint, evidence = check_joint(contaminated, first, second)
        self.assertIsNone(joint)
        self.assertEqual(evidence['shared_legs'], [])
        self.assertEqual(evidence['status'], 'unwanted cross coefficients')
        self.assertEqual(evidence['first_difference']['actual_minus_expected'], [1, 3])

    def test_dense_union_with_shared_input_space_has_its_own_comparator(self):
        """Two row blocks share B; their union is checked as a new dense target."""
        source = classical((2, 2, 2))
        first = RestrictionWitness(source.dimensions, (1, 2, 2), ((0, 1), (0, 1, 2, 3), (0, 1)))
        second = RestrictionWitness(source.dimensions, (1, 2, 2), ((2, 3), (0, 1, 2, 3), (2, 3)))
        joint, evidence = check_joint(source, first, second)
        self.assertEqual(joint.kind, 'dense matrix union')
        self.assertEqual(joint.shapes, ((2, 2, 2),))
        self.assertEqual(evidence['shared_legs'], [1])
        self.assertFalse(evidence['separate_cost_is_comparable'])
        recovered, recovery = recover_tensor_restriction(source, joint.selections,
            joint.target(), constant_polynomial(source), 0)
        self.assertEqual(recovered.expand(), source.expand())
        self.assertEqual(recovery['selected_products'], 8)
        _, _, comparison = recover_joint(source, joint, (first, second), constant_polynomial(source), 0)
        self.assertIsNone(comparison['separate_recovery_products'])
        self.assertEqual(comparison['same_target_classical_products'], 8)

    def test_independent_joint_recovery_pays_all_products_and_replays(self):
        """A positive joint path executes both verified matrix components exactly."""
        source, first, second = two_independent_parts()
        joint, evidence = check_joint(source, first, second)
        self.assertEqual(joint.kind, 'independent direct sum')
        self.assertTrue(evidence['separate_cost_is_comparable'])
        recovered, recovery = recover_tensor_restriction(source, joint.selections,
            joint.target(), constant_polynomial(source), 0)
        self.assertEqual(recovered.rank, 4)
        self.assertTrue(recovery['replay_verified'])
        execution = execution_evidence(recovered, tuple(SimpleNamespace(shape=s) for s in joint.shapes))
        self.assertTrue(execution['exact_outputs_verified'])
        self.assertEqual(execution['observation']['counts']['field_multiplications'], 4)
        _, _, comparison = recover_joint(source, joint, (first, second), constant_polynomial(source), 0)
        self.assertEqual(comparison['separate_recovery_products'], 4)
        self.assertFalse(comparison['joint_recovery_saves_products'])
        with self.assertRaisesRegex(ValueError, 'parent maps'):
            recover_joint(source, joint, (first, first), constant_polynomial(source), 0)
        with self.assertRaises(ValueError):
            recover_tensor_restriction(source, joint.selections, joint.target(), (), 0)
        bad = Tensor(source.dimensions, (*source.terms, (((0, 1),), ((2, 1),), ((2, 1),))))
        with self.assertRaisesRegex(ValueError, 'source restriction'):
            recover_tensor_restriction(bad, joint.selections, joint.target(), constant_polynomial(bad), 0)

    def test_pair_and_map_budgets_validate_before_acceptance(self):
        """Zero attempts, exhausted pools and forged seed maps remain distinct."""
        source, first, second = two_independent_parts()
        results, report = search_joints(source, (first, second), candidate_budget=0)
        self.assertFalse(results)
        self.assertTrue(report['budget_exhausted'])
        results, report = search_joints(source, (first, second), candidate_budget=1)
        self.assertEqual(len(results), 1)
        self.assertTrue(report['pair_pool_exhausted'])
        self.assertFalse(report['budget_exhausted'])
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                search_joints(source, (first, second), candidate_budget=value)
            with self.assertRaises(ValueError):
                search_neighborhood(source, (first,), candidate_budget=value)
        forged = RestrictionWitness(source.dimensions, (1, 1, 2), ((0,), (2, 3), (0, 1)))
        with self.assertRaises(ValueError):
            search_neighborhood(source, (forged,))
        with self.assertRaises(ValueError):
            JointRestriction(((0,), (0,), (0,)), ((1, 1, 1),), 'independent direct sum')

    def test_library_rechecks_direct_sum_and_rejects_ambiguous_shapes(self):
        """Saved flags cannot turn a cross term or conflicting shape into a sum."""
        source, _, _ = two_independent_parts()
        evidence = {'name': 'independent', 'component_shapes': [[1, 1, 2], [1, 1, 2]],
                    'products': 4, 'complete_identity_verified': True}
        library = {'schema': 'exact-recipe-parts-v1', 'provenance': {},
                   'parts': [{'tensor': tensor_document(source), 'evidence': evidence}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'parts.json'
            path.write_text(json.dumps(library), encoding='utf-8')
            parts, _ = load_parts(path)
            self.assertEqual(parts[0][0], source)
            bad = deepcopy(library)
            bad['parts'][0]['tensor']['terms'][0][1][0][0] = 2
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'independent-sum'):
                load_parts(path)
            bad = deepcopy(library)
            bad['parts'][0]['evidence']['shape'] = [1, 1, 2]
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                load_parts(path)


class CWNeighborhoodTests(unittest.TestCase):
    """Retain real negative CW evidence separately from correctness of the tooling."""

    @classmethod
    def setUpClass(cls):
        """Execute all three orientations and their squares once without mocks."""
        cls.parts, cls.report = explore_neighborhoods()

    def test_negative_result_has_exact_attempt_counts_and_paid_seed_controls(self):
        """These finite misses constrain only the declared graph and witness pool."""
        self.assertEqual(len(self.parts), 18)
        report = self.report
        self.assertEqual((report['new_verified_shapes'], report['strict_recovery_savings'], report['strict_classical_savings']), (0, 0, 0))
        self.assertEqual(sum(e['neighborhood']['attempted_candidates'] for e in report['experiments']), 5088)
        self.assertEqual(sum(e['joint_search']['attempted_candidates'] for e in report['experiments']), 21)
        for entry in report['experiments']:
            self.assertEqual(entry['seed_prefix']['attempted_candidates'], 2048)
            self.assertEqual(entry['neighborhood']['recognized_new_candidates'], 0)
            self.assertEqual(entry['joint_search']['recognized_candidates'], 0)
            self.assertTrue(entry['neighborhood']['retained_frontier_exhausted'])
            self.assertTrue(entry['joint_search']['pair_pool_exhausted'])
            self.assertEqual(len(entry['neighborhood']['candidates']), entry['neighborhood']['attempted_candidates'])
        for tensor, evidence in self.parts:
            self.assertEqual(tensor.rank, evidence['classical_products'])
            self.assertTrue(evidence['recovery']['replay_verified'])
            self.assertTrue(evidence['execution']['exact_outputs_verified'])

    def test_report_library_and_empty_cli_remain_reproducible(self):
        """The focused command saves raw misses and a checked library without a scheme claim."""
        with tempfile.TemporaryDirectory() as temp:
            with patch('matrix_multiplication.constructions.explore_neighborhoods', return_value=(self.parts, self.report)):
                folder, report = run_experiment(ConstructionRequest('cw-neighborhood', power=2), Path(temp) / 'run')
            parts, _ = load_parts(folder / 'parts.json')
            self.assertEqual([t for t, _ in parts], [t for t, _ in self.parts])
            self.assertFalse((folder / 'scheme.json').exists())
            self.assertNotIn('square_recursion', report)
            html = (folder / 'index.html').read_text(encoding='utf-8')
            for phrase in ('Seeded coordinate neighborhoods', 'Unused attempts', 'not an impossibility proof', 'joint polynomial restriction'):
                self.assertTrue(phrase in html, phrase)
            self.assertNotIn('"selected_coordinates"', html)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--output-root', temp, 'cw-neighborhood', '--q', '1', '--power', '1']), 0)
            folder = next(Path(temp).iterdir())
            self.assertEqual(json.loads((folder / 'parts.json').read_text(encoding='utf-8'))['parts'], [])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_experiment(ConstructionRequest('cw-neighborhood', neighborhood_budget=-1), Path(temp) / 'bad')
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
