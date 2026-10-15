"""Check reusable interior extraction, paid recovery, recombination and misses."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matrix_multiplication import certificate_checker
from matrix_multiplication.composition import Composition
from matrix_multiplication.constructions import ConstructionRequest, main, run_experiment, tensor_document
from matrix_multiplication.cw_interior import explore_interiors, powered_selections
from matrix_multiplication.recipe_parts import load_parts, recombine_parts, tensor_from_document
from matrix_multiplication.tensor_restrictions import (
    RestrictionWitness, compose_selections, product_witness, recover_restriction,
    search_restrictions, verify_restriction)
from matrix_multiplication.tensors import Tensor, classical, tensor_product
from matrix_multiplication.verification import check_product


class RestrictionTests(unittest.TestCase):
    """Protect exact map semantics, independent checking and explicit budgets."""

    def test_forged_maps_missing_terms_and_nonunit_coefficients_fail(self):
        """A plausible shape or saved witness cannot conceal a false identity."""
        source = classical((2, 2, 2))
        witness = RestrictionWitness(source.dimensions, (2, 2, 2), ((0, 1, 2, 3),)*3)
        self.assertEqual(verify_restriction(source, witness).expand(), source.expand())
        forged = RestrictionWitness(source.dimensions, (2, 2, 2), ((1, 0, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)))
        for bad_source, bad_witness in ((source, forged), (Tensor(source.dimensions, source.terms[:-1]), witness),
                (Tensor(source.dimensions, ((((0, 2),), source.terms[0][1], source.terms[0][2]),)+source.terms[1:]), witness)):
            with self.assertRaises(ValueError):
                verify_restriction(bad_source, bad_witness)
        for selections in (((0, 0, 2, 3),)*3, ((True, 1, 2, 3),)*3, ((0, 1, 2, 4),)*3):
            with self.assertRaises(ValueError):
                RestrictionWitness(source.dimensions, (2, 2, 2), selections)

    def test_nested_and_product_maps_preserve_nontrivial_matrix_order(self):
        """Source pairing and row-major matrix pairing require different orders."""
        left, right = classical((2, 1, 3)), classical((1, 2, 2))
        a = RestrictionWitness(left.dimensions, (2, 1, 3), tuple(tuple(range(d)) for d in left.dimensions))
        b = RestrictionWitness(right.dimensions, (1, 2, 2), tuple(tuple(range(d)) for d in right.dimensions))
        witness = product_witness(a, b)
        restricted = verify_restriction(tensor_product(left, right), witness)
        self.assertEqual(witness.shape, (2, 2, 6))
        self.assertEqual(restricted.expand(), classical(witness.shape).expand())
        self.assertEqual(compose_selections(((5, 2), (8, 4), (1, 7)), ((1,), (0,), (1,))), ((2,), (8,), (7,)))
        for inner in (((2,), (0,), (1,)), ((True,), (0,), (1,))):
            with self.assertRaises(ValueError):
                compose_selections(((5, 2), (8, 4), (1, 7)), inner)
        self.assertEqual(powered_selections(((1, 3),)*3, 4, 2), ((5, 7, 13, 15),)*3)

    def test_search_exhaustion_and_saved_part_limits_are_separate(self):
        """An exhausted coordinate family is narrower than a universal exclusion."""
        source = classical((1, 1, 2))
        parts, evidence = search_restrictions(source, candidate_budget=20, max_parts=1)
        self.assertEqual(len(parts), 1)
        self.assertTrue(evidence['candidate_family_exhausted'])
        self.assertFalse(evidence['budget_exhausted'])
        parts, evidence = search_restrictions(source, candidate_budget=0)
        self.assertEqual(parts, ())
        self.assertEqual(evidence['attempted_candidates'], 0)
        self.assertTrue(evidence['budget_exhausted'])
        for budget in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                search_restrictions(source, candidate_budget=budget)

    def test_wrong_polynomial_recovery_cannot_inherit_literal_success(self):
        """A correct source restriction does not excuse a corrupt polynomial recipe."""
        source = classical((1, 1, 2))
        witness = RestrictionWitness(source.dimensions, (1, 1, 2), ((0,), (0, 1), (0, 1)))
        with self.assertRaises(ValueError):
            recover_restriction(source, witness, (), 0)


class InteriorExperimentTests(unittest.TestCase):
    """Exercise real search, recovery, library reload and recursive integration."""

    @classmethod
    def setUpClass(cls):
        """Run one actual bounded experiment shared by independent assertions."""
        cls.parts, cls.report = explore_interiors()

    def test_all_orientations_recover_lift_and_recombine_without_rank_claim(self):
        """The known parts have classical cost; squared search misses remain visible."""
        self.assertEqual(self.report['unrecognized_sectors'], [[1, 1, 2], [1, 2, 1], [2, 1, 1]])
        base_shapes = set()
        for tensor, evidence in self.parts:
            shape = tuple(evidence['shape'])
            self.assertEqual(tensor.expand(), classical(shape).expand())
            self.assertEqual(tensor.rank, evidence['classical_products'])
            self.assertEqual(evidence['execution']['observation']['counts']['field_multiplications'], tensor.rank)
            self.assertTrue(evidence['execution']['exact_outputs_verified'])
            if evidence.get('component_power') == 1:
                base_shapes.add(shape)
                recovery = evidence['recovery']
                self.assertTrue(recovery['complete_matrix_identity_verified'])
                self.assertTrue(recovery['replay_verified'])
                self.assertGreaterEqual(recovery['interpolation_products'], tensor.rank)
        self.assertEqual(base_shapes, {(2, 1, 2), (1, 2, 2), (2, 2, 1)})
        for entry in self.report['experiments']:
            search = entry['search']
            self.assertEqual(search['attempted_candidates'], 2048)
            self.assertEqual(len(search['candidates']), 2048)
            self.assertTrue(search['budget_exhausted'])
            if entry['component_power'] == 2:
                self.assertEqual(search['recognized_candidates'], 0)
                self.assertEqual(entry['constructively_lifted_base_witnesses'], 1)
        tensor, evidence = self.parts[-1]
        self.assertEqual((tensor.rank, evidence['shape']), (64, [4, 4, 4]))

    def test_materialization_limits_and_empty_catalog_keep_honest_outcomes(self):
        """Preflight stops oversized sources and labels skipped powers explicitly."""
        with patch('matrix_multiplication.cw_interior.coarse_catalog') as catalog:
            with self.assertRaises(ValueError):
                explore_interiors(q=2, power=10)
            catalog.assert_not_called()
        parts, report = explore_interiors(candidate_budget=0, max_source_dimension=16)
        self.assertFalse(parts)
        self.assertEqual(sum(e['status'].startswith('skipped') for e in report['experiments']), 3)
        parts, report = explore_interiors(q=1, power=1)
        self.assertEqual((parts, report['experiments']), ((), []))

    def test_library_reloads_exact_parts_and_rejects_corruption(self):
        """Round-tripped terms can recombine; altered coefficients cannot pass flags."""
        document = {'schema': 'exact-recipe-parts-v1', 'provenance': {'license': 'GPL-3.0'},
                    'parts': [{'tensor': tensor_document(t), 'evidence': d} for t, d in self.parts]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'parts.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            loaded, provenance = load_parts(path)
            self.assertEqual([t for t, _ in loaded], [t for t, _ in self.parts])
            self.assertEqual([d for _, d in loaded], json.loads(json.dumps([d for _, d in self.parts])))
            self.assertEqual(provenance, document['provenance'])
            first, second = loaded[0], next(p for p in loaded if p[1]['shape'] == [1, 2, 2])
            tensor, shape = recombine_parts(first[0], first[1]['shape'], second[0], second[1]['shape'])
            self.assertEqual(tensor.expand(), classical(shape).expand())
            for value in ([2, 1], [1.0, 1], [1, 0], [True, 1]):
                corrupt = deepcopy(document)
                corrupt['parts'][0]['tensor']['terms'][0][0][0][1] = value
                path.write_text(json.dumps(corrupt), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_parts(path)
            path.write_text('{"schema":"bad","schema":"exact-recipe-parts-v1"}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_parts(path)
        tensor = tensor_from_document(tensor_document(self.parts[0][0]))
        self.assertEqual(tensor, self.parts[0][0])

    def test_cli_artifacts_certificate_and_actual_recursion(self):
        """One command produces reusable parts, reader explanations and a real scheme."""
        with tempfile.TemporaryDirectory() as temp:
            with patch('matrix_multiplication.constructions.explore_interiors', return_value=(self.parts, self.report)):
                folder, report = run_experiment(ConstructionRequest('cw-interior', power=2, certificate=True), Path(temp) / 'run')
            loaded, _ = load_parts(folder / 'parts.json')
            self.assertEqual([t for t, _ in loaded], [t for t, _ in self.parts])
            self.assertEqual(report['provenance']['source_parameters']['q'], 2)
            certificate_checker.check_recipe(json.loads((folder / 'scheme.json').read_text()))
            self.assertEqual(report['square_recursion']['exponent'], 3)
            self.assertTrue((folder / 'certificate' / 'certificate.json').exists())
            plan = Composition.from_spec(str(folder / 'scheme.json'))
            for size in (4, 5, 16):
                a = [[Fraction(i-j, 3) for j in range(size)] for i in range(size)]
                b = [[Fraction(i+j, 5) for j in range(size)] for i in range(size)]
                self.assertTrue(check_product(a, b, plan.multiply(a, b)).passed)
            html = (folder / 'index.html').read_text(encoding='utf-8')
            for phrase in ('Reusable parts', 'separate source copies', 'Recovered data products', 'not an arithmetic saving'):
                self.assertIn(phrase, html)
            self.assertNotIn('"selected_coordinates"', html)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--output-root', temp, 'cw-interior', '--q', '1', '--power', '1']), 0)
            self.assertTrue((next(Path(temp).iterdir()) / 'parts.json').exists())
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['--output-root', temp, 'cw-interior', '--candidate-budget', '-1'])
            self.assertEqual(list(Path(temp).iterdir()), [])
            with self.assertRaisesRegex(ValueError, 'supported square recombination'):
                run_experiment(ConstructionRequest('cw-interior', q=1, power=1, certificate=True), Path(temp) / 'invalid')
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
