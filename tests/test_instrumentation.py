"""Verify observation against algebraic operation budgets and independent results.

These tests guard the purpose of the refactor: tracing must preserve answers,
counts must expose recursion/padding/scaling, and failures must not leak observers.
"""

from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import explore
from matrix_multiplication import ALGORITHMS, Instrumentation, alphaevolve, multiply
from matrix_multiplication import scalars
from matrix_multiplication.instrumentation import session
from matrix_multiplication.verification import check_product
from tests.support import available_algorithms, requires_coefficients


class InstrumentationTests(unittest.TestCase):
    """Check mathematical budgets rather than mirroring implementation loops."""

    def test_classical_rectangular_budget(self):
        """A 2-by-3 times 3-by-4 classical product requires exactly 24 products."""
        observer = Instrumentation()
        a, b = [[1, 2, 3], [4, 5, 6]], [[1, 2, 3, 4]] * 3
        self.assertTrue(check_product(a, b, multiply(a, b, instrumentation=observer)).passed)
        self.assertEqual(observer.counts["field_multiplications"], 24)
        self.assertEqual(observer.counts["field_additions"], 24)
        self.assertEqual(observer.snapshot()["atomic_arithmetic_operations"], 48)
        self.assertEqual(observer.primitive_calls["vector.dot"], 8)

    def test_strassen_and_winograd_base_budgets(self):
        """Both use seven products; independently known block budgets are 18 and 15."""
        for name, block_adds, block_subs in [("strassen", 12, 6), ("strassen-winograd", 7, 8)]:
            with self.subTest(algorithm=name):
                observer = Instrumentation()
                result = multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]],
                                  algorithm=name, instrumentation=observer)
                self.assertEqual(result, [[19, 22], [43, 50]])
                self.assertEqual(observer.counts["field_multiplications"], 7)
                self.assertEqual(observer.counts["field_additions"], block_adds + 7)
                self.assertEqual(observer.counts["field_subtractions"], block_subs)
                self.assertEqual(observer.primitive_calls["block.add"], block_adds)
                self.assertEqual(observer.primitive_calls["block.subtract"], block_subs)
                self.assertEqual(observer.counts["recursive_nodes"], 8)
                self.assertEqual(observer.products_by_depth, {1: 7})

    def test_padding_recursion_and_cutoff_budgets(self):
        """Padding 3 to 4 exercises 7 squared leaves; cutoff 2 uses seven cubic leaves."""
        for cutoff, products, depth in [(1, 49, 2), (2, 56, 1), (4, 64, 0)]:
            observer = Instrumentation()
            a = [[1, 2, 3]] * 3
            multiply(a, a, algorithm="strassen", cutoff=cutoff, instrumentation=observer, size_policy="padded")
            self.assertEqual(observer.counts["field_multiplications"], products)
            self.assertEqual(observer.counts["padded_input_cells"], 32)
            self.assertEqual(observer.maximum_depth, depth)
        observer = Instrumentation()
        multiply([[0] * 4] * 4, [[0] * 4] * 4, algorithm="strassen", instrumentation=observer)
        self.assertEqual(observer.counts["field_multiplications"], 49)
        self.assertEqual(observer.counts["field_additions"], 181)
        self.assertEqual(observer.counts["field_subtractions"], 66)

    @requires_coefficients("alphaevolve")
    def test_alphaevolve_complex_expansion_and_coefficients(self):
        """Separate 48 data products from coefficient work and four-real expansions."""
        observer = Instrumentation()
        a = [[int(i == j) for j in range(4)] for i in range(4)]
        self.assertEqual(alphaevolve.multiply(a, a, instrumentation=observer), a)
        nonzeros = sum(len(column) for factor in alphaevolve._factors() for column in factor)
        self.assertEqual(observer.counts["field_multiplications"], 48)
        self.assertEqual(observer.counts["coefficient_multiplications"], nonzeros)
        self.assertEqual(observer.counts["real_multiplications"], 4 * (48 + nonzeros))
        self.assertEqual(observer.counts["real_additions"], 3 * nonzeros + 144)
        self.assertEqual(observer.counts["real_subtractions"], nonzeros + 48)
        self.assertEqual(observer.counts["field_normalizations"], 16)
        self.assertEqual(observer.counts["integer_divmods"], 32)

    @requires_coefficients("alphaevolve")
    def test_alphaevolve_two_level_product_budget(self):
        """Two radix-four levels retain exactly 48 squared data products."""
        observer = Instrumentation(profile=False)
        a = [[0] * 16 for _ in range(16)]
        alphaevolve.multiply(a, a, instrumentation=observer)
        self.assertEqual(observer.counts["field_multiplications"], 48 ** 2)
        self.assertEqual(observer.counts["recursive_nodes"], 1 + 48 + 48 ** 2)
        self.assertEqual(observer.products_by_depth, {2: 48 ** 2})

    @requires_coefficients("alphaevolve")
    def test_cold_coefficient_validation_is_not_product_work(self):
        """Cache warming must never change an experiment's arithmetic budget."""
        alphaevolve._factors.cache_clear()
        cold, warm = Instrumentation(), Instrumentation()
        alphaevolve.multiply([[2]], [[3]], instrumentation=cold)
        alphaevolve.multiply([[2]], [[3]], instrumentation=warm)
        self.assertEqual(cold.counts, warm.counts)
        self.assertEqual(cold.counts["field_multiplications"], 1)
        self.assertEqual(cold.counts["real_multiplications"], 4)

    def test_instrumentation_preserves_exact_and_floating_outputs(self):
        """Observation preserves operation order and outputs for every scalar domain."""
        for a, b in [([[Fraction(1, 3), 2]], [[3], [Fraction(2, 7)]]),
                     ([[1.25, -3.5]], [[2.75], [0.125]]),
                     ([[1j, 2]], [[3], [-4j]])]:
            for name, implementation in available_algorithms().items():
                with self.subTest(algorithm=name):
                    expected = implementation(a, b)
                    observer = Instrumentation(trace=True, capture_values=True, trace_limit=50)
                    self.assertEqual(implementation(a, b, instrumentation=observer), expected)
                    self.assertTrue(check_product(a, b, expected).passed)

    def test_native_complex_operations_remain_honestly_opaque(self):
        """Do not invent the interpreter's internal real multiplication count."""
        observer = Instrumentation()
        multiply([[1j]], [[2j]], instrumentation=observer)
        self.assertEqual(observer.counts["native_complex_multiplications"], 1)
        self.assertEqual(observer.counts["real_multiplications"], 0)

    def test_trace_stream_nesting_and_retention(self):
        """A bounded retained prefix can coexist with a complete streamed trace."""
        streamed = []
        observer = Instrumentation(trace=True, trace_limit=3, capture_values=True, sink=streamed.append)
        multiply([[2]], [[3]], instrumentation=observer)
        self.assertEqual(len(observer.events), 3)
        self.assertEqual(observer.dropped_events, len(streamed) - 3)
        active = []
        for event in streamed:
            if event["type"] == "enter":
                self.assertEqual(event["parent"], active[-1] if active else None)
                active.append(event["span"])
            elif event["type"] == "exit":
                self.assertEqual(active.pop(), event["span"])
            else:
                self.assertEqual(event["parent"], active[-1])
                self.assertIn("operands", event)
        self.assertEqual(active, [])
        report = json.loads(json.dumps(observer.snapshot()))
        self.assertEqual(report["counts"]["field_multiplications"], 1)
        self.assertEqual(report["profiles"]["naive.multiply"]["inclusive_counts"]["field_multiplications"], 1)

    def test_failure_restores_context_and_counts_attempted_atomic_work(self):
        """An arithmetic exception cannot cause later unobserved work to leak in."""
        observer = Instrumentation(trace=True)
        with self.assertRaises(ArithmeticError):
            with session(observer):
                scalars.divide(3, 2)
        before = observer.counts.copy()
        multiply([[2]], [[3]])
        self.assertEqual(observer.counts, before)
        self.assertEqual(observer.counts["integer_divmods"], 1)
        self.assertEqual(observer.events[-1]["status"], "error")

    def test_sink_arithmetic_is_not_algorithm_work(self):
        """A callback doing extra work cannot contaminate counts or recurse in tracing."""
        def sink(event):
            """Exercise an unmeasured calculation from inside the event consumer."""
            multiply([[9]], [[9]])

        observer = Instrumentation(sink=sink)
        multiply([[2]], [[3]], instrumentation=observer)
        self.assertEqual(observer.counts["field_multiplications"], 1)

    def test_generic_recipe_structure_is_validated(self):
        """Reject invalid configurations before the reusable evaluator can execute."""
        from matrix_multiplication.schemes import BilinearScheme
        with self.assertRaises(ValueError):
            BilinearScheme("broken", 1, (), (), ())
        with self.assertRaises(ValueError):
            BilinearScheme("broken", 2, (((4, 1),),), (((0, 1),),), (((0, 1),),))

    def test_observers_are_isolated_and_verification_is_excluded(self):
        """Independent experiments and the exact oracle cannot pollute each other."""
        first, second = Instrumentation(), Instrumentation()
        result = multiply([[2]], [[3]], instrumentation=first)
        before = first.counts.copy()
        with session(first):
            self.assertTrue(check_product([[2]], [[3]], result).passed)
            multiply([[4]], [[5]], instrumentation=second)
        self.assertEqual(first.counts, before)
        self.assertEqual(second.counts["field_multiplications"], 1)
        with self.assertRaises(TypeError):
            multiply([[1]], [[1]], instrumentation=object())
        with self.assertRaises(ValueError):
            Instrumentation(trace_limit=-1)

    def test_cli_exports_counts_and_bounded_traces(self):
        """The explorer exposes observation without measuring its correctness oracle."""
        from contextlib import redirect_stdout
        import io
        # Checkout-local temporary artifacts also work in restricted runners.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory, redirect_stdout(io.StringIO()):
            report, trace = Path(directory) / "report.json", Path(directory) / "trace.json"
            status = explore.main(["--algorithm", "strassen", "--report", str(report),
                                   "--trace", str(trace), "--trace-limit", "10"])
            self.assertEqual(status, 0)
            measured = json.loads(report.read_text())["strassen"]
            self.assertEqual(measured["counts"]["field_multiplications"], 7)
            self.assertEqual(measured["experiment"]["cutoff"], 1)
            self.assertEqual(measured["experiment"]["input_shapes"], [[2, 2], [2, 2]])
            self.assertEqual(measured["experiment"]["classical_data_product_baseline"], 8)
            data = json.loads(trace.read_text())
            self.assertEqual(len(data["events"]["strassen"]), 10)
            self.assertGreater(data["reports"]["strassen"]["dropped_trace_events"], 0)


if __name__ == "__main__":
    unittest.main()
