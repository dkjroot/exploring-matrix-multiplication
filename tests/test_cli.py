"""Exercise user-visible explorer results, errors, and exact/tolerant checks."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest

import explore


class CliTests(unittest.TestCase):
    """Check CLI contracts while capturing output without subprocess variability."""

    def run_cli(self, arguments):
        """Return exit status, standard output, and diagnostics for supplied arguments."""
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = explore.main(arguments)
        return status, output.getvalue(), errors.getvalue()

    def test_all_algorithms_and_rectangular_literals(self):
        """Run all algorithms on one explicit rectangular product."""
        status, output, errors = self.run_cli(["--a", "[[1,2,3]]", "--b", "[[4],[5],[6]]"])
        self.assertEqual(status, 0, errors)
        self.assertEqual(output.count("PASS: exact equality"), 4)
        self.assertEqual(output.count("[[32]]"), 4)

    def test_complex_literals(self):
        """Parse complex literals and display a verified complex answer."""
        status, output, errors = self.run_cli(["--algorithm", "alphaevolve", "--a", "[[1j]]", "--b", "[[2j]]"])
        self.assertEqual(status, 0, errors)
        self.assertIn("PASS", output)
        self.assertIn("(-2+0j)", output)

    def test_shape_errors_exit_nonzero(self):
        """Return a failure status with a helpful dimension diagnostic."""
        status, _, errors = self.run_cli(["--algorithm", "naive", "--a", "[[1,2]]", "--b", "[[3]]"])
        self.assertEqual(status, 1)
        self.assertIn("column count", errors)

    def test_numerical_failure_exits_nonzero(self):
        """Propagate an oracle-detected accuracy failure through the CLI status."""
        status, output, _ = self.run_cli(["--algorithm", "naive", "--a", "[[1e16,1.0,-1e16]]",
                                        "--b", "[[1.0],[1.0],[1.0]]"])
        self.assertEqual(status, 1)
        self.assertIn("FAIL", output)

    def test_list_distinguishes_the_exponents(self):
        """Display implemented exponents and distinguish the unimplemented bound."""
        status, output, _ = self.run_cli(["--list"])
        self.assertEqual(status, 0)
        self.assertIn("2.7925", output)
        self.assertIn("2.25", output)
        self.assertIn("not implemented", output)
