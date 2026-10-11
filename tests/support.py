"""Make optional research-data requirements explicit in regression tests.

Downloaded coefficient tables are deliberately ignored by Git. A fresh checkout
must still exercise our self-contained algorithms, construction machinery, and
missing-data diagnostics. Tests requiring external tables skip visibly until
the documented importers have been run; they never download data automatically.
Only absence causes a skip: an existing but corrupted table must still fail.
"""

from functools import wraps

from matrix_multiplication import ALGORITHMS, alphaevolve, smirnov


def available_algorithms():
    """Return public callbacks whose required local coefficient files are present.

    This filtering is for broad shared-contract tests only. Dedicated tests use
    requires_coefficients so the test runner visibly reports unavailable cases.
    The production registry always retains every algorithm and reports errors
    for missing data instead of silently filtering user-requested experiments.
    """
    optional = {"alphaevolve": alphaevolve.DATA_PATH, "smirnov": smirnov.DATA_PATH}
    return {name: callback for name, callback in ALGORITHMS.items()
            if name not in optional or optional[name].is_file()}


def requires_coefficients(*names):
    """Skip one test with an importer instruction when its optional data is absent."""
    def decorate(function):
        """Retain the test name and docstring while adding a runtime data check."""
        @wraps(function)
        def checked(test_case, *args, **kwargs):
            """Require each named table before executing this data-dependent test."""
            available = available_algorithms()
            for name in names:
                if name not in available:
                    test_case.skipTest(f"optional {name} data missing; run python tools/import_{name}.py --download")
            return function(test_case, *args, **kwargs)
        return checked
    return decorate
