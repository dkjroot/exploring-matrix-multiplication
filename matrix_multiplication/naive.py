"""Classical reference algorithm: one observable dot product per output cell.

This public adapter shares its leaf kernel with all recursive algorithms. For an
m-by-k times k-by-n product it performs m*k*n products and the same number of
accumulation additions. No padding is used; square arithmetic exponent is 3.
"""

from .common import classical, validate
from .instrumentation import count, scope, session


def multiply(a, b, *, cutoff=1, instrumentation=None):
    """Validate and multiply; cutoff is accepted for interface consistency only."""
    with session(instrumentation), scope("naive.multiply", cutoff=cutoff):
        rows, inner, columns = validate(a, b, cutoff)
        count("input_cells", rows * inner + inner * columns)
        count("classical_leaves")
        return classical(a, b)
