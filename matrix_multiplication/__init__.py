"""Public registry and dispatch for independently described multiplication schemes.

Algorithm modules share block/scalar/traversal primitives, while keeping their
formulas distinct. Instrumentation is opt-in and travels through a context to
every primitive, so old callers still receive an ordinary matrix result.
"""

from . import alphaevolve, naive, strassen, strassen_winograd, smirnov
from .instrumentation import Instrumentation

ALGORITHMS = {
    "naive": naive.multiply,
    "strassen": strassen.multiply,
    "strassen-winograd": strassen_winograd.multiply,
    "alphaevolve": alphaevolve.multiply,
    "smirnov": smirnov.multiply,
}

DESCRIPTIONS = {
    "naive": "Classical row-by-column; square exponent 3.",
    "strassen": "7 products per 2x2 block; exponent log_2(7), about 2.8074.",
    "strassen-winograd": "Same 7 products, 15 block additions; exponent about 2.8074.",
    "alphaevolve": "48 complex products per 4x4 block; exponent log_4(48), about 2.7925.",
    "smirnov": "40 products per rectangular stage; cyclic square exponent log_54(64000), about 2.7743.",
}


def multiply(a, b, *, algorithm="naive", cutoff=1, instrumentation=None, size_policy="fringe"):
    """Dispatch a named formula, optionally collecting counters/timings/traces."""
    try:
        implementation = ALGORITHMS[algorithm]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {', '.join(ALGORITHMS)}") from exc
    return implementation(a, b, cutoff=cutoff, instrumentation=instrumentation, size_policy=size_policy)
