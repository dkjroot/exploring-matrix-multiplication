"""Smirnov rank-40 cyclic square multiplier: exponent log_54(64000).

Coefficients are optional local research data. Three verified cyclic orientations
compose (3,3,6), (3,6,3), (6,3,3) into a balanced square base. No upstream execution
code is imported. Setup is cached outside measured matrix arithmetic.
"""

from functools import lru_cache
from pathlib import Path

from .rectangular import RectangularCycle, cyclic_rotation
from .scheme_io import load_scheme_document, scheme_from_document

DATA_PATH = Path(__file__).resolve().parents[1] / "Background Research" / "smirnov-336.json"


@lru_cache(maxsize=1)
def plan():
    """Load the exact local recipe and construct its certified cyclic orientations."""
    try:
        # Coefficients and notices come from one verified read. Rotation changes
        # index conventions, so every orientation is then independently checked
        # before the cycle can advertise a square recurrence.
        document = load_scheme_document(DATA_PATH)
        base = scheme_from_document(document)
        provenance = document.get("provenance", {})
    except FileNotFoundError as exc:
        raise FileNotFoundError("Smirnov data missing: run python tools/import_smirnov.py --download") from exc
    second = cyclic_rotation(base)
    return RectangularCycle((base, second, cyclic_rotation(second)), provenance=provenance)


def multiply(a, b, *, cutoff=1, instrumentation=None):
    """Multiply through our primitives with optional arithmetic instrumentation."""
    from .common import validate
    validate(a, b, cutoff)
    return plan().multiply(a, b, cutoff=cutoff, instrumentation=instrumentation)
