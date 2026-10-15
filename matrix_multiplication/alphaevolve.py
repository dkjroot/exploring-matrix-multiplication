"""Published rank-48 <4,4,4> scheme, recursively applied to matrix blocks.

The algorithm adapter loads and validates its published coefficients, translates
the transposed output convention, then delegates execution to the shared sparse
scheme and recursion primitives. Exact complex pairs expose real arithmetic while
preserving integers/Fractions. Coefficient verification is setup work, excluded
from product instrumentation; no notebook code is executed.
"""

from functools import lru_cache
import json
from numbers import Rational
from pathlib import Path

from .common import recursive_kernel, recursive_product
from .scalars import Gaussian
from .schemes import BilinearScheme, bilinear_step


DATA_PATH = Path(__file__).resolve().parent.parent / "Background Research" / "alphaevolve-48.json"


def validate_factors(factors):
    """Check all 4096 tensor coefficients, using exact Gaussian integer pairs.

    Each stored coefficient is twice the published value. Thus the reconstructed
    tensor must be eight times the matrix multiplication tensor. This is a finite
    verification of the whole bilinear identity, not just sampled input matrices.
    """
    if not isinstance(factors, (list, tuple)) or len(factors) != 3:
        raise ValueError("expected three factor matrices")
    for factor in factors:
        if len(factor) != 16 or any(len(row) != 48 for row in factor):
            raise ValueError("each factor must have shape 16 by 48")
        for row in factor:
            for pair in row:
                if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                        or any(type(v) is not int for v in pair)):
                    raise ValueError("coefficients must be pairs of doubled integers")
    u, v, w = factors
    for left in range(16):
        i, j = divmod(left, 4)
        for right in range(16):
            j2, k = divmod(right, 4)
            for output in range(16):
                real = imag = 0
                for term in range(48):
                    ar, ai = u[left][term]
                    br, bi = v[right][term]
                    cr, ci = w[output][term]
                    pr, pi = ar * br - ai * bi, ar * bi + ai * br
                    real += pr * cr - pi * ci
                    imag += pr * ci + pi * cr
                expected = 8 if j == j2 and output == k * 4 + i else 0
                if real != expected or imag != 0:
                    raise ValueError(f"incorrect tensor coefficient at {left}, {right}, {output}")


@lru_cache(maxsize=1)
def _factors():
    """Load/verify research data once, then compile sparse coefficient columns.

    This setup uses ordinary exact integer arithmetic, separate from measured
    multiplication. Cold and warm coefficient caches must give identical counts.
    """
    try:
        document = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "AlphaEvolve coefficients are missing. Run: python tools/import_alphaevolve.py --download"
        ) from exc
    factors = document["factors_doubled"]
    validate_factors(factors)
    # Sparse column lists describe the linear combinations for each product.
    return tuple(tuple(tuple((index, *row[term]) for index, row in enumerate(factor)
                             if row[term] != [0, 0]) for term in range(48))
                 for factor in factors)


def _scheme(factors):
    """Translate source conventions into a reusable, row-major scheme descriptor."""
    left, right, output = factors
    left = tuple(tuple((index, Gaussian(real, imag)) for index, real, imag in column)
                 for column in left)
    right = tuple(tuple((index, Gaussian(real, imag)) for index, real, imag in column)
                  for column in right)
    # Source W describes the transposed output leg: column*4+row -> row*4+column.
    output = tuple(tuple(((index % 4) * 4 + index // 4, Gaussian(real, imag))
                         for index, real, imag in column) for column in output)
    return BilinearScheme("alphaevolve", 4, left, right, output, divisor=8)


def _multiply(a, b, cutoff, factors, *, size_policy="fringe"):
    """Bind a fixed scheme to radix-four traversal without duplicating recursion."""
    scheme = _scheme(factors)

    def step(a_blocks, b_blocks, product):
        """Supply the shared traversal with this adapter's verified rank-48 recipe."""
        return bilinear_step(a_blocks, b_blocks, product, scheme)

    return recursive_kernel(a, b, cutoff, 4, step, name="alphaevolve", size_policy=size_policy)


def multiply(a, b, *, cutoff=1, instrumentation=None, size_policy="fringe"):
    """Multiply through exact pairs, restoring the caller's real/complex domain.

    Only computation of the product is observed; published tensor validation and
    input-domain inspection are setup. Real exact outputs must cancel imaginary
    components exactly; native complex inputs retain floating-point semantics.
    """
    # Preserve contract errors before touching optional research data.
    from .common import validate
    validate(a, b, cutoff)
    factors = _factors()
    complex_input = any(isinstance(x, complex) for matrix in (a, b) for row in matrix for x in row)

    def kernel(left, right, leaf_size):
        """Close over loaded coefficients while keeping the generic kernel API."""
        return _multiply(left, right, leaf_size, factors, size_policy=size_policy)

    def decode(value):
        """Restore public values; discard only floating imaginary cancellation noise."""
        if complex_input:
            return complex(value.real, value.imag)
        if isinstance(value.imag, Rational) and value.imag != 0:
            raise ArithmeticError("unexpected imaginary component for exact real inputs")
        return value.real

    return recursive_product(a, b, cutoff, 4, kernel, name="alphaevolve",
                             instrumentation=instrumentation, encode=Gaussian.from_number,
                             decode=decode, size_policy=size_policy)
