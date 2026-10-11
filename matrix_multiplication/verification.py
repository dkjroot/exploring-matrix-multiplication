"""Independent correctness reference using exact standard-library rational arithmetic.

Fractions represent the supplied values exactly, including each binary float's
actual value. Complex entries are represented as pairs of Fractions. This is
verification code only; none of the multiplication implementations calls it.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import hypot, isfinite
from numbers import Rational

from .common import validate


def _parts(value):
    """Represent a finite public scalar as exact rational real/imaginary parts."""
    if isinstance(value, bool) or not isinstance(value, (Rational, float, complex)):
        raise TypeError("output entries must be numbers")
    if isinstance(value, complex):
        return Fraction(value.real), Fraction(value.imag)
    return Fraction(value), Fraction(0)


def exact_reference(a, b):
    """Compute a classical rational oracle independently of instrumented primitives."""
    rows, inner, columns = validate(a, b)
    result = []
    for row in a:
        output = []
        for j in range(columns):
            real, imag = Fraction(0), Fraction(0)
            for k in range(inner):
                ar, ai = _parts(row[k])
                br, bi = _parts(b[k][j])
                real += ar * br - ai * bi
                imag += ar * bi + ai * br
            output.append((real, imag))
        result.append(output)
    return result


@dataclass(frozen=True)
class Verification:
    """Immutable result of comparing every output entry with the exact oracle."""
    passed: bool
    exact: bool
    max_absolute_error: float
    failures: tuple


def _magnitude(real, imag):
    """Produce a display-only norm; comparison itself avoids floating overflow."""
    try:
        return hypot(float(real), float(imag))
    except OverflowError:
        return float("inf")


def check_product(a, b, result, *, atol=1e-9, rtol=1e-9):
    """Exact equality for rational inputs; per-entry tolerance for float/complex.

    The floating tolerance is atol + rtol*abs(exact_reference_entry). It is a
    correctness check, not a universal numerical stability guarantee.
    """
    if not (isfinite(atol) and isfinite(rtol)) or atol < 0 or rtol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    rows, _, columns = validate(a, b)
    if not isinstance(result, (list, tuple)) or len(result) != rows:
        return Verification(False, False, float("inf"), ("incorrect row count",))
    if any(not isinstance(row, (list, tuple)) or len(row) != columns for row in result):
        return Verification(False, False, float("inf"), ("incorrect column count",))
    exact = all(not isinstance(x, (float, complex)) for m in (a, b) for row in m for x in row)
    expected = exact_reference(a, b)
    failures = []
    maximum = 0.0
    for i in range(rows):
        for j in range(columns):
            er, ei = expected[i][j]
            try:
                ar, ai = _parts(result[i][j])
            except (ValueError, TypeError, OverflowError):
                failures.append((i, j, "invalid or nonfinite output"))
                maximum = float("inf")
                continue
            dr, di = ar - er, ai - ei
            error = _magnitude(dr, di)
            maximum = max(maximum, error)
            if exact:
                matches = dr == 0 and di == 0
            else:
                # Square |error| <= atol + rtol*|expected| without approximating
                # the square root. This also avoids overflow at large magnitudes.
                absolute, relative = Fraction(atol), Fraction(rtol)
                expected_squared = er * er + ei * ei
                slack = dr * dr + di * di - absolute * absolute - relative * relative * expected_squared
                matches = (slack <= 0 or slack * slack <=
                           4 * absolute * absolute * relative * relative * expected_squared)
            if not matches:
                failures.append((i, j, "exact mismatch" if exact else "outside tolerance"))
    return Verification(not failures, exact, maximum, tuple(failures))
