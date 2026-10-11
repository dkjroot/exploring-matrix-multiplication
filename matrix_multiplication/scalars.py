"""Bottom layer: visible scalar arithmetic and exact complex pairs.

Algorithms call add/subtract/multiply/scale/divide rather than hidden arithmetic.
Each semantic field operation expands into observable atomic real operations
when using Gaussian pairs. Constant scaling is counted separately from products
of data-dependent input combinations, which determine a bilinear scheme's rank.
No zero/one shortcuts silently change the reference operation schedule.
"""

import operator
from dataclasses import dataclass

from .instrumentation import count, evaluate, scope


@dataclass(frozen=True, slots=True, repr=False)
class Gaussian:
    """An immutable real/imaginary pair, preserving supplied exact components.

    Arithmetic always constructs a new pair rather than altering its operands.
    This matters particularly for coefficients: once a scheme has passed exact
    verification, later callers must not be able to change its stored constants.
    The generated dataclass constructor stores components without float coercion;
    coefficient validation belongs to scheme_io, public input validation to common.
    """

    real: object = 0
    imag: object = 0

    @classmethod
    def from_number(cls, value):
        """Embed a real number or unpack a native complex without arithmetic."""
        if isinstance(value, cls):
            return value
        if isinstance(value, complex):
            return cls(value.real, value.imag)
        return cls(value, 0)

    def __repr__(self):
        """Make optional scalar traces show both components without rounding."""
        return f"Gaussian({self.real!r}, {self.imag!r})"


def _atomic(operation, a, b, *, role=None):
    """Apply a primitive operator, labelling native complex work as opaque."""
    domain = "native_complex" if isinstance(a, complex) or isinstance(b, complex) else "real"
    functions = {"add": operator.add, "subtract": operator.sub,
                 "multiply": operator.mul, "divide": operator.truediv}
    categories = {"add": "additions", "subtract": "subtractions",
                  "multiply": "multiplications", "divide": "divisions"}
    return evaluate(f"{domain}.{operation}", f"{domain}_{categories[operation]}",
                    functions[operation], a, b, role=role)


def add(a, b):
    """Add two field elements; complex pairs require two real additions."""
    with scope("scalar.add"):
        count("field_additions")
        if isinstance(a, Gaussian) or isinstance(b, Gaussian):
            a, b = Gaussian.from_number(a), Gaussian.from_number(b)
            return Gaussian(_atomic("add", a.real, b.real), _atomic("add", a.imag, b.imag))
        return _atomic("add", a, b)


def subtract(a, b):
    """Subtract field elements while exposing each complex component operation."""
    with scope("scalar.subtract"):
        count("field_subtractions")
        if isinstance(a, Gaussian) or isinstance(b, Gaussian):
            a, b = Gaussian.from_number(a), Gaussian.from_number(b)
            return Gaussian(_atomic("subtract", a.real, b.real), _atomic("subtract", a.imag, b.imag))
        return _atomic("subtract", a, b)


def _product(a, b, role):
    """Evaluate a product; the exact complex formula uses four real products."""
    if isinstance(a, Gaussian) or isinstance(b, Gaussian):
        a, b = Gaussian.from_number(a), Gaussian.from_number(b)
        # Preserve the original evaluation order and its floating-point behavior.
        real = _atomic("subtract", _atomic("multiply", a.real, b.real, role=role),
                       _atomic("multiply", a.imag, b.imag, role=role))
        imag = _atomic("add", _atomic("multiply", a.real, b.imag, role=role),
                       _atomic("multiply", a.imag, b.real, role=role))
        return Gaussian(real, imag)
    return _atomic("multiply", a, b, role=role)


def multiply(a, b):
    """Multiply two data-dependent field values: one bilinear product."""
    with scope("scalar.multiply"):
        count("field_multiplications")
        return _product(a, b, "data")


def scale(value, coefficient):
    """Multiply by a fixed coefficient, separately from the scheme's rank."""
    with scope("scalar.scale"):
        count("coefficient_multiplications")
        return _product(coefficient, value, "coefficient")


def _divide_component(value, divisor):
    """Use exact quotient/remainder for integers, ordinary division otherwise."""
    if isinstance(value, int) and isinstance(divisor, int):
        quotient, remainder = evaluate("integer.divmod", "integer_divmods", divmod,
                                       value, divisor, role="normalization")
        if remainder:
            raise ArithmeticError("normalization violated exact integer divisibility")
        return quotient
    return _atomic("divide", value, divisor, role="normalization")


def divide(value, divisor):
    """Normalize a field value; Gaussian pairs expose both component divisions."""
    with scope("scalar.divide"):
        count("field_normalizations")
        if isinstance(value, Gaussian):
            return Gaussian(_divide_component(value.real, divisor),
                            _divide_component(value.imag, divisor))
        return _divide_component(value, divisor)
