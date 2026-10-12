"""Execute generic rational bilinear recipes through observable scalar primitives.

Tensor construction produces sparse U/V/W terms, while this module executes them
on actual data. Keeping the kernel independent of CW extraction allows future
restrictions, direct sums, and other constructions to use the same arithmetic
schedule. Unit-size axes and multiple independent outputs need no special case.

This reference schedule counts every coefficient scaling, including one. It is
intentionally simpler than the optimized sparse matrix-scheme schedule; callers
must label which schedule their counters describe. Recipe construction and exact
identity checks are setup work, outside these observed calls.
"""

from fractions import Fraction

from . import scalars
from .instrumentation import scope, session


def evaluate_tensor(tensor, left, right, *, instrumentation=None):
    """Execute any rational bilinear tensor, including sums with unit-size axes.

    This deliberately simple kernel exposes every coefficient scaling, addition,
    and data multiplication through scalars.py. It accepts exact rational input
    vectors in the tensor's basis order; matrix reshaping belongs to the caller.
    It performs every surviving term even if this particular input makes a
    factor zero, so the data-product count equals the finite recipe's term count.
    """
    left, right = tuple(left), tuple(right)
    if (len(left) != tensor.dimensions[0] or len(right) != tensor.dimensions[1]
            or any(type(value) not in (int, Fraction) for value in (*left, *right))):
        raise ValueError("tensor inputs must be exact rational vectors of the correct lengths")
    with session(instrumentation), scope("tensor.evaluate"):
        output = [Fraction(0)]*tensor.dimensions[2]
        for u, v, w in tensor.terms:
            a = linear_form(u, left)
            b = linear_form(v, right)
            value = scalars.multiply(a, b)
            for coordinate, coefficient in w:
                output[coordinate] = scalars.add(output[coordinate], scalars.scale(value, coefficient))
        return tuple(output)


def linear_form(factor, values):
    """Evaluate one sparse input combination using the shared scalar primitives.

    Even coefficients equal to one are counted here. That is a readable fixed
    schedule, not the optimized sparse scheme schedule used by Composition.
    Counter snapshots identify this kernel, so the two schedules are not mixed.
    """
    with scope("tensor.linear_form"):
        result = Fraction(0)
        for coordinate, coefficient in factor:
            result = scalars.add(result, scalars.scale(values[coordinate], coefficient))
        return result
