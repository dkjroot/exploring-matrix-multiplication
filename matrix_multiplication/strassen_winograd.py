"""Winograd's rearrangement: seven products and 15 block additions per level.

This is the Strassen-Winograd variant, not the Coppersmith-Winograd tensor method.
Sharing intermediate combinations reduces the constant: 7 block additions and
8 subtractions, versus Strassen's 18 operations. Both call seven child products,
so the shared traversal gives the same log_2(7) exponent.
"""

from .common import add, recursive_kernel, recursive_product, subtract
from .instrumentation import scope


def step(a_blocks, b_blocks, product):
    """Reuse eight input combinations and seven output combinations per node."""
    a11, a12, a21, a22 = a_blocks
    b11, b12, b21, b22 = b_blocks
    # S/T intermediates are reused; expanding them again would increase add costs.
    s1 = add(a21, a22)
    s2 = subtract(s1, a11)
    s3 = subtract(a11, a21)
    s4 = subtract(a12, s2)
    t1 = subtract(b12, b11)
    t2 = subtract(b22, t1)
    t3 = subtract(b22, b12)
    t4 = subtract(t2, b21)
    p1 = product(a11, b11)
    p2 = product(a12, b21)
    p3 = product(s4, b22)
    p4 = product(a22, t4)
    p5 = product(s1, t1)
    p6 = product(s2, t2)
    p7 = product(s3, t3)
    # U intermediates share output assembly work without adding child products.
    u1 = add(p1, p2)
    u2 = add(p1, p6)
    u3 = add(u2, p7)
    u4 = add(u2, p5)
    u5 = add(u4, p3)
    u6 = subtract(u3, p4)
    u7 = add(u3, p5)
    return [u1, u5, u6, u7]


def _step(a_blocks, b_blocks, product):
    """Profile the formula separately from splitting, joining, and leaf arithmetic."""
    with scope("strassen-winograd.step"):
        return step(a_blocks, b_blocks, product)


def _multiply(a, b, cutoff, *, size_policy="fringe"):
    """Bind Winograd's formula to exactly the traversal used by Strassen."""
    return recursive_kernel(a, b, cutoff, 2, _step, name="strassen-winograd", size_policy=size_policy)


def multiply(a, b, *, cutoff=1, instrumentation=None, size_policy="fringe"):
    """Multiply through unpadded cores/fringes, or the optional padded baseline."""
    def kernel(left, right, leaf):
        """Bind the selected size policy to every recursive child."""
        return _multiply(left, right, leaf, size_policy=size_policy)

    return recursive_product(a, b, cutoff, 2, kernel, name="strassen-winograd",
                             instrumentation=instrumentation, size_policy=size_policy)
