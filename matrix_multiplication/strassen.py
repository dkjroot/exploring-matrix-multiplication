"""Strassen's explicit seven-product formula over shared block primitives.

One non-leaf node uses 12 block additions and 6 block subtractions. The common
traversal applies the formula recursively: T(n)=7*T(n/2)+O(n^2). Naming each
product keeps the algebra readable and the primitive trace easy to inspect.
"""

from .common import add, recursive_kernel, recursive_product, subtract
from .instrumentation import scope


def step(a_blocks, b_blocks, product):
    """Compute four output blocks using exactly seven calls to a child product."""
    a11, a12, a21, a22 = a_blocks
    b11, b12, b21, b22 = b_blocks
    # Ten input combinations trade one of the classical eight products for adds.
    m1 = product(add(a11, a22), add(b11, b22))
    m2 = product(add(a21, a22), b11)
    m3 = product(a11, subtract(b12, b22))
    m4 = product(a22, subtract(b21, b11))
    m5 = product(add(a11, a12), b22)
    m6 = product(subtract(a21, a11), add(b11, b12))
    m7 = product(subtract(a12, a22), add(b21, b22))
    # Eight more combinations reconstruct C11, C12, C21, C22 in row-major order.
    return [
        add(subtract(add(m1, m4), m5), m7),
        add(m3, m5),
        add(m2, m4),
        add(add(subtract(m1, m2), m3), m6),
    ]


def _step(a_blocks, b_blocks, product):
    """Give the readable formula its own profiling/trace boundary."""
    with scope("strassen.step"):
        return step(a_blocks, b_blocks, product)


def _multiply(a, b, cutoff):
    """Bind the Strassen formula to the shared binary recursive traversal."""
    return recursive_kernel(a, b, cutoff, 2, _step, name="strassen")


def multiply(a, b, *, cutoff=1, instrumentation=None):
    """Return a fresh rectangular product, optionally recording primitive work."""
    return recursive_product(a, b, cutoff, 2, _multiply, name="strassen",
                             instrumentation=instrumentation)
