"""Reusable bilinear-scheme layer: sparse coefficients become observable blocks.

U/V specify each product's input linear combinations. W specifies how to add its
result to output blocks. This is our evaluator, independent of source notebooks.
A fixed rank-r radix-u scheme makes r child calls; its non-product work is O(n^2).
More schemes can use this layer after their identities are independently verified.
"""

from dataclasses import dataclass

from .common import accumulate_scaled, linear_combination, map_entries, zeros
from .instrumentation import scope
from .scalars import divide


@dataclass(frozen=True)
class BilinearScheme:
    """A fixed, verified recipe in sparse (block_index, scalar_coefficient) columns.

    Output indices are normalized to row-major order before execution. Storing
    doubled coefficients can require one output normalization per recursive node.
    This descriptor is not itself a correctness certificate for arbitrary data.
    """

    name: str
    radix: int
    left: tuple
    right: tuple
    output: tuple
    divisor: int = 1
    dimensions: tuple = None

    def __post_init__(self):
        """Validate structure and snapshot nested containers before identity checking.

        A frozen dataclass prevents assignment to fields, but a list inside such
        a field remains mutable. Convert all factor/column/term containers to
        tuples so edits to a caller's source tables cannot invalidate a verified
        recipe later. Gaussian coefficients are immutable too. This protects the
        agreement between an executed formula, its model, and its certificate.
        """
        if self.dimensions is not None:
            if (self.radix is not None or not isinstance(self.dimensions, (tuple, list)) or len(self.dimensions) != 3
                    or any(type(x) is not int or x < 2 for x in self.dimensions)):
                raise ValueError("rectangular schemes need three dimensions at least two and no square radix")
        elif type(self.radix) is not int or self.radix < 2:
            raise ValueError("scheme radix must be an integer at least two")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("scheme name must be a nonempty string")
        if type(self.divisor) is not int or self.divisor < 1:
            raise ValueError("scheme divisor must be a positive integer")
        if not self.left or not (len(self.left) == len(self.right) == len(self.output)):
            raise ValueError("scheme factors must have the same positive rank")
        m, k, n = self.shape
        for factor, bound in zip((self.left, self.right, self.output), (m*k, k*n, m*n)):
            for column in factor:
                for term in column:
                    if (not isinstance(term, (tuple, list)) or len(term) != 2
                            or type(term[0]) is not int or not 0 <= term[0] < bound):
                        raise ValueError("scheme terms need an in-range block index and coefficient")
        for name in ("left", "right", "output"):
            object.__setattr__(self, name, tuple(tuple(tuple(term) for term in column)
                                                for column in getattr(self, name)))
        if self.dimensions is not None:
            object.__setattr__(self, "dimensions", tuple(self.dimensions))

    @property
    def shape(self):
        """Return the row, shared-inner, and column block counts of this recipe."""
        return self.dimensions or (self.radix, self.radix, self.radix)

    @property
    def rank(self):
        """Return the number of input-combination products at one non-leaf node."""
        return len(self.left)


def bilinear_step(a_blocks, b_blocks, product, scheme):
    """Evaluate a sparse scheme through shared block and scalar primitives.

    Contributions are accumulated in coefficient order, preserving numerical
    behavior. Product results need not all be retained; tracing observes execution
    rather than requiring a dense tensor or a materialized computation graph.
    """
    with scope(f"{scheme.name}.step", rank=scheme.rank):
        m, _, n = scheme.shape
        output = [zeros(len(a_blocks[0]), len(b_blocks[0][0])) for _ in range(m*n)]
        for term in range(scheme.rank):
            with scope("scheme.term", term=term):
                left = linear_combination(a_blocks, scheme.left[term])
                right = linear_combination(b_blocks, scheme.right[term])
                result = product(left, right)
                for index, coefficient in scheme.output[term]:
                    accumulate_scaled(output[index], result, coefficient)
        if scheme.divisor != 1:
            def normalize(value):
                """Remove the scheme's common coefficient scale exactly once."""
                return divide(value, scheme.divisor)

            output = [map_entries(block, normalize, name="block.normalize") for block in output]
        return output
