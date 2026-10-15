"""Minimal rectangular block machinery for a balanced square recursion cycle.

This layer exists to execute Smirnov's lower-exponent construction. It reuses
the sparse evaluator, traversal, padding, scalar operations, and observation.
Only balanced cycles are accepted: their three cumulative dimensions must agree.
"""

from math import log, prod

from .common import padded_dimensions, recursive_kernel, recursive_product, validate
from .scheme_io import coefficient_pair, scheme_document, scheme_from_document, verify_scheme
from .schemes import BilinearScheme, bilinear_step


def cyclic_rotation(scheme):
    """Rotate tr(A B C) and transpose the output leg back into row-major order.

    A(m,k), B(k,n), C(n,m) form the symmetric trilinear description. Rotating
    them gives a (k,n,m) multiplication. Each output coefficient describes the
    transposed trace leg, so the two moved legs require explicit index maps.
    """
    m, k, n = scheme.shape
    left = scheme.right
    # Source output (row i, column j) is W_ij, but the trace's third matrix
    # carries coordinate C_ji. After rotation C becomes an input, hence this
    # transpose. The old A input becomes the new output and is transposed too.
    right = tuple(tuple(((i % n)*m+i//n, c) for i, c in col) for col in scheme.output)
    output = tuple(tuple(((i % k)*m+i//k, c) for i, c in col) for col in scheme.left)
    rotated = BilinearScheme(f"{scheme.name}-rotated", None, left, right, output,
                             divisor=scheme.divisor, dimensions=(k, n, m))
    return verify_scheme(rotated)


class RectangularCycle:
    """A verified real-coefficient cycle whose full contraction is square."""

    def __init__(self, schemes, *, provenance=None):
        """Validate exact identities and square balance before executing inputs."""
        # Interchange canonicalizes zero-imaginary coefficients to real scalars,
        # preventing an accidental wrapper from changing this backend's domain.
        self.stages = tuple(scheme_from_document(scheme_document(s)) for s in schemes)
        if not self.stages:
            raise ValueError("cycle must contain stages")
        if any(coefficient_pair(c)[1] for s in self.stages for factor in
               (s.left, s.right, s.output) for col in factor for _, c in col):
            raise ValueError("this rectangular backend supports exact real coefficients")
        self.shapes = tuple(s.shape for s in self.stages)
        totals = tuple(prod(s[axis] for s in self.shapes) for axis in range(3))
        # After one complete cycle, each axis shrinks by its cumulative product.
        # Equality gives one square base dimension D. Without balance, describing
        # the result as T(n)=R*T(n/D)+O(n^2) would not be justified.
        if len(set(totals)) != 1:
            raise ValueError("stage cycle must contract to a square construction")
        self.radix = totals[0]
        self.rank = prod(s.rank for s in self.stages)
        self.documents = tuple(scheme_document(s, provenance=provenance) for s in self.stages)

    def model(self):
        """Derive the square recurrence from the balanced rectangular cycle."""
        return {"algorithm": "rectangular-cycle", "square_arithmetic_exponent": log(self.rank)/log(self.radix),
                "exponent_expression": f"log_{self.radix}({self.rank})",
                "base_dimension": self.radix, "base_bilinear_rank": self.rank,
                "recurrence": f"T(n) = {self.rank} T(n/{self.radix}) + O(n^2)",
                "parameter_source": "exact verified rectangular ranks and balanced dimensions",
                "arithmetic_domain": "rational/real field; native complex operations opaque",
                "schedule": [s.name for s in self.stages], "schedule_repeats": True,
                "stage_dimensions": [list(s) for s in self.shapes],
                "scheme_documents": list(self.documents),
                "identity_validation": "complete exact identity for each cyclic orientation",
                "assumptions": ["unit-cost field arithmetic", "fixed cutoff", "balanced repeating cycle"]}

    def budget(self, rows, inner, columns, *, cutoff=1, size_policy="fringe"):
        """Count each level's distinct block areas and all selected boundary work."""
        from .size_policy import fringe_budget, validate_size_policy
        validate_size_policy(size_policy)
        if size_policy == "fringe":
            return fringe_budget(self.stages, (rows, inner, columns), cutoff)
        if type(cutoff) is not int or cutoff < 1:
            raise ValueError("cutoff must be positive")
        if any(type(x) is not int or x < 1 for x in (rows, inner, columns)):
            raise ValueError("dimensions must be positive integers")
        shape = padded_dimensions(rows, inner, columns, self.shapes)
        padded = shape
        nodes, depth, operations, levels = 1, 0, 0, []
        while max(shape) > cutoff:
            stage = self.stages[depth % len(self.stages)]
            child = tuple(x//y for x, y in zip(shape, stage.shape))
            m, k, n = child
            # U combinations operate on m*k cells, V on k*n, and W on m*n.
            # A square-area shortcut would miscount rectangular stages. Each
            # stored coefficient performs one scale and one accumulation.
            for factor, area in zip((stage.left, stage.right, stage.output), (m*k, k*n, m*n)):
                operations += nodes * area * 2 * sum(len(c) for c in factor)
            if stage.divisor != 1:
                operations += nodes * m*n * stage.shape[0]*stage.shape[2]
            levels.append({"depth": depth, "stage": stage.name, "nodes": nodes,
                           "size": list(shape), "radix": list(stage.shape), "rank": stage.rank})
            nodes *= stage.rank
            shape, depth = child, depth+1
        products = nodes * prod(shape)
        return {"padded_size": max(padded), "padded_dimensions": list(padded),
                "leaf_size": max(shape), "leaf_dimensions": list(shape), "levels": levels,
                "classical_leaves": nodes, "field_multiplications": products,
                "field_arithmetic_operations": operations+2*products,
                "atomic_arithmetic_operations": operations+2*products,
                "atomic_domain": "real scalars; native complex operations opaque"}

    def multiply(self, a, b, *, cutoff=1, instrumentation=None, size_policy="fringe"):
        """Run the square-target cycle through shared rectangular block primitives."""
        validate(a, b, cutoff)

        def select(depth):
            """Bind this level's verified coefficients without storing a giant tensor."""
            scheme = self.stages[depth % len(self.stages)]

            def step(left, right, product):
                """Apply the shared sparse scale/add/product/output evaluator."""
                return bilinear_step(left, right, product, scheme)

            return scheme.shape, step

        def kernel(left, right, leaf):
            """Use the same recursive engine as Strassen and AlphaEvolve."""
            return recursive_kernel(left, right, leaf, self.shapes[0], None,
                                    name="rectangular-cycle", select_step=select, size_policy=size_policy)

        return recursive_product(a, b, cutoff, self.shapes[0], kernel,
                                 name="rectangular-cycle", radices=self.shapes,
                                 instrumentation=instrumentation, size_policy=size_policy)
