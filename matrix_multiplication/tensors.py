"""Exact rational tensor constructions, before a multiplication recipe exists.

A term is three sparse vectors U, V, W; its tensor entries are their outer
product. Tensor arithmetic is construction work, deliberately separate from
the scalar operations measured when the resulting algorithm multiplies data.
These primitives support direct sums, powers, linear restrictions, and exact
recovery of polynomial coefficients. No border/asymptotic rank is silently
treated as a finite executable rank. All coefficients here belong to Q.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from .scheme_io import coefficient_pair, verify_scheme
from .schemes import BilinearScheme


def vector(entries):
    """Collect repeated coordinates, remove zeros, and reject approximate values."""
    result = {}
    for index, value in entries:
        if type(index) is not int or index < 0 or type(value) not in (int, Fraction):
            raise ValueError("vectors require nonnegative integer indices and exact rational values")
        # Different construction paths may land on the same coordinate. Sum
        # them exactly before removing cancellations; sorting makes comparison,
        # serialization, projective normalization, and replay deterministic.
        result[index] = result.get(index, 0) + Fraction(value)
    return tuple(sorted((i, c) for i, c in result.items() if c))


def combine(a, b, scale=1):
    """Return a + scale*b using sparse exact coefficient arithmetic."""
    if type(scale) not in (int, Fraction):
        raise ValueError("vector scale must be an exact rational")
    return vector((*a, *((i, scale*c) for i, c in b)))


def kronecker(a, b, width):
    """Pair vector indices in lexicographic order: (i,j) becomes i*width+j."""
    if type(width) is not int or width < 1:
        raise ValueError("Kronecker coordinate width must be a positive integer")
    return vector((i*width+j, u*v) for i, u in a for j, v in b)


@dataclass(frozen=True)
class Tensor:
    """A finite sum of rational rank-one terms in three labelled vector spaces.

    ``rank`` is the length of this decomposition, an upper bound on minimum
    tensor rank. A generic tensor need not be a matrix multiplication tensor.
    """

    dimensions: tuple
    terms: tuple

    def __post_init__(self):
        """Canonicalize vectors and remove identically zero rank-one terms."""
        if (len(self.dimensions) != 3 or any(type(n) is not int or n < 1 for n in self.dimensions)):
            raise ValueError("tensor spaces need three positive integer dimensions")
        terms = []
        for term in self.terms:
            if len(term) != 3:
                raise ValueError("rank-one terms need exactly three factors")
            factors = tuple(vector(v) for v in term)
            if any(any(i >= n for i, _ in v) for v, n in zip(factors, self.dimensions)):
                raise ValueError("tensor coordinate outside its vector space")
            if all(factors):
                terms.append(factors)
        object.__setattr__(self, "terms", tuple(terms))
        object.__setattr__(self, "dimensions", tuple(self.dimensions))

    @property
    def rank(self):
        """Return this executable decomposition's product count, not a minimum."""
        return len(self.terms)

    def expand(self):
        """Expand every coefficient exactly; this can use substantial memory."""
        result = {}
        for a, b, c in self.terms:
            # One rank-one term contributes u_i*v_j*w_k at every triple of its
            # nonzero coordinates. Terms may overlap and cancel. The resulting
            # dictionary describes the tensor itself, rather than one choice of
            # decomposition, so differently shaped recipes can compare equal.
            for (i, u), (j, v), (k, w) in product(a, b, c):
                key = (i, j, k)
                result[key] = result.get(key, 0) + u*v*w
        return {key: value for key, value in result.items() if value}


def classical(shape):
    """Generate the m*k*n elementary products in row-major output convention."""
    m, k, n = shape
    if any(type(d) is not int or d < 1 for d in shape):
        raise ValueError("matrix dimensions must be positive integers")
    # A_ij*B_jl contributes to C_il. Flatten each matrix row by row, using its
    # own column count: k for A, n for B and C. Each of these elementary
    # products becomes one term with a single nonzero coordinate in each leg.
    terms = tuple((((i*k+j, 1),), ((j*n+l, 1),), ((i*n+l, 1),))
                  for i in range(m) for j in range(k) for l in range(n))
    return Tensor((m*k, k*n, m*n), terms)


def from_scheme(scheme):
    """Normalize a verified real rational recipe into a generic tensor."""
    verify_scheme(scheme)
    factors = []
    for name in ("left", "right", "output"):
        columns = []
        for column in getattr(scheme, name):
            entries = []
            for i, value in column:
                real, imag = coefficient_pair(value)
                if imag:
                    raise ValueError("construction/search currently supports Q, not Q(i)")
                entries.append((i, real/scheme.divisor if name == "output" else real))
            columns.append(vector(entries))
        factors.append(columns)
    m, k, n = scheme.shape
    return Tensor((m*k, k*n, m*n), tuple(zip(*factors)))


def to_scheme(tensor, shape, name="generated"):
    """Require the complete matrix identity before exporting an executable recipe."""
    m, k, n = shape
    if tensor.dimensions != (m*k, k*n, m*n):
        raise ValueError("tensor spaces do not match the requested matrix shape")
    factors = tuple(tuple(t[leg] for t in tensor.terms) for leg in range(3))
    square = m == k == n
    return verify_scheme(BilinearScheme(name, m if square else None, *factors,
                                       dimensions=None if square else tuple(shape)))


def tensor_product(a, b):
    """Multiply tensors with lexicographic coordinates and rank at most Ra*Rb."""
    # Multiply every term of a by every term of b, just as distributing
    # (sum a_i)*(sum b_j). Each vector-space basis now consists of pairs of
    # coordinates. No cancellation-based rank reduction is claimed here.
    dimensions = tuple(x*y for x, y in zip(a.dimensions, b.dimensions))
    terms = tuple(tuple(kronecker(x, y, width) for x, y, width in
                        zip(left, right, b.dimensions))
                  for left in a.terms for right in b.terms)
    return Tensor(dimensions, terms)


def tensor_power(tensor, exponent):
    """Materialize a nonnegative tensor power; rank and storage grow exponentially."""
    if type(exponent) is not int or exponent < 0:
        raise ValueError("tensor power must be a nonnegative integer")
    result = Tensor((1, 1, 1), ((((0, 1),),)*3,))
    for _ in range(exponent):
        result = tensor_product(result, tensor)
    return result


def linear_maps(tensor, dimensions, maps):
    """Apply three sparse coordinate maps to a tensor without adding products.

    Each map is a sequence of columns, one for every source coordinate. An
    empty column deletes a variable. Several columns may identify variables;
    coefficients allow general rational linear restrictions, not only zeroing.
    """
    if len(dimensions) != 3 or any(type(d) is not int or d < 1 for d in dimensions):
        raise ValueError("linear maps need three positive destination dimensions")
    if len(maps) != 3 or any(len(mapping) != n for mapping, n in zip(maps, tensor.dimensions)):
        raise ValueError("linear maps must cover every source coordinate")
    canonical = tuple(tuple(vector(column) for column in mapping) for mapping in maps)
    if any(any(i >= d for column in mapping for i, _ in column)
           for mapping, d in zip(canonical, dimensions)):
        raise ValueError("linear map has an out-of-range destination")
    # A source coefficient u at basis coordinate i maps to u times column i.
    # Accumulate all such contributions within each factor. Applying a linear
    # map to each factor preserves rank-one structure, though zero factors may
    # delete terms and several mapped terms can become algebraically dependent.
    terms = tuple(tuple(vector((j, u*v) for i, u in factor for j, v in mapping[i])
                        for factor, mapping in zip(term, canonical)) for term in tensor.terms)
    return Tensor(tuple(dimensions), terms)


def direct_sum(a, b):
    """Place tensors in disjoint spaces; this is not a single larger matrix product."""
    shifted = tuple(tuple(tuple((i+offset, c) for i, c in factor)
                          for factor, offset in zip(term, a.dimensions)) for term in b.terms)
    return Tensor(tuple(x+y for x, y in zip(a.dimensions, b.dimensions)), a.terms+shifted)


def restrict_coordinates(tensor, selections):
    """Keep ordered coordinate subsets, deleting others and densely reindexing.

    This is a building block for zeroing/extraction from tensor powers. The
    restriction alone does not establish that a surviving component is matrix
    multiplication; ``to_scheme`` must check that identity before execution.
    """
    if len(selections) != 3 or any(not selected for selected in selections):
        raise ValueError("coordinate restrictions need three nonempty selections")
    maps = []
    for selected, bound in zip(selections, tensor.dimensions):
        if (len(set(selected)) != len(selected)
                or any(type(i) is not int or not 0 <= i < bound for i in selected)):
            raise ValueError("selected coordinates must be distinct and in range")
        destinations = {source: target for target, source in enumerate(selected)}
        maps.append(tuple(((destinations[i], 1),) if i in destinations else () for i in range(bound)))
    return linear_maps(tensor, tuple(map(len, selections)), maps)


def matrix_product(a, shape_a, b, shape_b):
    """Tensor two matrix recipes and reindex into the combined row-major matrices.

    Generic tensor coordinates pair entire flattened matrices. Multiplication
    needs paired rows followed by paired columns; the explicit permutation
    below is essential and is verified on conversion back to a recipe.
    """
    to_scheme(a, shape_a)
    to_scheme(b, shape_b)
    paired = tensor_product(a, b)
    ma, ka, na = shape_a
    mb, kb, nb = shape_b
    maps = []
    for rows_a, cols_a, rows_b, cols_b in ((ma, ka, mb, kb), (ka, na, kb, nb), (ma, na, mb, nb)):
        mapping = []
        for ia in range(rows_a*cols_a):
            ra, ca = divmod(ia, cols_a)
            for ib in range(rows_b*cols_b):
                rb, cb = divmod(ib, cols_b)
                # Generic Kronecker order groups (ra, ca) before (rb, cb).
                # A combined matrix instead groups its row (ra, rb) before its
                # column (ca, cb). This permutation is what turns the generic
                # tensor product into the larger matrix multiplication tensor.
                mapping.append((((ra*rows_b+rb)*(cols_a*cols_b)+ca*cols_b+cb, 1),))
        maps.append(tuple(mapping))
    return linear_maps(paired, paired.dimensions, maps)


def coppersmith_winograd(q):
    """Construct CW_q literally with 3q+3 products; do not label its rank q+2.

    Variables 0, 1..q, q+1 belong to the three laser-method groups. Their
    overlap is retained: the six sectors are not independent direct summands.
    The smaller q+2 border/asymptotic bound needs a degeneration and extraction
    argument, neither supplied by this literal constructor.
    """
    if type(q) is not int or q < 1:
        raise ValueError("CW parameter q must be a positive integer")
    coordinates = [(0, 0, q+1), (0, q+1, 0), (q+1, 0, 0)]
    for i in range(1, q+1):
        coordinates.extend(((0, i, i), (i, 0, i), (i, i, 0)))
    return Tensor((q+2,)*3, tuple(tuple(((i, 1),) for i in indices) for indices in coordinates))


def polynomial_coefficient(dimensions, terms, degree, coefficient):
    """Recover an exact tensor coefficient from a polynomial rank decomposition.

    Input vector entries are (coordinate, (c0,c1,...)) rational polynomials.
    The caller supplies a degree upper bound for the *whole tensor polynomial*;
    we check it from the factors rather than assuming absent high terms. Exact
    interpolation at 0..degree avoids numerical epsilon limits. It uses at most
    (degree+1)*rank products, explicitly paying the recovery overhead. This is
    a primitive for degenerations, not a finite realization of border rank.
    """
    if (type(degree) is not int or degree < 0 or type(coefficient) is not int
            or not 0 <= coefficient <= degree):
        raise ValueError("coefficient must lie within the nonnegative degree bound")
    if len(dimensions) != 3 or any(type(d) is not int or d < 1 for d in dimensions):
        raise ValueError("polynomial tensor needs three positive dimensions")
    terms = tuple(tuple(tuple((i, tuple(poly)) for i, poly in factor) for factor in term)
                  for term in terms)
    for term in terms:
        if len(term) != 3 or any(not factor for factor in term):
            raise ValueError("polynomial terms need three nonempty factors")
        degrees = []
        for factor, bound in zip(term, dimensions):
            powers = []
            for index, poly in factor:
                # Check the source coordinate even when its polynomial is zero
                # at every used interpolation point. Evaluation must not silently
                # erase a malformed coordinate and make invalid input look valid.
                if type(index) is not int or not 0 <= index < bound:
                    raise ValueError("polynomial coordinate outside its vector space")
                if not poly or any(type(c) not in (int, Fraction) for c in poly):
                    raise ValueError("polynomial coefficients must be exact rationals")
                powers.extend(j for j, c in enumerate(poly) if c)
            degrees.append(max(powers, default=0))
        if sum(degrees) > degree:
            raise ValueError("degree bound is too small for a rank-one polynomial term")
    recovered = []
    for point in range(degree+1):
        # Build this Lagrange basis polynomial by multiplying linear factors.
        basis, denominator = [Fraction(1)], Fraction(1)
        for other in range(degree+1):
            if other == point:
                continue
            updated = [Fraction(0)]*(len(basis)+1)
            for j, c in enumerate(basis):
                updated[j] -= other*c
                updated[j+1] += c
            basis = updated
            denominator *= point-other
        weight = basis[coefficient]/denominator
        # Lagrange interpolation says P(e)=sum P(point)*L_point(e). Taking
        # the requested coefficient gives a weighted sum of evaluated tensors.
        # Put the weight in W only: scaling all three factors would cube it.
        if not weight:
            continue
        for term in terms:
            factors = []
            for leg, factor in enumerate(term):
                entries = ((i, sum(Fraction(c)*point**j for j, c in enumerate(poly)))
                           for i, poly in factor)
                evaluated = vector(entries)
                factors.append(vector((i, weight*c) for i, c in evaluated) if leg == 2 else evaluated)
            recovered.append(tuple(factors))
    return Tensor(tuple(dimensions), tuple(recovered))


def coefficient_expansion(dimensions, terms, coefficient):
    """Recover a polynomial coefficient by explicit convolution of vector planes.

    Write each factor as sum(e**a * U_a). Its rank-one product contributes
    U_a tensor V_b tensor W_c when a+b+c equals the requested degree. Each
    nonzero triple is a finite bilinear product; its cost is counted rather than
    hidden in a border-rank annotation. Sparse degrees can make this cheaper than
    interpolation, whereas dense polynomial factors can make it more expensive.
    Subsequent exact dependency reductions may eliminate redundant terms.

    This is independent of Lagrange recovery: no evaluations, small epsilon,
    target tensor entries, or classical multiplication recipe are used to build
    the result. Every coordinate is validated even if its polynomial is zero.
    """
    if type(coefficient) is not int or coefficient < 0:
        raise ValueError("coefficient degree must be a nonnegative integer")
    if len(dimensions) != 3 or any(type(d) is not int or d < 1 for d in dimensions):
        raise ValueError("polynomial tensor needs three positive dimensions")
    expanded = []
    for term in terms:
        if len(term) != 3 or any(not factor for factor in term):
            raise ValueError("polynomial terms need three nonempty factors")
        planes = []
        for factor, bound in zip(term, dimensions):
            entries = {}
            for index, polynomial in factor:
                if type(index) is not int or not 0 <= index < bound:
                    raise ValueError("polynomial coordinate outside its vector space")
                if not polynomial or any(type(c) not in (int, Fraction) for c in polynomial):
                    raise ValueError("polynomial coefficients must be exact rationals")
                for degree, value in enumerate(polynomial):
                    if value and degree <= coefficient:
                        entries.setdefault(degree, []).append((index, value))
            # Collect duplicate coordinates before deciding whether a plane is
            # zero. A cancellation is exact construction algebra, not a shortcut
            # based on a particular matrix's input values.
            factor_planes = {}
            for degree, values in entries.items():
                plane = vector(values)
                if plane:
                    factor_planes[degree] = plane
            planes.append(factor_planes)
        for a, u in sorted(planes[0].items()):
            for b, v in sorted(planes[1].items()):
                w = planes[2].get(coefficient-a-b)
                if w:
                    expanded.append((u, v, w))
    return Tensor(tuple(dimensions), tuple(expanded))
