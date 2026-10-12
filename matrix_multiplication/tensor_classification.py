"""Recognize ordinary matrix tensors by exact, reproducible basis permutations.

A space dimension is not a matrix shape, and a matching dimension is not an
identity proof. This deliberately narrow recognizer first checks dimensions,
then reconstructs matrix indices from coefficient-one support neighborhoods,
and finally checks every coefficient against an independently generated matrix
tensor. It accepts arbitrary permutations of row-major bases, including products
with unit axes. It does not search general linear changes of basis or prove
non-isomorphism when its neighborhood test fails.

Classification is construction work. Executing a recognized recipe still uses
the shared observable bilinear primitives, with its original product costs.
"""

from dataclasses import dataclass
from math import isqrt

from .tensors import classical, restrict_coordinates


@dataclass(frozen=True)
class MatrixLayout:
    """A proved matrix shape and source coordinates in A/B/C row-major order."""

    shape: tuple
    selections: tuple

    def document(self):
        """Export the permutation witness rather than just a successful label."""
        return {"shape": list(self.shape),
                "coordinates_in_matrix_order": [list(s) for s in self.selections],
                "complete_identity_verified": True}


def dimension_shape(dimensions):
    """Infer the unique positive integer (m,k,n) with spaces (mk,kn,mn).

    The relation m*m = X*Z/Y is necessary, including when an axis is one.
    Integer square roots avoid rounding a nearly-square large integer. A failed
    check rules out a full-space ordinary matrix tensor of these dimensions;
    smaller restrictions or different constructions may still be useful.
    """
    x, y, z = dimensions
    square, remainder = divmod(x*z, y)
    m = isqrt(square)
    if remainder or m < 1 or m*m != square or x % m or z % m:
        return None
    shape = (m, x//m, z//m)
    return shape if shape[1]*shape[2] == y else None


def neighborhood_groups(neighbors):
    """Group coordinates with identical neighbors in deterministic source order.

    In A, equal B neighborhoods mean a shared inner index, while equal C
    neighborhoods mean a shared row. In B, equal C neighborhoods mean a shared
    column. These equivalence classes recover labels without guessing them.
    """
    groups = {}
    for coordinate, adjacent in enumerate(neighbors):
        groups.setdefault(frozenset(adjacent), []).append(coordinate)
    return tuple(tuple(group) for group in groups.values())


def classify_matrix_tensor(tensor):
    """Return a verified layout, or bounded evidence explaining non-recognition.

    The result is (layout_or_None, JSON-compatible evidence). Failure after the
    dimension check is explicitly limited to this coefficient-one permutation
    recognizer. Most importantly, a candidate mapping is accepted only after a
    complete tensor comparison, so incomplete incidence patterns cannot pass.
    """
    shape = dimension_shape(tensor.dimensions)
    evidence = {"method": "coefficient-one support neighborhoods and full basis-permutation identity",
                "dimensions": list(tensor.dimensions), "recognized": False,
                "general_linear_isomorphism_searched": False}
    if shape is None:
        evidence.update(status="dimension obstruction",
                        reason="These full spaces cannot equal (m*k,k*n,m*n) for positive integer matrix dimensions.")
        return None, evidence
    evidence["dimension_candidate_shape"] = list(shape)
    support = tensor.expand()
    if not support or any(value != 1 for value in support.values()):
        evidence.update(status="not recognized",
                        reason="This recognizer requires nonempty coefficient-one support; scaling and general basis changes remain unsearched.")
        return None, evidence
    # neighbors[a][b][i] contains all b-leg coordinates joined to coordinate i
    # on leg a. Keeping both directions makes the witness independently readable.
    neighbors = [[ [set() for _ in range(tensor.dimensions[a])] for b in range(3)]
                 for a in range(3)]
    for triple in support:
        for a in range(3):
            for b in range(3):
                if a != b:
                    neighbors[a][b][triple[a]].add(triple[b])
    rows = neighborhood_groups(neighbors[0][2])
    inners = neighborhood_groups(neighbors[0][1])
    columns = neighborhood_groups(neighbors[1][2])
    if tuple(map(len, (rows, inners, columns))) != shape:
        evidence.update(status="not recognized",
                        reason="Support neighborhoods do not yield the required row, inner and column classes; general basis changes remain unsearched.")
        return None, evidence
    row_labels = {coordinate: i for i, group in enumerate(rows) for coordinate in group}
    inner_labels = {coordinate: j for j, group in enumerate(inners) for coordinate in group}
    column_labels = {coordinate: ell for ell, group in enumerate(columns) for coordinate in group}
    # Opposite-leg neighborhoods must equal whole classes, not merely overlap.
    # Failure is evidence about this recognizer, never a global impossibility.
    row_lookup = {frozenset(group): i for i, group in enumerate(rows)}
    inner_lookup = {frozenset(group): j for j, group in enumerate(inners)}
    column_lookup = {frozenset(group): ell for ell, group in enumerate(columns)}
    selections = [[None]*dimension for dimension in tensor.dimensions]
    m, k, n = shape
    for leg, dimension in enumerate(tensor.dimensions):
        for coordinate in range(dimension):
            if leg == 0:
                destination = row_labels[coordinate]*k+inner_labels[coordinate]
            elif leg == 1:
                j = inner_lookup.get(frozenset(neighbors[1][0][coordinate]))
                if j is None:
                    break
                destination = j*n+column_labels[coordinate]
            else:
                i = row_lookup.get(frozenset(neighbors[2][0][coordinate]))
                ell = column_lookup.get(frozenset(neighbors[2][1][coordinate]))
                if i is None or ell is None:
                    break
                destination = i*n+ell
            if selections[leg][destination] is not None:
                break
            selections[leg][destination] = coordinate
        if any(value is None for value in selections[leg]):
            evidence.update(status="not recognized",
                            reason="Neighborhood classes do not form bijective matrix entry labels; general basis changes remain unsearched.")
            return None, evidence
    layout = MatrixLayout(shape, tuple(tuple(s) for s in selections))
    candidate = restrict_coordinates(tensor, layout.selections)
    if candidate.expand() != classical(shape).expand():
        evidence.update(status="not recognized",
                        reason="The candidate permutation fails the complete matrix tensor identity; general basis changes remain unsearched.")
        return None, evidence
    evidence.update(status="verified matrix product", recognized=True, **layout.document())
    return layout, evidence
