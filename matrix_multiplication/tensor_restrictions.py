"""Bounded coordinate extraction with reusable, independently replayable maps.

A restriction keeps an ordered subset of each of a tensor's three spaces.
The witness records those subsets in matrix row-major order, so later products
can recombine parts without repeating recognition or guessing index layouts.
Search is deliberately limited to coefficient-one basis permutations; it is
neither a rank minimizer nor a general linear-isomorphism decision procedure.
"""

from dataclasses import dataclass
from itertools import combinations, product
from math import prod

from .tensor_classification import classify_matrix_tensor, dimension_shape
from .tensors import Tensor, restrict_coordinates


@dataclass(frozen=True)
class RestrictionWitness:
    """Immutable source dimensions, matrix shape and ordered source selections."""

    source_dimensions: tuple
    shape: tuple
    selections: tuple

    def __post_init__(self):
        """Snapshot maps and reject ambiguous, repeated or out-of-range indices."""
        dimensions, shape = tuple(self.source_dimensions), tuple(self.shape)
        selections = tuple(tuple(s) for s in self.selections)
        if any(len(values) != 3 for values in (dimensions, shape, selections)):
            raise ValueError("restriction witnesses require three spaces and matrix dimensions")
        if any(type(n) is not int or n < 1 for n in (*dimensions, *shape)):
            raise ValueError("witness dimensions must be positive integers")
        m, k, n = shape
        if tuple(map(len, selections)) != (m*k, k*n, m*n):
            raise ValueError("witness selections do not fit the matrix shape")
        for selected, bound in zip(selections, dimensions):
            if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < bound for i in selected):
                raise ValueError("witness coordinates must be distinct integers within source spaces")
        object.__setattr__(self, "source_dimensions", dimensions)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "selections", selections)

    def document(self):
        """Serialize maps without claiming verification merely from valid syntax."""
        return {"schema": "matrix-restriction-witness-v1",
                "source_dimensions": list(self.source_dimensions), "shape": list(self.shape),
                "source_coordinates_in_matrix_order": [list(s) for s in self.selections]}


def verify_restriction(source, witness):
    """Check all coefficients independently of the neighborhood recognizer.

    Enumerate the actual restriction from the source coefficient dictionary,
    then compare against the explicit A[i,j] B[j,l] C[i,l] definition. This
    checks missing coefficients, extra cross-products and nonunit coefficients.
    The source decomposition is retained by restriction for subsequent recovery.
    """
    if source.dimensions != witness.source_dimensions:
        raise ValueError("restriction witness belongs to different source spaces")
    maps = tuple({index: position for position, index in enumerate(s)} for s in witness.selections)
    actual = {tuple(mapping[i] for mapping, i in zip(maps, indices)): value
              for indices, value in source.expand().items()
              if all(i in mapping for mapping, i in zip(maps, indices))}
    m, k, n = witness.shape
    expected = {(i*k+j, j*n+ell, i*n+ell): 1
                for i in range(m) for j in range(k) for ell in range(n)}
    if actual != expected:
        raise ValueError("restriction witness fails the complete matrix coefficient identity")
    return restrict_coordinates(source, witness.selections)


def compose_selections(outer, inner):
    """Compose ordered coordinate restrictions: final -> intermediate -> source."""
    if len(outer) != 3 or len(inner) != 3:
        raise ValueError("coordinate composition needs three legs")
    result = []
    for source, selected in zip(outer, inner):
        if (not source or not selected or len(set(source)) != len(source)
                or any(type(i) is not int or i < 0 for i in source)
                or len(set(selected)) != len(selected)
                or any(type(i) is not int or not 0 <= i < len(source) for i in selected)):
            raise ValueError("invalid nested coordinate selections")
        result.append(tuple(source[i] for i in selected))
    return tuple(result)


def product_witness(left, right):
    """Combine matrix restrictions inside a generic tensor product's coordinates.

    Generic pairing flattens (source_a, source_b). Matrix ordering instead pairs
    rows first and columns second. The four nested loops explicitly implement
    that permutation; no multiplication products are added by changing indices.
    The returned map must still pass verify_restriction on the actual product.
    """
    ma, ka, na = left.shape
    mb, kb, nb = right.shape
    selections = []
    for leg, (ra, ca, rb, cb) in enumerate(((ma, ka, mb, kb), (ka, na, kb, nb), (ma, na, mb, nb))):
        selections.append(tuple(left.selections[leg][i*ca+j]*right.source_dimensions[leg]
                                +right.selections[leg][u*cb+v]
                                for i in range(ra) for u in range(rb)
                                for j in range(ca) for v in range(cb)))
    return RestrictionWitness(tuple(a*b for a, b in zip(left.source_dimensions, right.source_dimensions)),
                              tuple(a*b for a, b in zip(left.shape, right.shape)), tuple(selections))


def recover_restriction(source, witness, polynomial_terms, leading):
    """Recover a witnessed part from a polynomial recipe on the source spaces.

    Restrict before recovery, retain every finite interpolation product, compare
    coefficient expansion and replayed reductions, then independently verify the
    chosen recipe against the matrix definition. Imports are local to keep the
    coordinate-map primitives usable without initializing the recovery layer.
    """
    from .tensors import classical
    verify_restriction(source, witness)
    recovered, evidence = recover_tensor_restriction(source, witness.selections,
        classical(witness.shape), polynomial_terms, leading)
    evidence.update(classical_products=prod(witness.shape), complete_matrix_identity_verified=True)
    return recovered, evidence


def recover_tensor_restriction(source, selections, expected, polynomial_terms, leading):
    """Recover any completely checked target, including independent direct sums.

    The literal source and expected specification must agree on every coefficient
    before recovery. Interpolation and replayed coefficient-plane reductions pay
    their full finite costs. The specification is never used as the executable
    recipe. A coordinate union alone does not establish component independence.
    """
    from .cw_extraction import optimize_recovery, restrict_polynomial
    from .degenerations import polynomial_degree_bound
    from .tensors import polynomial_coefficient

    literal = restrict_coordinates(source, selections)
    if literal.dimensions != expected.dimensions or literal.expand() != expected.expand():
        raise ValueError("source restriction fails the complete target identity")
    restricted = restrict_polynomial(polynomial_terms, selections)
    degree = polynomial_degree_bound(restricted)
    interpolated = polynomial_coefficient(expected.dimensions, restricted, degree, leading)
    target = expected.expand()
    if interpolated.expand() != target:
        raise ValueError("polynomial restriction fails the complete recovered matrix identity")
    recovered, evidence = optimize_recovery(interpolated, restricted, leading)
    if recovered.expand() != target:
        raise ValueError("optimized restriction fails the complete target identity")
    evidence.update(degree_bound=degree, coefficient=leading,
                    retained_polynomial_products=len(restricted))
    return recovered, evidence


def recognize_restriction(source_dimensions, support, selections):
    """Recognize a subset from exact coefficients without compiling a new recipe.

    Singleton terms below are classification data only, never a substitute for
    paid recovery from the original polynomial. Callers independently check any
    proposed witness against the original source decomposition before admission.
    """
    maps = tuple({i: j for j, i in enumerate(s)} for s in selections)
    shape = dimension_shape(tuple(map(len, selections)))
    if shape is None:
        return None, "dimension obstruction"
    coefficients = {tuple(mapping[i] for mapping, i in zip(maps, indices)): c
                    for indices, c in support.items()
                    if all(i in mapping for mapping, i in zip(maps, indices))}
    if len(coefficients) != prod(shape):
        return None, "support count mismatch"
    literal = Tensor(tuple(map(len, selections)), tuple(
        (((i, c),), ((j, 1),), ((k, 1),)) for (i, j, k), c in coefficients.items()))
    layout, evidence = classify_matrix_tensor(literal)
    if layout is None:
        return None, evidence["status"]
    return RestrictionWitness(source_dimensions, layout.shape,
                              compose_selections(selections, layout.selections)), evidence["status"]


def coordinate_candidates(dimensions):
    """Yield non-scalar, dimension-compatible subsets, largest volume first.

    Only sizes (mk,kn,mn) can hold a matrix tensor. Shapes and each leg's sorted
    subsets have deterministic order. Combination iterators stay lazy, including
    when a source power has far more subsets than the caller's finite budget.
    """
    sizes = []
    for x, y, z in product(*(range(1, d+1) for d in dimensions)):
        shape = dimension_shape((x, y, z))
        if shape is not None and prod(shape) > 1:
            sizes.append((shape, (x, y, z)))
    for shape, sizes_for_shape in sorted(sizes, key=lambda item: (-prod(item[0]), max(item[0])-min(item[0]), item[0])):
        # itertools.product would eagerly cache every combination pool. Nested
        # loops avoid materializing enormous subset lists for a bounded search.
        for x in combinations(range(dimensions[0]), sizes_for_shape[0]):
            for y in combinations(range(dimensions[1]), sizes_for_shape[1]):
                for z in combinations(range(dimensions[2]), sizes_for_shape[2]):
                    yield shape, (x, y, z)


def search_restrictions(source, *, candidate_budget=512, max_parts=4):
    """Enumerate a bounded prefix, retaining every miss and reusable witnesses.

    A support-size filter rejects candidates cheaply; survivors undergo complete
    classification and an independent coefficient check. max_parts bounds saved
    recipes, not attempts or successful-candidate counts. Exhaustion describes
    only this coordinate/permutation candidate family, never general extraction.
    """
    if type(candidate_budget) is not int or candidate_budget < 0 or type(max_parts) is not int or max_parts < 1:
        raise ValueError("candidate budget must be nonnegative and max_parts positive")
    support, records, witnesses = source.expand(), [], []
    candidates = iter(coordinate_candidates(source.dimensions))
    exhausted = False
    hits = 0
    for _ in range(candidate_budget):
        candidate = next(candidates, None)
        if candidate is None:
            exhausted = True
            break
        shape, selections = candidate
        record = {"shape": list(shape), "selected_coordinates": [list(s) for s in selections]}
        witness, record["status"] = recognize_restriction(source.dimensions, support, selections)
        if witness is not None:
            verify_restriction(source, witness)
            hits += 1
            record["complete_identity_verified"] = True
            if len(witnesses) < max_parts:
                record["saved_part_index"] = len(witnesses)
                witnesses.append(witness)
        records.append(record)
    if not exhausted:
        exhausted = next(candidates, None) is None
    return tuple(witnesses), {"candidate_budget": candidate_budget, "max_saved_parts": max_parts,
        "attempted_candidates": len(records), "recognized_candidates": hits,
        "candidate_family_exhausted": exhausted, "budget_exhausted": not exhausted,
        "candidate_order": "descending classical volume, balance, shape, lexicographic coordinate subsets",
        "candidates": records, "limitation": "Coordinate subsets and coefficient-one basis permutations only; no general linear maps or minimum-rank claim."}
