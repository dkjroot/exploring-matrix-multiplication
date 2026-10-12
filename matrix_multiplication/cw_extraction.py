"""Extract and execute completely verified independent products from CW powers.

This is the first finite extraction layer of a laser-method experiment, not an
implementation of its asymptotic sum inequality or dense-product packing. The
six CW sectors have group indices summing to two. A sector word chooses one of
these sectors at each tensor-power position. Its middle coordinates supply row,
inner, or column indices of an ordinary matrix product.

The implementation has three independent checks: group support detects unwanted
cross-sectors; literal CW coefficients verify the selected direct sum; exact
polynomial recovery verifies the executable recipe including interpolation cost.
Small actual calls additionally compare outputs with the independent matrix
oracle. All data arithmetic uses our observable scalar primitives. Construction
arithmetic and these calls have separate scopes.

Mathematical reference: Alman et al., More Asymmetry Yields Faster Matrix
Multiplication, section 2, https://arxiv.org/abs/2404.16349. The polynomial family
and its attribution live in degenerations.py. This code is independently written.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod

from .degenerations import cw_family, polynomial_degree_bound
from .decomposition_search import reduce_all, replay
from .instrumentation import Instrumentation
from .tensors import (classical, coppersmith_winograd, direct_sum,
                      coefficient_expansion, polynomial_coefficient,
                      restrict_coordinates, tensor_power)
from .verification import exact_reference
from .tensor_execution import evaluate_tensor


SECTORS = ((0, 0, 2), (0, 1, 1), (0, 2, 0),
           (1, 0, 1), (1, 1, 0), (2, 0, 0))
# A middle coordinate shared by X and Z is a row index; shared by X and Y
# it is an inner index; shared by Y and Z it is a column index. The singleton
# sectors (002,020,200) contribute no growing matrix index.
AXIS_SECTORS = ((1, 0, 1), (1, 1, 0), (0, 1, 1))
LEG_AXES = ((0, 1), (1, 2), (0, 2))


def sector_word(word):
    """Snapshot and validate a nonempty sequence of the six group triples."""
    word = tuple(tuple(sector) for sector in word)
    if not word or any(len(sector) != 3 or any(type(i) is not int for i in sector)
                       or sector not in SECTORS for sector in word):
        raise ValueError("a sector word needs triples from 002,011,020,101,110,200")
    return word


def validate_parameters(q, power):
    """Reject ambiguous booleans and nonpositive tensor sizes before allocation."""
    if any(type(value) is not int or value < 1 for value in (q, power)):
        raise ValueError("CW q and power must be positive integers")


def digits(value, widths):
    """Decode a flattened row/inner/column index in mixed-radix position order.

    Width-one positions consume no information. Returning a digit at every
    position keeps this representation aligned with the original tensor word.
    The final position is the fastest varying coordinate, as in tensor_power.
    """
    result = [0]*len(widths)
    for position in reversed(range(len(widths))):
        value, result[position] = divmod(value, widths[position])
    return tuple(result)


@dataclass(frozen=True)
class Component:
    """A matrix product's shape and its ordered coordinates in the source power.

    selections[0] is ordered as A's row-major entries, selections[1] as B's,
    and selections[2] as C's. This ordering performs the required permutation;
    simply sorting the retained CW coordinates would generally be incorrect.
    """

    word: tuple
    shape: tuple
    groups: tuple
    selections: tuple

    def document(self):
        """Expose explicit coordinate maps for replay and inspection by humans."""
        return {"word": [''.join(map(str, sector)) for sector in self.word],
                "shape": list(self.shape), "group_words": [list(g) for g in self.groups],
                "source_coordinates_in_matrix_order": [list(s) for s in self.selections],
                "classical_products": prod(self.shape)}


def component(q, word):
    """Derive a sector word's matrix layout without inspecting its coefficients.

    For example, 101 followed by 110 followed by 011 yields a q-by-q square
    product. A position contributes width q to exactly one of row/inner/column.
    Within a leg with group 1, its digit is whichever of that leg's two matrix
    axes grows at that position. Groups 0 and 2 always use singleton coordinates.
    """
    word = sector_word(word)
    validate_parameters(q, len(word))
    widths = tuple(tuple(q if sector == kind else 1 for sector in word)
                   for kind in AXIS_SECTORS)
    shape = tuple(prod(axis) for axis in widths)
    groups = tuple(tuple(sector[leg] for sector in word) for leg in range(3))
    selections = []
    for leg, (row_axis, col_axis) in enumerate(LEG_AXES):
        selected = []
        for row in range(shape[row_axis]):
            for column in range(shape[col_axis]):
                axis_digits = {row_axis: digits(row, widths[row_axis]),
                               col_axis: digits(column, widths[col_axis])}
                coordinate = 0
                for position, sector in enumerate(word):
                    group = sector[leg]
                    if group == 1:
                        axis = AXIS_SECTORS.index(sector)
                        digit = axis_digits[axis][position]+1
                    else:
                        digit = 0 if group == 0 else q+1
                    coordinate = coordinate*(q+2)+digit
                selected.append(coordinate)
        selections.append(tuple(selected))
    return Component(word, shape, groups, tuple(selections))


def selection_conflicts(words):
    """List shared groups and unintended surviving sectors for a proposed sum.

    Distinct X groups, distinct Y groups, and distinct Z groups are necessary
    but insufficient. After zeroing everything else, any combination of retained
    groups whose digits sum to two at every position also survives. Such a
    combination may join X from one desired component to Y and Z from others.
    This support check detects those cross-products before coefficient recovery.
    """
    words = tuple(sector_word(word) for word in words)
    if not words or len({len(word) for word in words}) != 1:
        raise ValueError("select at least one sector word of a common power")
    groups = tuple(tuple(tuple(sector[leg] for sector in word) for word in words)
                   for leg in range(3))
    shared = [leg for leg, selected in enumerate(groups) if len(set(selected)) != len(selected)]
    intended = set(words)
    unexpected = []
    for x, y, z in product(*(sorted(set(g)) for g in groups)):
        candidate = tuple(zip(x, y, z))
        if all(sum(sector) == 2 for sector in candidate) and candidate not in intended:
            unexpected.append(candidate)
    return {"shared_legs": shared, "unexpected_words": unexpected}


def choose_independent_words(q, power):
    """Make a reproducible greedy induced matching, without claiming optimality.

    Prefer products with larger volume, then more balanced dimensions. Every
    accepted addition must leave an induced matching: no shared group in any
    leg and no extra supported combination of retained groups. A greedy miss
    proves nothing about what larger or differently chosen extractions can do.
    The complete 6**power catalog is deliberately finite and materialized here.
    """
    validate_parameters(q, power)
    candidates = [component(q, word) for word in product(SECTORS, repeat=power)]
    candidates.sort(key=lambda item: (-prod(item.shape), max(item.shape)-min(item.shape), item.word))
    chosen = []
    for item in candidates:
        conflicts = selection_conflicts((*chosen, item.word))
        if not conflicts["shared_legs"] and not conflicts["unexpected_words"]:
            chosen.append(item.word)
    return tuple(chosen)


def restrict_polynomial(terms, selections):
    """Zero and reindex factors BEFORE interpolation, retaining rank-one form.

    Taking a coefficient commutes with these coordinate maps. Restricting first
    saves materialization of unwanted coordinates and removes terms with a zero
    factor. It does not discard an inconvenient cancellation: the complete
    resulting tensor is checked against an independently built target below.
    selections come from already validated Components with disjoint spaces.
    """
    maps = tuple({source: destination for destination, source in enumerate(selected)}
                 for selected in selections)
    restricted = []
    for term in terms:
        factors = tuple(tuple((mapping[i], polynomial) for i, polynomial in factor if i in mapping)
                        for factor, mapping in zip(term, maps))
        if all(factors):
            restricted.append(factors)
    return tuple(restricted)


def execution_evidence(tensor, components):
    """Execute simultaneous signed rational pairs and independently check outputs.

    Input/output blocks are concatenated in component order, the same convention
    used by direct_sum. Oracle work occurs outside the instrumented call. This
    example supplements the complete coefficient identity; it does not replace
    that identity with a test on a single pair.
    """
    left = tuple(Fraction(i % 7-3, 3) for i in range(tensor.dimensions[0]))
    right = tuple(Fraction(2-i % 5, 5) for i in range(tensor.dimensions[1]))
    observer = Instrumentation(profile=False)
    result = evaluate_tensor(tensor, left, right, instrumentation=observer)
    expected, offset_a, offset_b = [], 0, 0
    for item in components:
        m, k, n = item.shape
        a = [list(left[offset_a+i*k:offset_a+(i+1)*k]) for i in range(m)]
        b = [list(right[offset_b+j*n:offset_b+(j+1)*n]) for j in range(k)]
        expected.extend(real for row in exact_reference(a, b) for real, imag in row)
        offset_a += m*k
        offset_b += k*n
    if result != tuple(expected):
        raise AssertionError("extracted tensor execution disagrees with independent matrix products")
    if observer.counts["field_multiplications"] != tensor.rank:
        raise AssertionError("extraction execution did not pay every finite recipe product")
    return {"exact_outputs_verified": True, "kernel": "tensor.evaluate; counts coefficient-one scaling",
            "finite_recipe_products": tensor.rank, "observation": observer.snapshot()}


def extract_cw(q, power=1, words=None):
    """Recover a verified direct sum, with explicit maps and fully paid overhead.

    Both the literal restriction and the recovered polynomial recipe must equal
    independently constructed classical products on every tensor coefficient.
    Source-space offsets and row-major selections are recorded for replay.
    There is no inference here from border rank to finite rank, nor from a sum
    of matrix products to one dense matrix multiplication's exponent.
    """
    validate_parameters(q, power)
    automatic = words is None
    words = choose_independent_words(q, power) if automatic else tuple(sector_word(w) for w in words)
    if not words or any(len(word) != power for word in words):
        raise ValueError("selected sector words must match the requested power")
    conflicts = selection_conflicts(words)
    if conflicts["shared_legs"] or conflicts["unexpected_words"]:
        raise ValueError(f"selected components are not an independent induced sum: {conflicts}")
    components = tuple(component(q, word) for word in words)
    selections = tuple(tuple(i for item in components for i in item.selections[leg]) for leg in range(3))
    expected = classical(components[0].shape)
    for item in components[1:]:
        expected = direct_sum(expected, classical(item.shape))
    literal = restrict_coordinates(tensor_power(coppersmith_winograd(q), power), selections)
    if literal.expand() != expected.expand():
        raise AssertionError("CW coordinate restriction failed its complete direct-sum identity")
    dimensions, terms, degree, leading = cw_family(q, power)
    restricted_terms = restrict_polynomial(terms, selections)
    # Restriction can remove high-degree coordinates. Bound the degree of the
    # surviving factors rather than paying for the unreduced 10*power bound.
    # This uses their actual rational coefficients, not a guessed degree or an
    # unproved cancellation. The generic interpolation checker validates it too.
    restricted_degree = polynomial_degree_bound(restricted_terms)
    recovered = polynomial_coefficient(tuple(map(len, selections)), restricted_terms, restricted_degree, leading)
    if recovered.expand() != expected.expand():
        raise AssertionError("restricted CW polynomial recovery failed its complete direct-sum identity")
    # Each evaluated polynomial term uses at most one data product. Zero factors
    # and zero interpolation weights can remove terms, so the actual surviving
    # rank is reported alongside the conservative interpolation bound.
    report = {"schema": "cw-extraction-v1", "q": q, "power": power,
              "catalog_sector_words": 6**power,
              "selection_method": "deterministic greedy induced matching; no optimality claim" if automatic else "explicit sector words",
              "components": [item.document() for item in components],
              "selected_components": len(components), "source_dimensions": list(dimensions),
              "retained_dimensions": list(recovered.dimensions),
              "retained_source_coordinates": [list(s) for s in selections],
              "disjoint_variable_spaces_verified": True, "no_unwanted_cross_sectors_verified": True,
              "literal_direct_sum_identity_verified": True, "recovered_direct_sum_identity_verified": True,
              "polynomial_rank_before_restriction": len(terms),
              "polynomial_rank_after_restriction": len(restricted_terms),
              "original_degree_bound": degree, "degree_bound": restricted_degree,
              "coefficient": leading, "interpolation_points": restricted_degree+1,
              "finite_rank_bound": (restricted_degree+1)*len(restricted_terms),
              "finite_recipe_products": recovered.rank, "classical_direct_sum_products": expected.rank,
              "execution": execution_evidence(recovered, components),
              "dense_matrix_packing_implemented": False,
              "exponent": None,
              "limitation": "A verified sum of independent products; no dense packing or lower-exponent claim. The greedy selection is not optimal.",
              "source": "https://arxiv.org/abs/2404.16349"}
    return recovered, report


def optimize_recovery(interpolated, restricted, leading):
    """Compile and replay cheaper exact coefficient recovery for any restriction.

    Fine sector words and coarser components share this algebra. The caller must
    first verify the interpolated baseline against its independent target. This
    primitive generates coefficient-plane products, checks every coefficient,
    replays dependency reductions, and retains the cheaper verified recipe.
    Measurement belongs to the caller because its output may be a general
    tensor rather than a direct sum of ordinary matrix products.
    """
    expanded = coefficient_expansion(interpolated.dimensions, restricted, leading)
    if expanded.expand() != interpolated.expand():
        raise AssertionError("coefficient expansion disagrees with exact CW interpolation")
    reduced, events = reduce_all(expanded)
    if replay(expanded, events) != reduced or reduced.expand() != interpolated.expand():
        raise AssertionError("CW recovery reductions failed replay or full identity verification")
    use_reduced = reduced.rank <= interpolated.rank
    chosen = reduced if use_reduced else interpolated
    evidence = {
        "interpolation_products": interpolated.rank,
        "coefficient_expansion_products": expanded.rank,
        "reduced_expansion_products": reduced.rank,
        "selected_method": "coefficient expansion with replayed dependency reductions" if use_reduced else "interpolation",
        "selected_products": chosen.rank, "reduction_events": events,
        "reduction_count": len(events), "complete_identity_verified": True,
        "replay_verified": True, "minimum_rank_claim": False}
    return chosen, evidence


def optimize_cw_extraction(q, power=1, words=None):
    """Compare exact recovery routes and replay reductions of coefficient planes.

    The original interpolated recipe remains available as a verified baseline.
    An independently generated coefficient expansion is reduced using our exact
    two-factor dependency primitive, then replayed from its original term order.
    Every coefficient must agree before and after reduction. The classical
    direct sum is used only as the independently verified specification from
    extract_cw, never substituted as the proposed recovery implementation.

    We retain the less expensive recipe by data-product count. This comparison
    does not assert minimum rank or an asymptotic bound. Construction/replay are
    setup work; the chosen recipe's actual data arithmetic is separately observed.
    """
    interpolated, evidence = extract_cw(q, power, words)
    _, terms, _, leading = cw_family(q, power)
    selections = evidence["retained_source_coordinates"]
    restricted = restrict_polynomial(terms, selections)
    chosen, recovery = optimize_recovery(interpolated, restricted, leading)
    components = tuple(component(q, tuple(tuple(map(int, token)) for token in item["word"]))
                       for item in evidence["components"])
    recovery["execution"] = execution_evidence(chosen, components)
    evidence["optimized_recovery"] = recovery
    return chosen, evidence
