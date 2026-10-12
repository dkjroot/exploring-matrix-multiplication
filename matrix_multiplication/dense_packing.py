"""Pack a verified direct sum of matrix products into a dense matrix identity.

The dense product consists of triples (row, inner, column). A component can
compute a rectangular tile of those triples. We partition this volume, assign
tiles to component slots in successive batches, zero-pad partial slots, reuse
the appropriate input entries, and sum the corresponding output entries.

Packing is implemented as explicit linear maps on an already verified bilinear
recipe. Thus all coefficient products, additions, and surviving data products
remain visible in the exported matrix scheme and its existing execution engine.
Every logical dense-product triple must occur exactly once; the final complete
tensor identity is checked independently of that combinatorial coverage check.

This finite tiling construction is independently written elementary algebra.
It is not the large-power conversion behind Schonhage's asymptotic sum
inequality. No exponent is inferred from a sum inequality or border rank. The
actual exported dense recipe determines any recursive exponent in the report.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod

from .cw_extraction import optimize_cw_extraction
from .cw_coarsening import extract_coarse_cw
from .decomposition_search import reduce_all, replay
from .instrumentation import Instrumentation
from .tensor_execution import evaluate_tensor
from .tensors import Tensor, classical, direct_sum, linear_maps
from .verification import check_product


def matrix_shape(shape):
    """Snapshot three positive exact dimensions in row/inner/column order."""
    shape = tuple(shape)
    if len(shape) != 3 or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError("matrix shapes need three positive integer dimensions")
    return shape


def volume(origin, extent):
    """Enumerate a tile's dense-product triples in row-major lexicographic order.

    The inner coordinate is a summation index, not an output column. Two tiles
    can contribute to the same C entry when their inner ranges differ. They must
    never compute the same (row,inner,column) triple twice.
    """
    return tuple(product(*(range(start, start+width) for start, width in zip(origin, extent))))


@dataclass(frozen=True)
class Tile:
    """A component slot's starting triple and its actual unpadded dimensions."""

    origin: tuple
    extent: tuple

    def __post_init__(self):
        """Keep immutable validated coordinates even when callers supply lists."""
        origin = tuple(self.origin)
        if len(origin) != 3 or any(type(i) is not int or i < 0 for i in origin):
            raise ValueError("tile origins need three nonnegative integers")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "extent", matrix_shape(self.extent))

    def document(self):
        """Store enough index information to reconstruct a packing without search."""
        return {"origin": list(self.origin), "extent": list(self.extent)}


def packing_schedule(shapes, target):
    """Greedily cover a dense product with component-shaped tiles in batches.

    Every slot starts at the lexicographically first uncovered triple. Try all
    fitting extents there, largest volume first, and accept only an entirely
    uncovered tile. A singleton always fits, so every nonempty batch progresses.
    This guarantees termination and correct coverage, not an optimal packing.
    The deliberately materialized volume is suitable for small reference cases;
    succinct large-power packing remains separate research work.
    """
    shapes = tuple(matrix_shape(shape) for shape in shapes)
    target = matrix_shape(target)
    if not shapes:
        raise ValueError("packing needs at least one matrix component")
    uncovered = set(volume((0, 0, 0), target))
    batches = []
    while uncovered:
        batch = []
        for shape in shapes:
            if not uncovered:
                batch.append(None)
                continue
            origin = min(uncovered)
            maxima = tuple(min(width, bound-start) for width, bound, start in zip(shape, target, origin))
            candidates = sorted(product(*(range(1, width+1) for width in maxima)),
                                key=lambda extent: (prod(extent), extent), reverse=True)
            for extent in candidates:
                triples = volume(origin, extent)
                if all(triple in uncovered for triple in triples):
                    batch.append(Tile(origin, extent))
                    uncovered.difference_update(triples)
                    break
            else:
                raise AssertionError("the first uncovered triple must fit a singleton tile")
        batches.append(tuple(batch))
    return tuple(batches)


def verify_schedule(shapes, target, batches):
    """Independently reject holes, overlap, out-of-bounds tiles, and wrong slots.

    Unlike the greedy planner's mutable uncovered set, this check accumulates a
    multiplicity for every triple in a supplied schedule. External or future
    planners use this same check before any coefficient maps are accepted.
    """
    shapes = tuple(matrix_shape(shape) for shape in shapes)
    target = matrix_shape(target)
    if not shapes or not batches:
        raise ValueError("packing needs nonempty shapes and batches")
    coverage = {}
    for batch in batches:
        if len(batch) != len(shapes) or not any(tile is not None for tile in batch):
            raise ValueError("each packing batch needs the declared slots and an active tile")
        for tile, shape in zip(batch, shapes):
            if tile is None:
                continue
            if not isinstance(tile, Tile):
                raise ValueError("packing slots must contain validated Tile objects or None")
            if any(width > capacity or start+width > bound
                   for start, width, capacity, bound in zip(tile.origin, tile.extent, shape, target)):
                raise ValueError("packing tile exceeds its component or dense target")
            for triple in volume(tile.origin, tile.extent):
                coverage[triple] = coverage.get(triple, 0)+1
    expected = set(volume((0, 0, 0), target))
    if set(coverage) != expected or any(count != 1 for count in coverage.values()):
        raise ValueError("packing must cover every dense-product triple exactly once")
    return True


def batch_maps(shapes, target, batch):
    """Encode one batch's zero-padding, input reuse, and output accumulation.

    Each source basis entry is one entry in one component's A/B/C matrix. It
    maps either to a dense matrix entry with coefficient one or to zero. Several
    source entries may map to the same destination: this reuses inputs and sums
    independent inner-range contributions. No matrix multiplication is performed
    while constructing these maps; linear_maps applies them to the source recipe.
    """
    m, k, n = target
    maps = [[], [], []]
    for shape, tile in zip(shapes, batch):
        a, b, c = shape
        for leg, (rows, columns) in enumerate(((a, b), (b, c), (a, c))):
            for row in range(rows):
                for column in range(columns):
                    if tile is None:
                        maps[leg].append(())
                        continue
                    i, j, ell = tile.origin
                    x, y, z = tile.extent
                    row_limit, col_limit = ((x, y), (y, z), (x, z))[leg]
                    if row >= row_limit or column >= col_limit:
                        maps[leg].append(())
                        continue
                    # The three destination flattenings are respectively A_ij,
                    # B_jl and C_il. Keeping the inner index out of C is precisely
                    # what makes different inner tiles add into the same output.
                    destination = ((i+row)*k+j+column if leg == 0 else
                                   (j+row)*n+ell+column if leg == 1 else
                                   (i+row)*n+ell+column)
                    maps[leg].append(((destination, 1),))
    return tuple(tuple(mapping) for mapping in maps)


def pack_tensor(source, shapes, target, *, batches=None):
    """Construct a verified dense recipe from any exact independent-product sum.

    First verify the source against the declared shapes, not merely its vector
    space dimensions. Then verify the schedule and apply its explicit maps. Each
    batch costs at most source.rank data products; zero padding or exact mapped
    cancellations may remove terms. The final term count is the executable finite
    rank upper bound, and includes every retained batch's recovery overhead.
    """
    shapes = tuple(matrix_shape(shape) for shape in shapes)
    target = matrix_shape(target)
    if not shapes:
        raise ValueError("packing needs at least one matrix component")
    expected_source = classical(shapes[0])
    for shape in shapes[1:]:
        expected_source = direct_sum(expected_source, classical(shape))
    if source.dimensions != expected_source.dimensions or source.expand() != expected_source.expand():
        raise ValueError("source recipe is not the declared independent matrix-product sum")
    automatic = batches is None
    batches = packing_schedule(shapes, target) if automatic else tuple(tuple(batch) for batch in batches)
    verify_schedule(shapes, target, batches)
    m, k, n = target
    dimensions = (m*k, k*n, m*n)
    terms, batch_products = [], []
    for batch in batches:
        mapped = linear_maps(source, dimensions, batch_maps(shapes, target, batch))
        terms.extend(mapped.terms)
        batch_products.append(mapped.rank)
    # Batches share dense matrix spaces, so add their terms in those spaces.
    # direct_sum would instead allocate separate outputs and give a wrong tensor.
    mapped_recipe = Tensor(dimensions, tuple(terms))
    # Identifying input entries can create new two-factor dependencies between
    # mapped terms, even when the direct-sum source had none. Compile those exact
    # reductions into the final recipe, and replay them to check the chain.
    # Executing this final recipe needs no runtime search or tensor expansion.
    packed, reductions = reduce_all(mapped_recipe)
    if replay(mapped_recipe, reductions) != packed:
        raise AssertionError("dense packing reductions failed exact replay")
    if packed.expand() != classical(target).expand():
        raise AssertionError("packed recipe failed its complete dense matrix identity")
    evidence = {"schema": "finite-dense-packing-v1", "target_shape": list(target),
                "component_shapes": [list(shape) for shape in shapes],
                "selection_method": "greedy finite volume tiling; no optimality claim" if automatic else "explicit supplied schedule",
                "batches": [[tile.document() if tile else None for tile in batch] for batch in batches],
                "batch_count": len(batches), "batch_products": batch_products,
                "source_products_per_batch": source.rank,
                "batch_product_bound": len(batches)*source.rank,
                "mapped_products_before_reductions": mapped_recipe.rank,
                "packing_reduction_events": reductions, "packing_replay_verified": True,
                "packed_products": packed.rank, "classical_dense_products": prod(target),
                "source_direct_sum_identity_verified": True,
                "exact_volume_coverage_verified": True, "dense_matrix_identity_verified": True,
                "packing_kind": "explicit finite tiles and linear maps; not the asymptotic sum inequality conversion"}
    return packed, evidence


def packed_execution_evidence(packed, size):
    """Observe an actual signed rational dense call outside construction work.

    Fine and coarse extraction feed the same execution checker. This preserves
    the separation between discovering coefficients and multiplying input data,
    and prevents a new construction from silently losing count verification.
    """
    if type(size) is not int or size < 2:
        raise ValueError("a recursive dense packing base must have integer size at least two")
    if packed.dimensions != (size*size,)*3:
        raise ValueError("dense execution needs the requested square matrix spaces")
    a = [[Fraction(i-2*j+1, 3) for j in range(size)] for i in range(size)]
    b = [[Fraction(2*i+j-1, 5) for j in range(size)] for i in range(size)]
    observer = Instrumentation(profile=False)
    flat = evaluate_tensor(packed, [value for row in a for value in row],
                           [value for row in b for value in row], instrumentation=observer)
    result = [list(flat[i*size:(i+1)*size]) for i in range(size)]
    if not check_product(a, b, result).passed or observer.counts["field_multiplications"] != packed.rank:
        raise AssertionError("packed CW dense execution failed its product or arithmetic count check")
    return {"exact_outputs_verified": True, "observation": observer.snapshot(),
            "kernel": "tensor.evaluate; counts coefficient-one scaling"}


def pack_cw(q, power=1, size=2, words=None):
    """Improve fine CW recovery, pack a square base, and measure a dense call.

    The returned tensor is suitable for to_scheme and recursive Composition.
    Construction, recovery optimization, and packing maps are setup work. Only
    executing the completed recipe is observed; the independent product oracle
    verifies that execution afterward, outside the counter session.
    """
    if type(size) is not int or size < 2:
        raise ValueError("a recursive dense packing base must have integer size at least two")
    source, extraction = optimize_cw_extraction(q, power, words)
    shapes = tuple(tuple(item["shape"]) for item in extraction["components"])
    packed, evidence = pack_tensor(source, shapes, (size,)*3)
    evidence["execution"] = packed_execution_evidence(packed, size)
    evidence["extraction"] = extraction
    evidence["strictly_fewer_products_than_classical"] = packed.rank < size**3
    return packed, evidence


def pack_coarse_cw(q, power=2, size=2):
    """Use proved coarse components while preserving the entire catalog evidence.

    Unrecognized tensors remain in the extraction report; they are not relabelled
    as matrix slots. The final dense scheme is checked and measured by exactly
    the same primitives as fine extraction, making comparisons reproducible.
    """
    if type(size) is not int or size < 2:
        raise ValueError("a recursive dense packing base must have integer size at least two")
    source, extraction = extract_coarse_cw(q, power)
    packed, packing = pack_tensor(source, extraction["selected_component_shapes"], (size,)*3)
    packing["execution"] = packed_execution_evidence(packed, size)
    packing["strictly_fewer_products_than_classical"] = packed.rank < size**3
    extraction["packing"] = packing
    return packed, extraction
