"""Exact support-guided matrix restrictions with fair, explicitly bounded work.

For shape (m,k,n), retain mk coordinates of A, kn of B and mn of C.
Every retained C coordinate must have exactly k coefficient-one products;
every retained A coordinate must occur n times, and every B coordinate m times.
These necessary incidence conditions prune proposals, but only complete matrix
recognition plus an independent source check admits a witness. General linear
maps and nonunit coefficient rescaling are deliberately outside this search.
"""

from collections import defaultdict
from itertools import combinations

from .tensor_restrictions import recognize_restriction, verify_restriction


def ordered_coordinates(pool, footprints):
    """Interleave seed footprints, then append unused allowed source coordinates."""
    pool = tuple(pool)
    allowed, seen, order = set(pool), set(), []
    footprints = tuple(tuple(f) for f in footprints)
    for position in range(max(map(len, footprints), default=0)):
        for footprint in footprints:
            if position < len(footprint):
                coordinate = footprint[position]
                if coordinate in allowed and coordinate not in seen:
                    seen.add(coordinate)
                    order.append(coordinate)
    return tuple(order+[i for i in pool if i not in seen])


def prioritized_subsets(pool, width, footprints=()):
    """Try fitting seed sets first, then mixed subsets in interleaved source order.

    Combinations stay lazy: a bounded search never constructs all subsets of a
    large space. Source subsets are canonicalized to sorted coordinate order;
    matrix entry order is supplied later by a completely verified layout.
    """
    allowed, seen = set(pool), set()
    for footprint in footprints:
        key = tuple(sorted(footprint))
        if len(key) == width and len(set(key)) == width and set(key) <= allowed and key not in seen:
            seen.add(key)
            yield key
    for subset in combinations(ordered_coordinates(pool, footprints), width):
        key = tuple(sorted(subset))
        if key not in seen:
            seen.add(key)
            yield key


def diagonal_pairs(left, right):
    """Fairly enumerate two lazy subset streams by increasing index sum.

    A lexicographic nested loop can spend its whole budget on one A subset.
    This schedule visits both axes. Cached prefixes grow only as the caller
    requests pairs; finite input streams terminate without a guessed bound.
    """
    streams, cache, ended = [iter(left), iter(right)], [[], []], [False, False]
    diagonal = 0
    while True:
        for leg in range(2):
            while len(cache[leg]) <= diagonal and not ended[leg]:
                item = next(streams[leg], None)
                if item is None:
                    ended[leg] = True
                else:
                    cache[leg].append(item)
        if any(ended[leg] and not cache[leg] for leg in range(2)):
            return
        if all(ended) and diagonal > len(cache[0])+len(cache[1])-2:
            return
        for i in range(diagonal+1):
            j = diagonal-i
            if i < len(cache[0]) and j < len(cache[1]):
                yield cache[0][i], cache[1][j]
        diagonal += 1


def support_proposals(source, shape, seeds=(), pools=None):
    """Yield separately countable A/B screens and full layout proposals.

    Support is expanded exactly before indexing by the A/B coordinate pair.
    This incorporates all cancellations in the source tensor. A/B screens are
    recorded even when too few eligible outputs remain. Output subsets consume
    separate attempt units, preventing an unbounded hidden loop inside a screen.
    """
    m, k, n = shape
    widths = (m*k, k*n, m*n)
    pools = validate_pools(source.dimensions, pools)
    if any(len(pool) < width for pool, width in zip(pools, widths)):
        return
    footprints = tuple(tuple(w.selections[leg] for w in seeds) for leg in range(3))
    index = defaultdict(list)
    output_pool = set(pools[2])
    for (a, b, c), coefficient in source.expand().items():
        if c in output_pool:
            index[a, b].append((c, coefficient))
    left = prioritized_subsets(pools[0], widths[0], footprints[0])
    right = prioritized_subsets(pools[1], widths[1], footprints[1])
    for a, b in diagonal_pairs(left, right):
        outputs = defaultdict(list)
        for x in a:
            for y in b:
                for z, value in index[x, y]:
                    outputs[z].append((x, y, value))
        eligible = tuple(sorted(z for z, terms in outputs.items() if len(terms) == k and all(v == 1 for _, _, v in terms)))
        yield {"kind": "input-pair screen", "left_coordinates": list(a), "right_coordinates": list(b),
               "eligible_outputs": list(eligible), "required_outputs": widths[2],
               "status": "output subsets available" if len(eligible) >= widths[2] else "insufficient degree-compatible outputs"}, None
        if len(eligible) < widths[2]:
            continue
        for c in prioritized_subsets(eligible, widths[2], footprints[2]):
            degree_a, degree_b = dict.fromkeys(a, 0), dict.fromkeys(b, 0)
            for z in c:
                for x, y, _ in outputs[z]:
                    degree_a[x] += 1
                    degree_b[y] += 1
            good = all(d == n for d in degree_a.values()) and all(d == m for d in degree_b.values())
            yield {"kind": "layout proposal", "selected_coordinates": [list(a), list(b), list(c)],
                   "left_degrees": list(degree_a.values()), "required_left_degree": n,
                   "right_degrees": list(degree_b.values()), "required_right_degree": m,
                   "status": "degree conditions passed" if good else "input degree mismatch"}, (a, b, c) if good else None


def search_support(source, shape=(2, 2, 2), *, seeds=(), pools=None, candidate_budget=4096, max_parts=2):
    """Run bounded exact screens/layout checks and retain completely proved maps.

    The budget counts both input-pair screens and full layout proposals; their
    separate totals are reported. Reaching the cap leaves exhaustion unknown:
    no extra unrecorded proposal is evaluated merely to peek at the stream.
    Saved-map limits never erase the number of other recognized candidates.
    """
    shape = tuple(shape)
    if len(shape) != 3 or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError('support target needs three positive integer matrix dimensions')
    if type(candidate_budget) is not int or candidate_budget < 0 or type(max_parts) is not int or max_parts < 1:
        raise ValueError('support budget must be nonnegative and saved-part cap positive')
    seeds = tuple(seeds)
    for seed in seeds:
        verify_restriction(source, seed)
    pools = validate_pools(source.dimensions, pools)
    proposals = iter(support_proposals(source, shape, seeds, pools))
    records, witnesses, hits, exhausted = [], [], 0, False
    support = source.expand()
    for _ in range(candidate_budget):
        proposal = next(proposals, None)
        if proposal is None:
            exhausted = True
            break
        record, selections = proposal
        if selections is not None:
            witness, status = recognize_restriction(source.dimensions, support, selections)
            record['status'] = status
            if witness is not None:
                verify_restriction(source, witness)
                if witness.shape != shape:
                    raise AssertionError('support recognition changed the requested matrix shape')
                record['complete_identity_verified'] = True
                record['witness'] = witness.document()
                hits += 1
                if len(witnesses) < max_parts:
                    record['saved_part_index'] = len(witnesses)
                    witnesses.append(witness)
        records.append(record)
    return tuple(witnesses), {"shape": list(shape), "candidate_budget": candidate_budget,
        "attempt_units": "each A/B screen and each output-layout proposal consumes one unit",
        "attempted_candidates": len(records), "input_pair_screens": sum(r['kind'] == 'input-pair screen' for r in records),
        "layout_proposals": sum(r['kind'] == 'layout proposal' for r in records),
        "recognized_candidates": hits, "retained_parts": len(witnesses), "max_parts": max_parts,
        "stream_exhausted": exhausted, "budget_reached": len(records) == candidate_budget,
        "exhaustion_unknown_at_budget": not exhausted, "candidates": records,
        "limitation": "Exact coordinate restrictions with coefficient-one layouts only; degree pruning is necessary but not sufficient. No general linear maps or global exclusion."}


def validate_pools(dimensions, pools):
    """Snapshot allowed coordinate subsets, including empty but never invalid pools."""
    pools = tuple(tuple(range(d)) for d in dimensions) if pools is None else tuple(tuple(p) for p in pools)
    if (len(pools) != 3 or any(len(set(p)) != len(p) or any(type(i) is not int or not 0 <= i < d for i in p)
                              for p, d in zip(pools, dimensions))):
        raise ValueError('support pools need three distinct in-range integer selections')
    return pools
