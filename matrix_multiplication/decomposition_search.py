"""Independently written exact flip/reduction search for rational decompositions.

Flips change two rank-one terms without changing their tensor. Reductions use
an exact linear dependence between two-factor products to eliminate one term.
This searches finite recipes; it does not realize the OpenAI existence proof,
guarantee a minimum rank, or infer real identities from finite-field samples.
Every accepted result passes the existing complete matrix identity verifier.
Walks are seeded, bounded, and replayable. Construction statistics deliberately
stay separate from the arithmetic counts of executing a candidate multiplier.
"""

from fractions import Fraction
from itertools import combinations
from random import Random

from .tensors import Tensor, combine, kronecker, to_scheme, vector


def projective(factor):
    """Return a factor's leading scale and its normalized direction.

    For example, (2, -4) and (-1, 2) share the direction (1, -2). Their
    leading scales differ, but a flip can absorb those scales into another
    factor without changing the outer product. Tensor construction has already
    sorted coordinates and removed zero entries, so the first value is nonzero.
    """
    if not factor:
        return Fraction(0), ()
    scale = factor[0][1]
    return scale, tuple((i, c/scale) for i, c in factor)


def flip_choices(tensor):
    """List pairs with a proportional common factor in any of the three legs."""
    choices = []
    for leg in range(3):
        # "Leg" means one of the U, V, W vector spaces. A flip needs a shared
        # direction in one space; its other two factors may be entirely different.
        groups = {}
        for i, term in enumerate(tensor.terms):
            _, key = projective(term[leg])
            groups.setdefault(key, []).append(i)
        for group in groups.values():
            for i, j in combinations(group, 2):
                # The two shear directions are different neighbors, so keep
                # both ordered pairs rather than just changing the sign.
                choices.extend(((leg, i, j), (leg, j, i)))
    return choices


def flip(tensor, leg, i, j, amount):
    """Preserve u*v*w + u*v'*w' by shearing the two other factors.

    After absorbing proportional common-factor scales into the next leg, use
    v -> v+t*v' and w' -> w'-t*w. Cross terms cancel exactly. Cyclically
    selecting the common leg covers all three kinds of flips over Q.
    """
    if leg not in range(3) or not (0 <= i < tensor.rank and 0 <= j < tensor.rank) or i == j:
        raise ValueError("invalid flip coordinates")
    if type(amount) not in (int, Fraction) or not amount:
        raise ValueError("flip amount must be a nonzero exact rational")
    first, second = [list(tensor.terms[index]) for index in (i, j)]
    s1, common1 = projective(first[leg])
    s2, common2 = projective(second[leg])
    if common1 != common2:
        raise ValueError("flip needs proportional common factors")
    v, w = (leg+1) % 3, (leg+2) % 3
    # Each term's three scales multiply. Move the common leg's scale to v,
    # leaving exactly the same normalized u in both terms. The shear identity
    # is now u*(v1+t*v2)*w1 + u*v2*(w2-t*w1): the two t cross terms cancel.
    first[leg] = second[leg] = common1
    first[v] = vector((k, s1*c) for k, c in first[v])
    second[v] = vector((k, s2*c) for k, c in second[v])
    first[v] = combine(first[v], second[v], amount)
    second[w] = combine(second[w], first[w], -amount)
    terms = list(tensor.terms)
    terms[i], terms[j] = tuple(first), tuple(second)
    return Tensor(tensor.dimensions, tuple(terms))


def dependence(tensor, free_leg):
    """Find one dependence among pair-factor columns with sparse rational elimination.

    Store each pivot column together with its expression in original columns.
    A zero residual gives sum(alpha_t*pair_t)=0. This is exact Gaussian
    elimination written here, without NumPy, floating tolerances, or a solver.
    """
    if type(free_leg) is not int or free_leg not in range(3):
        raise ValueError("dependency needs one of three factor legs")
    a, b = (free_leg+1) % 3, (free_leg+2) % 3
    pivots = {}
    for index, term in enumerate(tensor.terms):
        residual = dict(kronecker(term[a], term[b], tensor.dimensions[b]))
        # Flatten the outer product of the two fixed factors into a sparse
        # column. Only these columns need to be dependent; the free factors can
        # be arbitrary vectors, which is why this detects more than duplicate terms.
        relation = {index: Fraction(1)}
        while residual:
            key = min(residual)
            scale = residual[key]
            if key not in pivots:
                # Normalize a newly independent column to a leading one and
                # remember which original columns produced it. Later elimination
                # applies the same subtraction to that expression as to residual.
                pivots[key] = ({i: c/scale for i, c in residual.items()},
                               {i: c/scale for i, c in relation.items()})
                break
            column, expression = pivots[key]
            for target, c in column.items():
                residual[target] = residual.get(target, 0)-scale*c
                if not residual[target]:
                    del residual[target]
            for target, c in expression.items():
                relation[target] = relation.get(target, 0)-scale*c
                if not relation[target]:
                    del relation[target]
        else:
            # residual=0; its own index has coefficient 1 before normalization.
            return index, tuple(sorted(relation.items()))
    return None


def reduce_relation(tensor, free_leg, removed, relation):
    """Validate a pair dependence, then absorb the removed term into remaining ones.

    If pair_p = -sum(alpha_t/alpha_p * pair_t), add that multiple of the
    removed free factor to free_t. The tensor stays identical and rank drops.
    Checking the relation here also makes replayed search logs untrusted data.
    """
    if free_leg not in range(3) or not 0 <= removed < tensor.rank:
        raise ValueError("invalid reduction coordinates")
    relation = vector(relation)
    if any(i >= tensor.rank for i, _ in relation):
        raise ValueError("reduction references an absent term")
    alpha = dict(relation).get(removed, 0)
    if not alpha:
        raise ValueError("removed term must have a nonzero dependence coefficient")
    a, b = (free_leg+1) % 3, (free_leg+2) % 3
    check = vector((j, c*v) for i, c in relation
                   for j, v in kronecker(tensor.terms[i][a], tensor.terms[i][b], tensor.dimensions[b]))
    if check:
        raise ValueError("reduction relation is not an exact dependence")
    terms = list(tensor.terms)
    source = terms[removed][free_leg]
    # Solve the zero relation for the removed pair, then distribute its free
    # factor among the remaining terms. No extra products are introduced:
    # changing a free factor still leaves each surviving term rank one.
    for index, value in relation:
        if index == removed:
            continue
        term = list(terms[index])
        term[free_leg] = combine(term[free_leg], source, -value/alpha)
        terms[index] = tuple(term)
    del terms[removed]
    return Tensor(tensor.dimensions, tuple(terms))


def reduce_all(tensor):
    """Apply available pair dependencies repeatedly; return tensor and replay events."""
    events = []
    while True:
        for leg in range(3):
            found = dependence(tensor, leg)
            if found is not None:
                removed, relation = found
                tensor = reduce_relation(tensor, leg, removed, relation)
                events.append({"kind": "reduction", "leg": leg, "removed": removed,
                               "relation": [[i, str(c)] for i, c in relation]})
                break
        else:
            return tensor, events


def split(tensor, term_index, leg, piece):
    """Split one factor f into piece + (f-piece), adding at most one product.

    Allowing temporary rank increases can leave components unreachable by
    monotone flips/reductions. This identity is exact, but does not promise
    that subsequent moves will recover the extra product and improve rank.
    """
    if leg not in range(3) or not 0 <= term_index < tensor.rank:
        raise ValueError("invalid split coordinates")
    piece = vector(piece)
    if any(i >= tensor.dimensions[leg] for i, _ in piece):
        raise ValueError("split piece outside the factor space")
    first = list(tensor.terms[term_index])
    second = list(first)
    second[leg] = combine(first[leg], piece, -1)
    first[leg] = piece
    terms = list(tensor.terms)
    terms[term_index] = tuple(first)
    terms.append(tuple(second))
    return Tensor(tensor.dimensions, tuple(terms))


def replay(start, events):
    """Replay explicit algebraic transformations; reject invalid dependencies/flips."""
    tensor = start
    for event in events:
        # Every log entry names an algebraic operation, not arbitrary code.
        # Indices refer to the CURRENT term order: reduction deletes a term and
        # shifts subsequent indices, so replay must use the recorded sequence.
        if event["kind"] == "flip":
            tensor = flip(tensor, event["leg"], event["i"], event["j"], Fraction(event["amount"]))
        elif event["kind"] == "reduction":
            tensor = reduce_relation(tensor, event["leg"], event["removed"],
                                     tuple((i, Fraction(c)) for i, c in event["relation"]))
        elif event["kind"] == "split":
            tensor = split(tensor, event["term"], event["leg"],
                           tuple((i, Fraction(c)) for i, c in event["piece"]))
        else:
            raise ValueError("unknown search event")
    return tensor


def height(tensor):
    """Bound coefficient numerator/denominator growth, not tensor arithmetic work."""
    return max((max(abs(c.numerator), c.denominator) for term in tensor.terms
                for factor in term for _, c in factor), default=0)


def search(start, shape, *, target_rank, steps=1000, restarts=10, seed=0, max_height=1,
           extra_rank=0, split_every=50):
    """Walk exact rational flips with monotone reductions and bounded restarts.

    Each restart begins from the verified input. Optional periodic splits let
    the walk rise up to start.rank+extra_rank; this relaxes monotone search.
    A height ceiling bounds the starting recipe and every accepted walk state,
    including coefficients formed by dependence reduction. Rejected flips
    consume an attempted step.
    There is no completeness guarantee: no result is evidence only about this
    bounded walk. The best replay path and all attempt counts are retained.
    """
    to_scheme(start, shape)
    if (type(target_rank) is not int or target_rank < 1 or type(steps) is not int or steps < 0
            or type(restarts) is not int or restarts < 1 or type(max_height) is not int or max_height < 1
            or type(extra_rank) is not int or extra_rank < 0 or type(split_every) is not int or split_every < 1):
        raise ValueError("search needs a positive rank/restarts/height and nonnegative steps")
    if height(start) > max_height:
        raise ValueError("starting recipe exceeds max_height; raise --max-height to admit its coefficients")
    rng = Random(seed)
    reduced_start, initial_events = reduce_all(start)
    # A reduction preserves the tensor but can add free factors, increasing
    # coefficient height. Keep the original recipe if that reduction leaves the
    # requested search region; otherwise it is a valid initial improvement.
    if height(reduced_start) <= max_height:
        best, events = reduced_start, initial_events
    else:
        best, events = start, []
    # "best" is the lowest-rank decomposition found; "current" is the present
    # walk position. Keeping them separate permits wandering through equal or
    # temporarily higher ranks without overwriting an earlier improvement.
    statistics = {"seed": seed, "attempted_flips": 0, "accepted_flips": 0,
                  "height_rejections": 0, "restarts_started": 0,
                  "attempted_splits": 0, "accepted_splits": 0}
    for _ in range(restarts):
        if best.rank <= target_rank:
            break
        statistics["restarts_started"] += 1
        current = reduced_start if height(reduced_start) <= max_height else start
        path = list(initial_events) if current is reduced_start else []
        for attempt in range(steps):
            if extra_rank and attempt % split_every == 0 and current.rank < start.rank+extra_rank:
                term_index, leg = rng.randrange(current.rank), rng.randrange(3)
                coordinate = rng.randrange(current.dimensions[leg])
                piece = ((coordinate, Fraction(rng.choice((-1, 1)))),)
                statistics["attempted_splits"] += 1
                candidate = split(current, term_index, leg, piece)
                if height(candidate) <= max_height:
                    current = candidate
                    path.append({"kind": "split", "term": term_index, "leg": leg,
                                 "piece": [[i, str(c)] for i, c in piece]})
                    statistics["accepted_splits"] += 1
            choices = flip_choices(current)
            if not choices:
                break
            leg, i, j = rng.choice(choices)
            amount = rng.choice((-1, 1))
            statistics["attempted_flips"] += 1
            candidate = flip(current, leg, i, j, amount)
            if height(candidate) > max_height:
                statistics["height_rejections"] += 1
                continue
            reduced, reductions = reduce_all(candidate)
            if height(reduced) > max_height:
                statistics["height_rejections"] += 1
                continue
            statistics["accepted_flips"] += 1
            current = reduced
            # Append the flip before its reductions because subsequent indices
            # describe the reduced ordering. Copy the path when preserving best;
            # future accepted moves must not alter the saved recipe's transcript.
            path.append({"kind": "flip", "leg": leg, "i": i, "j": j, "amount": str(amount)})
            path.extend(reductions)
            if current.rank < best.rank:
                best, events = current, list(path)
            if best.rank <= target_rank:
                break
    to_scheme(best, shape)
    if replay(start, events).terms != best.terms:
        raise AssertionError("search transcript did not reproduce the accepted recipe")
    return best, {**statistics, "initial_rank": start.rank, "best_rank": best.rank,
                  "target_rank": target_rank, "target_reached": best.rank <= target_rank,
                  "events": events}


def beats_exponent(rank, dimension, numerator, denominator):
    """Compare log_dimension(rank) < numerator/denominator with integer powers.

    This avoids floating rounding when deciding whether a square finite recipe
    beats a target such as 11/4 (2.75). It says nothing about correctness; the
    caller must verify the full matrix identity and recursive overhead too.
    """
    if any(type(x) is not int or x < 1 for x in (rank, dimension, numerator, denominator)) or dimension < 2:
        raise ValueError("exponent comparison needs positive integers and dimension >= 2")
    return rank**denominator < dimension**numerator
