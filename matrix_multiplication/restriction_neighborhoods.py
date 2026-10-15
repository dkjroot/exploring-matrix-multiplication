"""Seeded coordinate neighborhoods and fully checked unions or independent sums.

Coordinates are zero-based source-space indices. A neighborhood changes exactly
one retained coordinate in one leg, preserving subset sizes. Accepted neighbors
can seed another step until the explicit attempt or witness-pool budget stops
the breadth-first walk. Pair unions may change the shape. Shared coordinates
never imply independent products; disjoint spaces still require checking every
cross coefficient. These searches do not consider general linear basis changes.
"""

from dataclasses import dataclass
from itertools import combinations

from .tensor_restrictions import (recognize_restriction, verify_restriction,
                                  recover_restriction, recover_tensor_restriction)
from .tensors import classical, direct_sum, restrict_coordinates


def subset_key(selections):
    """Identify retained coordinates independently of their matrix-order labels."""
    return tuple(tuple(sorted(s)) for s in selections)


def coordinate_neighbors(witness):
    """Yield all single-coordinate replacements in deterministic leg/position order."""
    selected = subset_key(witness.selections)
    for leg in range(3):
        kept = set(selected[leg])
        for removed in selected[leg]:
            for added in range(witness.source_dimensions[leg]):
                if added not in kept:
                    candidate = list(selected)
                    candidate[leg] = tuple(sorted((kept-{removed}) | {added}))
                    yield tuple(candidate), {"leg": leg, "removed": removed, "added": added}


def search_neighborhood(source, seeds, *, candidate_budget=2048, max_witnesses=64):
    """Walk verified seed neighborhoods, recording all unique attempted subsets.

    Only admitted maps enter the next frontier. max_witnesses bounds both saved
    witnesses and future expansion; accepted-but-unsaved maps are counted and
    explicitly marked. Frontier exhaustion is limited to that retained pool,
    not an exhaustive search of the entire coordinate graph.
    """
    if type(candidate_budget) is not int or candidate_budget < 0 or type(max_witnesses) is not int or max_witnesses < 1:
        raise ValueError("neighborhood budget must be nonnegative and witness cap positive")
    seeds = tuple(seeds)
    if len(seeds) > max_witnesses:
        raise ValueError("seed count exceeds the retained witness cap")
    witnesses, seen = [], set()
    for seed in seeds:
        verify_restriction(source, seed)
        key = subset_key(seed.selections)
        if key not in seen:
            witnesses.append(seed)
            seen.add(key)
    seed_count, records, hits, clipped = len(witnesses), [], 0, 0
    support = source.expand()
    parent = 0
    stopped = False
    while parent < len(witnesses) and not stopped:
        for selections, edit in coordinate_neighbors(witnesses[parent]):
            key = subset_key(selections)
            if key in seen:
                continue
            if len(records) == candidate_budget:
                stopped = True
                break
            seen.add(key)
            witness, status = recognize_restriction(source.dimensions, support, selections)
            record = {"parent_witness": parent, "edit": edit,
                      "selected_coordinates": [list(s) for s in selections], "status": status}
            if witness is not None:
                verify_restriction(source, witness)
                hits += 1
                record["complete_identity_verified"] = True
                if len(witnesses) < max_witnesses:
                    record["witness_index"] = len(witnesses)
                    witnesses.append(witness)
                else:
                    clipped += 1
                    record["not_retained_due_to_pool_cap"] = True
            records.append(record)
        parent += 1
    return tuple(witnesses), {"candidate_budget": candidate_budget, "attempted_candidates": len(records),
        "seed_count": seed_count, "recognized_new_candidates": hits, "retained_witnesses": len(witnesses),
        "max_witnesses": max_witnesses, "accepted_maps_not_retained": clipped,
        "budget_exhausted": stopped, "retained_frontier_exhausted": not stopped,
        "witnesses": [w.document() for w in witnesses], "candidates": records,
        "limitation": "Single-coordinate edits of retained accepted maps only; rejected maps do not expand, and pool limits can omit paths."}


@dataclass(frozen=True)
class JointRestriction:
    """A checked joint source map with one matrix shape or independent sum shapes."""

    selections: tuple
    shapes: tuple
    kind: str

    def __post_init__(self):
        """Snapshot target shapes and forbid repeated source coordinates or false kinds."""
        shapes = tuple(tuple(s) for s in self.shapes)
        selections = tuple(tuple(s) for s in self.selections)
        if (not shapes or len(selections) != 3 or any(len(s) != 3 for s in shapes)
                or any(type(d) is not int or d < 1 for shape in shapes for d in shape)):
            raise ValueError("joint restrictions need positive matrix shapes and three source selections")
        expected = tuple(sum(s[a]*s[b] for s in shapes) for a, b in ((0, 1), (1, 2), (0, 2)))
        if tuple(map(len, selections)) != expected:
            raise ValueError("joint source maps do not fit the component matrix shapes")
        if any(len(set(s)) != len(s) or any(type(i) is not int or i < 0 for i in s) for s in selections):
            raise ValueError("joint source selections must be distinct nonnegative integers")
        if ((self.kind == 'dense matrix union' and len(shapes) != 1)
                or (self.kind == 'independent direct sum' and len(shapes) < 2)
                or self.kind not in ('dense matrix union', 'independent direct sum')):
            raise ValueError("joint kind disagrees with its component shapes")
        object.__setattr__(self, 'shapes', shapes)
        object.__setattr__(self, 'selections', selections)

    def target(self):
        """Construct the independent classical specification, never the recipe."""
        expected = classical(self.shapes[0])
        for shape in self.shapes[1:]:
            expected = direct_sum(expected, classical(shape))
        return expected

    def document(self):
        """Retain source maps and component shapes for independent replay."""
        return {"kind": self.kind, "shapes": [list(s) for s in self.shapes],
                "source_coordinates_in_component_matrix_order": [list(s) for s in self.selections]}


def check_joint(source, left, right):
    """Check a pair as a dense union first, then as a genuinely independent sum.

    A dense union has its own matrix shape and classical baseline. Separate
    overlapping parts generally compute a different bilinear map, so their
    summed product counts are not a valid comparator for that dense target.
    A disjoint direct sum has the same target as separate execution only after
    the complete union identity excludes every unintended cross-product.
    """
    verify_restriction(source, left)
    verify_restriction(source, right)
    union = tuple(tuple(sorted(set(a) | set(b))) for a, b in zip(left.selections, right.selections))
    overlap = [leg for leg, (a, b) in enumerate(zip(left.selections, right.selections)) if set(a) & set(b)]
    witness, status = recognize_restriction(source.dimensions, source.expand(), union)
    evidence = {"shared_legs": overlap, "union_coordinates": [list(s) for s in union],
                "dense_recognition_status": status}
    if witness is not None:
        verify_restriction(source, witness)
        joint = JointRestriction(witness.selections, (witness.shape,), "dense matrix union")
        evidence.update(status="verified dense union", complete_identity_verified=True,
                        separate_cost_is_comparable=False)
        return joint, evidence
    if overlap:
        evidence["status"] = "shared spaces; no dense union recognized"
        return None, evidence
    selections = tuple(a+b for a, b in zip(left.selections, right.selections))
    joint = JointRestriction(selections, (left.shape, right.shape), "independent direct sum")
    actual, expected = restrict_coordinates(source, selections).expand(), joint.target().expand()
    if actual != expected:
        mismatch = next(key for key in sorted(actual.keys() | expected.keys()) if actual.get(key, 0) != expected.get(key, 0))
        value = actual.get(mismatch, 0)-expected.get(mismatch, 0)
        evidence.update(status="unwanted cross coefficients", first_difference={
            "local_coordinate": list(mismatch), "actual_minus_expected": [value.numerator, value.denominator]})
        return None, evidence
    evidence.update(status="verified independent direct sum", complete_identity_verified=True,
                    separate_cost_is_comparable=True)
    return joint, evidence


def search_joints(source, witnesses, *, candidate_budget=2048, max_results=4):
    """Test a bounded pair prefix, preserving failures and accepted joint maps."""
    if type(candidate_budget) is not int or candidate_budget < 0 or type(max_results) is not int or max_results < 1:
        raise ValueError("joint budget must be nonnegative and result cap positive")
    witnesses = tuple(witnesses)
    for witness in witnesses:
        verify_restriction(source, witness)
    pairs, results, records, seen = iter(combinations(range(len(witnesses)), 2)), [], [], set()
    hits, exhausted = 0, False
    for _ in range(candidate_budget):
        pair = next(pairs, None)
        if pair is None:
            exhausted = True
            break
        a, b = pair
        joint, evidence = check_joint(source, witnesses[a], witnesses[b])
        evidence["pair"] = [a, b]
        if joint is not None:
            hits += 1
            key = (joint.kind, joint.shapes, joint.selections)
            if key not in seen and len(results) < max_results:
                seen.add(key)
                evidence["result_index"] = len(results)
                results.append((joint, pair))
        records.append(evidence)
    if not exhausted:
        exhausted = next(pairs, None) is None
    return tuple(results), {"candidate_budget": candidate_budget, "attempted_candidates": len(records),
        "recognized_candidates": hits, "retained_results": len(results), "max_results": max_results,
        "pair_pool_exhausted": exhausted, "budget_exhausted": not exhausted, "candidates": records,
        "limitation": "Pairs of the retained witness pool only; overlap is not independence, and failed recognition is not a general exclusion."}


def recover_joint(source, joint, parents, polynomial_terms, leading):
    """Recover a checked joint map and compare only recipes for the same target.

    Replay the pair check to bind the claimed target to its stated parents.
    Direct sums compare fully paid joint and separate polynomial recovery.
    Dense unions use their own classical target and deliberately leave the
    separate comparator unavailable. All this is recipe construction, separate
    from the caller's measured execution of the chosen recovered terms.
    """
    if len(parents) != 2:
        raise ValueError("joint recovery requires the two stated parent witnesses")
    checked, _ = check_joint(source, *parents)
    if checked != joint:
        raise ValueError("joint target does not match its checked parent maps")
    tensor, recovery = recover_tensor_restriction(source, joint.selections,
        joint.target(), polynomial_terms, leading)
    classical_cost = sum(s[0]*s[1]*s[2] for s in joint.shapes)
    comparison = {"products": tensor.rank, "same_target_classical_products": classical_cost,
                  "strictly_better_than_classical": tensor.rank < classical_cost}
    if joint.kind == 'independent direct sum':
        separate = [recover_restriction(source, parent, polynomial_terms, leading) for parent in parents]
        count = sum(recipe.rank for recipe, _ in separate)
        comparison.update(separate_recovery_products=count, joint_recovery_saves_products=tensor.rank < count,
                          separate_recovery_methods=[e['selected_method'] for _, e in separate])
    else:
        comparison.update(separate_recovery_products=None,
            separate_cost_not_comparable="Overlapping separate products may compute a different target; use the dense target baseline.")
    return tensor, recovery, comparison
