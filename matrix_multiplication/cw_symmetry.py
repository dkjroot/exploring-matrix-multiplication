"""Test cyclic transport and whole-target recovery on balanced CW₂ layouts.

All pairs are tested in a shared restriction of CW₂ to the fourth power.
Keeping the union of all candidate coordinates preserves every coefficient any
pair can see, including terms spanning different coarse sectors.
"""

from collections import Counter
from itertools import combinations, product
from types import SimpleNamespace

from .cw_extraction import execution_evidence, restrict_polynomial
from .cw_interior import powered_selections
from .degenerations import cw_family
from .restriction_neighborhoods import check_joint
from .tensor_restrictions import (RestrictionWitness, compose_selections,
                                  recover_restriction, recover_tensor_restriction)
from .tensors import Tensor, classical, coppersmith_winograd, tensor_power


def rotate_legs(values):
    """Move the old Y, Z and X spaces into the new X, Y and Z positions."""
    return values[1], values[2], values[0]


def matrix_cycle(shape):
    """Give the new shape and new-to-old row-major indices after one leg cycle.

    The monomial X[i,j] Y[j,l] Z[i,l] becomes X'[j,l] Y'[l,i] Z'[j,i].
    The old output and left input must be transposed in their new roles.
    """
    m, k, n = shape
    return (k, n, m), (tuple(range(k*n)),
                       tuple(i*n+l for l in range(n) for i in range(m)),
                       tuple(i*k+j for j in range(k) for i in range(m)))


def rotate_matrix_map(shape, selections):
    """Relabel an ambient coordinate witness in the rotated matrix convention."""
    new_shape, order = matrix_cycle(shape)
    return new_shape, compose_selections(rotate_legs(selections), order)


def rotate_matrix_recipe(tensor, shape):
    """Relabel rank-one factors and check the complete rotated matrix identity."""
    new_shape, order = matrix_cycle(shape)
    inverse = tuple({old: new for new, old in enumerate(leg)} for leg in order)
    terms = tuple(tuple(tuple((dest[i], value) for i, value in factor)
                        for dest, factor in zip(inverse, rotate_legs(term))) for term in tensor.terms)
    rotated = Tensor(rotate_legs(tensor.dimensions), terms)
    if rotated.expand() != classical(new_shape).expand():
        raise ValueError("cyclic recipe fails the complete rotated matrix identity")
    return rotated


def ambient_coefficients(component, ambient):
    """Embed every squared-component coefficient in CW fourth-power coordinates."""
    return {tuple(ambient[leg][i] for leg, i in enumerate(indices)): value
            for indices, value in tensor_power(component.tensor, 2).expand().items()}


def common_source(pool, max_source_terms):
    """Stream literal CW terms through the coordinate pool before materializing.

    A fourth-power coordinate has four base-four digits. Iterating the four
    elementary monomials lazily keeps only surviving terms, including mixed
    sectors, under the same term cap as the squared component search.
    """
    base = coppersmith_winograd(2).expand()
    local = tuple({i: j for j, i in enumerate(leg)} for leg in pool)
    terms = []
    for words in product(base, repeat=4):
        indices = tuple(64*a+16*b+4*c+d for a, b, c, d in zip(*words))
        if all(i in index for i, index in zip(indices, local)):
            if len(terms) == max_source_terms:
                return None
            # Each literal CW monomial has coefficient one.
            terms.append(tuple(((index[i], 1),) for i, index in zip(indices, local)))
    return Tensor(tuple(map(len, pool)), terms)


def explore_symmetry(components, recognized, refined_parts, *, max_source_dimension, max_source_terms):
    """Check the first eight balanced maps, their rotations, and 92 pair slots.

    Test the original within-sector pairs and all original-to-next-sector
    pairs. A simultaneous leg rotation of each pair would reach the other
    sector combinations. Each actual acceptance is recovered as one target.
    """
    sector = (1, 1, 2)
    recognized = recognized[:8]
    if not recognized:
        return (), {"status": "no balanced maps in the (1,1,2) search prefix"}
    sectors = (sector, rotate_legs(sector), rotate_legs(rotate_legs(sector)))
    by_sector = {item.sector: item for item in components}
    if not all(s in by_sector for s in sectors):
        return (), {"status": "a cyclic source was skipped by materialization limits"}

    ambient = [powered_selections(by_sector[s].selections, 16, 2) for s in sectors]
    original = ambient_coefficients(by_sector[sector], ambient[0])
    expected = original
    for s, selected in zip(sectors[1:], ambient[1:]):
        expected = {rotate_legs(indices): value for indices, value in expected.items()}
        if ambient_coefficients(by_sector[s], selected) != expected:
            raise AssertionError("squared CW sectors fail complete cyclic coefficient equality")

    shape = (2, 2, 2)
    maps = [compose_selections(ambient[0], selected) for selected in recognized]
    count = len(maps)
    for rotation in (1, 2):
        maps.extend(rotate_matrix_map(shape, selected)[1]
                    for selected in maps[(rotation-1)*count:rotation*count])

    # Compress only after computing maps in the actual common source. Keeping
    # every candidate coordinate together is essential for checking cross terms.
    pool = tuple(tuple(sorted({i for selected in maps for i in selected[leg]})) for leg in range(3))
    if max(map(len, pool)) > max_source_dimension:
        return (), {"status": "common source exceeds the declared dimension budget"}
    source = common_source(pool, max_source_terms)
    if source is None:
        return (), {"status": "common source exceeds the declared term budget"}
    local = tuple({i: j for j, i in enumerate(leg)} for leg in pool)
    _, polynomial, _, leading = cw_family(2, 4)
    polynomial = restrict_polynomial(polynomial, pool)
    witnesses = tuple(RestrictionWitness(source.dimensions, shape,
                        tuple(tuple(index[i] for i in selected) for index, selected in zip(local, m)))
                      for m in maps)

    parts, map_records, paid, best = [], [], [], []
    refined_by_map = {tuple(tuple(leg) for leg in d["ambient_source_coordinates_in_matrix_order"]): (t, d["name"])
                      for t, d in refined_parts}
    for i, witness in enumerate(witnesses):
        rotation, origin = divmod(i, count)
        tensor, recovery = recover_restriction(source, witness, polynomial, leading)
        name = f"symmetry-map-{i}"
        description = {"name": name, "shape": list(shape), "sector": list(sectors[rotation]),
                       "parent_map_index": origin, "cyclic_rotations": rotation,
                       "products": tensor.rank, "classical_products": 8,
                       "ambient_source_coordinates_in_matrix_order": maps[i], "recovery": recovery,
                       "execution": execution_evidence(tensor, (witness,))}
        parts.append((tensor, description))
        paid.append(tensor.rank)
        best.append(tensor.rank)
        record = {"index": i, "part": name, "sector": list(sectors[rotation]),
                  "ambient_source_coordinates_in_matrix_order": maps[i]}
        if maps[origin] in refined_by_map:
            refined, parent = refined_by_map[maps[origin]]
            for _ in range(rotation):
                refined = rotate_matrix_recipe(refined, shape)
            if refined.expand() != tensor.expand():
                raise AssertionError("transported reduction disagrees with paid polynomial recovery")
            best[i] = min(tensor.rank, refined.rank)
            if rotation:
                refined_name = name+"-reduced"
                refined_description = {k: v for k, v in description.items() if k != "recovery"}
                parts.append((refined, {**refined_description, "name": refined_name, "products": refined.rank,
                              "parent_part": parent, "source_kind": "cyclic transport of exact rank reduction",
                              "execution": execution_evidence(refined, (witness,))}))
                record["refined_part"] = refined_name
            else:
                record["refined_part"] = parent
        map_records.append(record)

    records = []
    for pair_indices, scope in ((combinations(range(count), 2), "within original sector"),
                                (product(range(count), range(count, 2*count)), "between cyclic sectors")):
        for a, b in pair_indices:
            joint, record = check_joint(source, witnesses[a], witnesses[b])
            record.update(pair=[a, b], scope=scope,
                          union_coordinates=compose_selections(pool, record["union_coordinates"]))
            if joint is not None:
                target = joint.target()
                tensor, recovery = recover_tensor_restriction(source, joint.selections, target, polynomial, leading)
                name = f"symmetry-joint-{a}-{b}"
                description = {"name": name, "source_kind": joint.kind, "products": tensor.rank,
                               "component_shapes": [list(s) for s in joint.shapes],
                               "classical_products": target.rank, "input_parts": [f"symmetry-map-{i}" for i in (a, b)],
                               "ambient_source_coordinates_in_matrix_order": compose_selections(pool, joint.selections),
                               "recovery": recovery,
                               "execution": execution_evidence(tensor, tuple(SimpleNamespace(shape=s) for s in joint.shapes))}
                if joint.kind == "independent direct sum":
                    description.update(separate_recovery_products=paid[a]+paid[b], best_known_separate_products=best[a]+best[b])
                else:
                    description["shape"] = list(joint.shapes[0])
                parts.append((tensor, description))
                record.update({k: v for k, v in description.items() if k in (
                    "products", "component_shapes", "classical_products", "separate_recovery_products",
                    "best_known_separate_products")}, part=name)
            records.append(record)

    return tuple(parts), {"status": "completed", "origin_map_cap": 8, "pair_cap": 92,
        "sectors": sectors, "complete_sector_symmetry_verified": True,
        "common_source_coordinates": pool, "common_source_literal_products": source.rank,
        "maps": map_records, "pairs": records,
        "outcomes": {scope: dict(Counter(r["status"] for r in records if r["scope"] == scope))
                     for scope in ("within original sector", "between cyclic sectors")},
        "joint_recovery_savings": sum(d["products"] < d.get("separate_recovery_products", 0) for _, d in parts),
        "savings_against_best_known_separate": sum(d["products"] < d.get("best_known_separate_products", 0) for _, d in parts),
        "dense_union_savings": sum(d["source_kind"] == "dense matrix union" and d["products"] < d["classical_products"]
                                   for _, d in parts if "source_kind" in d)}
