"""Extract, recover and recombine small exact parts from interior CW sectors.

The coarse catalog supplies exact source tensors and original coordinate maps.
Each bounded search retains all candidate outcomes. A matrix witness is useful
only after recovery from the actual CW polynomial family pays its finite cost.
Powers of witnessed parts give explicit reusable constructions, separately from
fresh searches on powered sources. Independently found parts in different source
tensors may be tensored together; they are never assumed to be an independent
direct sum in their overlapping original CW spaces.
"""

from itertools import product
from math import prod
from html import escape

from .cw_coarsening import coarse_catalog
from .cw_extraction import execution_evidence, restrict_polynomial
from .degenerations import cw_family
from .recipe_parts import recombine_parts
from .tensor_restrictions import (RestrictionWitness, product_witness,
                                  recover_restriction, search_restrictions,
                                  verify_restriction)
from .tensors import classical, tensor_power


def powered_selections(selections, source_width, power):
    """Lift local component coordinates into the flattened ambient CW power.

    A tuple of source coordinates (a,b) is flattened as a*source_width+b.
    This matches tensor_product and the polynomial family's lexicographic order.
    Each leg may have a different number of retained component coordinates.
    """
    if type(source_width) is not int or source_width < 1 or type(power) is not int or power < 1:
        raise ValueError("powered coordinate maps need positive integer width and power")
    if len(selections) != 3:
        raise ValueError("powered coordinate maps need three legs")
    result = []
    for selected in selections:
        if (not selected or len(set(selected)) != len(selected)
                or any(type(i) is not int or not 0 <= i < source_width for i in selected)):
            raise ValueError("powered coordinate selections must be distinct and in range")
        coordinates = []
        for word in product(selected, repeat=power):
            coordinate = 0
            for digit in word:
                coordinate = coordinate*source_width+digit
            coordinates.append(coordinate)
        result.append(tuple(coordinates))
    return tuple(result)


def explore_interiors(q=2, power=2, *, candidate_budget=2048, max_parts=2,
                      max_source_dimension=64, max_source_terms=4096):
    """Run powers one and two of each unrecognized coarse sector under budgets.

    Resource ceilings are checked from integer dimensions and literal term
    counts before materializing powers or polynomial families. A skipped power
    is reported, never described as a failed extraction. The first retained
    base witness is also squared explicitly, so a fresh bounded search miss
    cannot erase a known constructive part. Returned parts carry execution and
    recovery evidence and are suitable for generic serialization/recombination.
    """
    if any(type(n) is not int or n < 1 for n in
           (q, power, max_parts, max_source_dimension, max_source_terms)):
        raise ValueError("interior parameters and resource limits must be positive integers")
    if type(candidate_budget) is not int or candidate_budget < 0:
        raise ValueError("interior candidate budget must be nonnegative")
    # Bound catalog construction itself, not just powers of its components.
    if (q+2)**power > max_source_dimension or (3*q+3)**power > max_source_terms:
        raise ValueError("coarse catalog exceeds the declared source materialization budget")
    catalog, _ = coarse_catalog(q, power)
    interiors = tuple(item for item in catalog if item.layout is None)
    reports, parts, base_parts = [], [], []
    for item in interiors:
        base_witnesses = ()
        for component_power in (1, 2):
            dimensions = tuple(d**component_power for d in item.tensor.dimensions)
            total_power = power*component_power
            entry = {"sector": list(item.sector), "component_power": component_power,
                     "source_dimensions": list(dimensions)}
            if (max(dimensions) > max_source_dimension or item.tensor.rank**component_power > max_source_terms
                    or (q+2)**total_power > max_source_terms):
                entry.update(status="skipped by materialization budget")
                reports.append(entry)
                continue
            source = tensor_power(item.tensor, component_power)
            witnesses, search = search_restrictions(source, candidate_budget=candidate_budget, max_parts=max_parts)
            entry.update(status="completed", search=search, parts=[])
            if component_power == 1:
                base_witnesses = witnesses
            elif base_witnesses:
                # A known restriction remains available even if the prefix of
                # the fresh powered search contains no recognizable products.
                lifted = product_witness(base_witnesses[0], base_witnesses[0])
                verify_restriction(source, lifted)
                if lifted not in witnesses:
                    witnesses = (*witnesses, lifted)
                    entry["constructively_lifted_base_witnesses"] = 1
            ambient_maps = powered_selections(item.selections, (q+2)**power, component_power)
            _, full_polynomial, _, leading = cw_family(q, total_power)
            local_polynomial = restrict_polynomial(full_polynomial, ambient_maps)
            for index, witness in enumerate(witnesses):
                tensor, recovery = recover_restriction(source, witness, local_polynomial, leading)
                name = "sector-"+"-".join(map(str, item.sector))+f"-power-{component_power}-part-{index}"
                description = {"name": name, "q": q, "source_cw_power": power,
                    "sector": list(item.sector), "component_power": component_power,
                    "witness": witness.document(), "recovery": recovery,
                    "ambient_source_coordinates_in_matrix_order":
                        [[ambient_maps[leg][i] for i in chosen] for leg, chosen in enumerate(witness.selections)],
                    "execution": execution_evidence(tensor, (witness,)),
                    "source_kind": "CW polynomial recovery", "shape": list(witness.shape),
                    "products": tensor.rank, "classical_products": prod(witness.shape)}
                parts.append((tensor, description))
                entry["parts"].append(name)
                if component_power == 1 and index == 0:
                    base_parts.append((tensor, witness.shape, name))
            reports.append(entry)
    recombinations = []
    if len(base_parts) >= 2:
        combined, shape, first_name = base_parts[0]
        names = [first_name]
        for tensor, next_shape, name in base_parts[1:]:
            combined, shape = recombine_parts(combined, shape, tensor, next_shape)
            names.append(name)
        description = {"name": "recombined-interior-parts", "shape": list(shape),
            "source_kind": "tensor product of recovered parts in separate source copies",
            "input_parts": names, "products": combined.rank, "classical_products": prod(shape),
            "complete_identity_verified": combined.expand() == classical(shape).expand()}
        layout = RestrictionWitness(combined.dimensions, shape, tuple(tuple(range(d)) for d in combined.dimensions))
        description["execution"] = execution_evidence(combined, (layout,))
        parts.append((combined, description))
        recombinations.append(description)
    report = {"schema": "cw-interior-experiment-v1", "q": q, "cw_power": power,
        "settings": {"candidate_budget_per_sector_power": candidate_budget, "max_saved_search_parts": max_parts,
                     "component_powers": [1, 2], "max_source_dimension": max_source_dimension,
                     "max_source_terms": max_source_terms},
        "unrecognized_sectors": [list(item.sector) for item in interiors], "experiments": reports,
        "parts": [description for _, description in parts], "recombinations": recombinations,
        "source": "Local exact coarse_catalog, tensor restrictions and CW polynomial recovery; see CONSTRUCTIONS.md for mathematical references.",
        "limitation": "Bounded coordinate/permutation search, with explicit powers of known restrictions. Parts overlap within their source and are not an independent direct sum. No lower-exponent result is implied."}
    return tuple(parts), report


def interior_summary(report):
    """Explain paid costs, reusable maps and bounded misses without code knowledge."""
    rows = ''.join('<tr><td>'+escape(part['name'])+'</td><td>'
                   +escape(' × '.join(map(str, part['shape'])))+'</td><td>'
                   +str(part['products'])+'</td><td>'+str(part['classical_products'])+'</td></tr>'
                   for part in report['parts'])
    attempts = sum(e.get('search', {}).get('attempted_candidates', 0) for e in report['experiments'])
    hits = sum(e.get('search', {}).get('recognized_candidates', 0) for e in report['experiments'])
    skipped = sum(e['status'].startswith('skipped') for e in report['experiments'])
    return ('<h2>Reusable parts from interior CW components</h2>'
            '<p>A tensor describes which input combinations contribute to which outputs. Some merged '
            'CW components cannot themselves be whole matrix products. A coordinate restriction keeps '
            'selected variables and sets the others to zero. We search for smaller matrix products '
            'inside these components, checking every coefficient of each accepted identity. A shape '
            '(m,k,n) means an m×k input multiplied by a k×n input; an axis of one is allowed for a part.</p>'
            '<p>A witness records the retained coordinates in matrix order. It can be replayed without '
            'repeating the search, or combined with another witness by pairing coordinates. Squaring '
            'a known restriction is a constructive result, distinct from finding an additional '
            'restriction in a fresh search on the squared source. Parts can overlap in their original '
            'source: we do not add them as independent products. The recombination here uses separate '
            'source copies and an explicitly checked tensor product.</p>'
            f'<p>The bounded searches tested {attempts} candidate subsets and recognized {hits}. '
            f'{skipped} source powers were skipped by materialization limits. The saved JSON retains '
            'every attempted subset and its outcome. The search considers only coordinate subsets '
            'and coefficient-one basis permutations. A miss does not exclude other subsets beyond '
            'the budget, general linear maps, or more effective mathematical constructions.</p>'
            '<table><thead><tr><th>Reusable part</th><th>Matrix shape</th>'
            '<th>Recovered data products</th><th>Classical data products</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            '<p>Recovered counts pay for finite polynomial coefficient recovery, including interpolation '
            'or exact expansion with replayed dependency reductions. We retain the less costly verified '
            'route, then execute signed rational inputs and record additions and coefficient arithmetic '
            'separately. Construction and verification are setup work; their raw coefficient-operation '
            'counts remain unmeasured. Counts equal to the classical baseline establish reusable exact '
            'parts, not an arithmetic saving or a lower exponent. parts.json contains each executable '
            'rational tensor with its maps and provenance; a supported recombination also exports '
            'scheme.json for recursive execution and independent certification.</p>')
