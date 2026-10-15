"""Bounded seeded CW map searches with exact joint recovery and honest controls.

The original coordinate prefix is retained as a control. Verified restrictions
seed a single-coordinate neighborhood, and products of base witnesses seed the
squared source. Pair tests distinguish a dense coordinate union from independent
direct sums. Accepted targets recover the actual polynomial coefficient before
their costs are compared. A completed negative search is not an exclusion proof.
"""

from html import escape
from itertools import product
from math import isqrt, prod
from types import SimpleNamespace

from .cw_coarsening import coarse_catalog
from .cw_extraction import execution_evidence, restrict_polynomial
from .cw_interior import powered_selections
from .degenerations import cw_family
from .restriction_neighborhoods import search_joints, search_neighborhood, recover_joint
from .tensor_restrictions import product_witness, recover_restriction, search_restrictions
from .tensors import tensor_power


def explore_neighborhoods(q=2, power=2, *, seed_budget=2048, candidate_budget=4096,
                          max_parts=2, max_witnesses=64, max_source_dimension=64,
                          max_source_terms=4096):
    """Try seed neighborhoods and pairs under a split per-source attempt budget.

    Half the budget is allocated to single-coordinate edits and half to pair
    tests. Unused attempts are not silently transferred or called executed work.
    Seed-prefix work is a separately declared budget. A finite pool bounds both
    frontier expansion and pair enumeration, while max_parts bounds new recipes
    compiled per source and joint targets retained. Seed recipes are paid and
    retained separately so comparisons use executable, rather than border, cost.
    """
    if any(type(n) is not int or n < 1 for n in
           (q, power, max_parts, max_witnesses, max_source_dimension, max_source_terms)):
        raise ValueError("neighborhood parameters and resource caps must be positive integers")
    if any(type(n) is not int or n < 0 for n in (seed_budget, candidate_budget)):
        raise ValueError("seed and candidate budgets must be nonnegative integers")
    if (q+2)**power > max_source_dimension or (3*q+3)**power > max_source_terms:
        raise ValueError("coarse catalog exceeds the declared source materialization budget")
    catalog, _ = coarse_catalog(q, power)
    reports, parts = [], []
    for item in catalog:
        if item.layout is not None:
            continue
        # At most two original parts, with a square seed pool that fits its cap.
        base, base_search = search_restrictions(item.tensor, candidate_budget=seed_budget,
                                               max_parts=min(2, isqrt(max_witnesses)))
        for component_power in (1, 2):
            dimensions = tuple(d**component_power for d in item.tensor.dimensions)
            total_power = power*component_power
            entry = {"sector": list(item.sector), "component_power": component_power,
                     "source_dimensions": list(dimensions), "parts": []}
            if (max(dimensions) > max_source_dimension or item.tensor.rank**component_power > max_source_terms
                    or (q+2)**total_power > max_source_terms):
                entry["status"] = "skipped by materialization budget"
                reports.append(entry)
                continue
            source = tensor_power(item.tensor, component_power)
            if component_power == 1:
                seeds, prefix = base, base_search
            else:
                seeds = tuple(product_witness(a, b) for a, b in product(base, repeat=2))
                _, prefix = search_restrictions(source, candidate_budget=seed_budget, max_parts=1)
            baseline_shapes = {tuple(c['shape']) for c in prefix['candidates']
                               if c.get('complete_identity_verified')}
            baseline_shapes.update(w.shape for w in seeds)
            witnesses, neighbors = search_neighborhood(source, seeds,
                candidate_budget=candidate_budget//2, max_witnesses=max_witnesses)
            joints, joint_search = search_joints(source, witnesses,
                candidate_budget=candidate_budget-candidate_budget//2, max_results=max_parts)
            entry.update(status="completed", seed_prefix=prefix, neighborhood=neighbors,
                         joint_search=joint_search, baseline_shapes=[list(s) for s in sorted(baseline_shapes)],
                         comparisons=[])
            ambient = powered_selections(item.selections, (q+2)**power, component_power)
            _, polynomial, _, leading = cw_family(q, total_power)
            polynomial = restrict_polynomial(polynomial, ambient)
            cache = {}

            def recover_one(index):
                """Pay and cache one source-specific witness recipe, retaining provenance."""
                if index in cache:
                    return cache[index]
                witness = witnesses[index]
                tensor, recovery = recover_restriction(source, witness, polynomial, leading)
                name = "sector-"+'-'.join(map(str, item.sector))+f"-power-{component_power}-map-{index}"
                description = {"name": name, "q": q, "source_cw_power": power,
                    "sector": list(item.sector), "component_power": component_power,
                    "source_kind": "verified seed" if index < neighbors['seed_count'] else "verified coordinate neighbor",
                    "shape": list(witness.shape), "products": tensor.rank,
                    "classical_products": prod(witness.shape), "witness": witness.document(),
                    "ambient_source_coordinates_in_matrix_order":
                        [[ambient[leg][i] for i in chosen] for leg, chosen in enumerate(witness.selections)],
                    "recovery": recovery, "execution": execution_evidence(tensor, (witness,))}
                parts.append((tensor, description))
                entry['parts'].append(name)
                cache[index] = (tensor, description)
                return cache[index]

            for index in range(min(len(witnesses), neighbors['seed_count']+max_parts)):
                tensor, description = recover_one(index)
                entry['comparisons'].append({"part": description['name'], "shape": description['shape'],
                    "products": tensor.rank, "same_target_classical_products": description['classical_products'],
                    "strictly_better_than_classical": tensor.rank < description['classical_products'],
                    "new_shape_relative_to_control": tuple(description['shape']) not in baseline_shapes})
            for joint_index, (joint, pair) in enumerate(joints):
                tensor, recovery, comparison = recover_joint(source, joint,
                    tuple(witnesses[i] for i in pair), polynomial, leading)
                components = tuple(SimpleNamespace(shape=s) for s in joint.shapes)
                name = "sector-"+'-'.join(map(str, item.sector))+f"-power-{component_power}-joint-{joint_index}"
                classical_cost = sum(prod(s) for s in joint.shapes)
                description = {"name": name, "q": q, "source_cw_power": power,
                    "sector": list(item.sector), "component_power": component_power,
                    "source_kind": joint.kind, "component_shapes": [list(s) for s in joint.shapes],
                    "products": tensor.rank, "classical_products": classical_cost,
                    "joint_witness": joint.document(), "input_witness_indices": list(pair),
                    "recovery": recovery, "execution": execution_evidence(tensor, components)}
                comparison.update(part=name,
                    new_shape_relative_to_control=len(joint.shapes) == 1 and joint.shapes[0] not in baseline_shapes)
                if joint.kind == "independent direct sum":
                    for index in pair:
                        recover_one(index)
                else:
                    description['shape'] = list(joint.shapes[0])
                    comparison['shape'] = list(joint.shapes[0])
                parts.append((tensor, description))
                entry['parts'].append(name)
                entry['comparisons'].append(comparison)
            reports.append(entry)
    comparisons = [c for e in reports for c in e.get('comparisons', [])]
    report = {"schema": "cw-neighborhood-experiment-v1", "q": q, "cw_power": power,
        "settings": {"seed_prefix_budget": seed_budget, "candidate_budget_per_sector_power": candidate_budget,
            "single_coordinate_budget": candidate_budget//2,
            "joint_pair_budget": candidate_budget-candidate_budget//2, "max_parts": max_parts,
            "max_witnesses": max_witnesses, "max_source_dimension": max_source_dimension,
            "max_source_terms": max_source_terms},
        "experiments": reports, "parts": [d for _, d in parts],
        "strict_recovery_savings": sum(c.get('joint_recovery_saves_products', False) for c in comparisons),
        "strict_classical_savings": sum(c['strictly_better_than_classical'] for c in comparisons),
        "new_verified_shapes": len({tuple(c['shape']) for c in comparisons if c['new_shape_relative_to_control']}),
        "limitation": "Only single-coordinate moves from accepted retained seeds and pairs from that pool are searched. No general linear restrictions, minimum-rank claim or impossibility proof."}
    return tuple(parts), report


def neighborhood_summary(report):
    """Explain negative results, controls and the independent-recovery comparator."""
    rows = ''
    for entry in report['experiments']:
        if entry['status'] != 'completed':
            rows += '<tr><td>'+escape(str(entry['sector']))+'</td><td>'+str(entry['component_power'])+'</td><td colspan="4">Skipped by materialization limits</td></tr>'
            continue
        n, j = entry['neighborhood'], entry['joint_search']
        rows += (f'<tr><td>{escape(str(entry["sector"]))}</td><td>{entry["component_power"]}</td>'
                 f'<td>{n["attempted_candidates"]}</td><td>{n["recognized_new_candidates"]}</td>'
                 f'<td>{j["attempted_candidates"]}</td><td>{j["recognized_candidates"]}</td></tr>')
    return ('<h2>Seeded coordinate neighborhoods and checked joint restrictions</h2>'
        '<p>A verified coordinate map keeps selected variables in each tensor space. This experiment '
        'changes one retained variable at a time, checks the complete matrix identity, and expands '
        'only accepted neighbors. Products of known maps seed the squared source. The initial '
        'largest-shape coordinate search remains a separate control; its attempts are not counted '
        'as neighborhood attempts. Half the declared follow-up budget is for edits and half for '
        'pairs. Unused attempts are reported as unused.</p>'
        '<table><thead><tr><th>Sector</th><th>Component power</th><th>Edits tested</th>'
        '<th>New accepted edits</th><th>Pairs tested</th><th>Accepted joint targets</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p>A pair may form one larger dense matrix product or two independent products. Overlapping '
        'spaces are never assumed independent. Even disjoint spaces can leave unwanted cross terms: '
        'we compare every coefficient with the proposed target. For an accepted independent sum, '
        'we recover the joint polynomial restriction and compare its full cost with separately '
        'recovered parts. A dense union is compared with its own classical product; summing costs '
        'of overlapping maps could compare different computations and is not credited as a saving.</p>'
        f'<p>Verified new shapes: {report["new_verified_shapes"]}; joint-recovery savings: '
        f'{report["strict_recovery_savings"]}; savings against the classical target: '
        f'{report["strict_classical_savings"]}. A zero means this experiment did not find that '
        'improvement. Single-coordinate moves can leave witnesses isolated even when useful coordinated '
        'moves or different shapes exist. Frontier exhaustion covers only the retained move graph; '
        'pair exhaustion covers only its finite pool. Exhaustion is not an impossibility proof.</p>'
        '<p>The saved JSON retains attempted maps, edits, rejected pairs, seed controls and exact '
        'recovery evidence. The parts library retains executable rational recipes, maps and provenance. '
        'Data products, additions and coefficient arithmetic are observed during separate signed '
        'rational executions; raw construction coefficient-operation counts remain unmeasured.</p>')


def compact_neighborhood_report(report):
    """Omit thousands of raw attempt rows from HTML, preserving them in JSON."""
    return {**report, 'experiments': [{**entry, **{
        key: {k: v for k, v in entry[key].items() if k != 'candidates'}
        for key in ('seed_prefix', 'neighborhood', 'joint_search') if key in entry}}
        for entry in report['experiments']]}
