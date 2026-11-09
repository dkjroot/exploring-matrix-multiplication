"""Three exact support experiments followed by cyclic transport and joint recovery.

Step one searches only the coordinate pool of existing seed witnesses. Step two
broadens to the whole squared component and neighboring balanced target shapes.
Step three recovers retained identities from the actual polynomial, runs bounded
exact rank-reduction walks, replays every retained path, and tensors a successful
refined part with itself. The CW₂ follow-up transports balanced layouts and
recovers checked pairs in a common source. Every unsuccessful search remains
explicit evidence.
"""

from html import escape
from itertools import product
from math import prod

from .cw_coarsening import coarse_catalog
from .cw_extraction import execution_evidence, restrict_polynomial
from .cw_interior import powered_selections
from .cw_symmetry import explore_symmetry
from .decomposition_search import height, replay, search
from .degenerations import cw_family
from .recipe_parts import identity_witness, recombine_parts
from .support_search import search_support
from .tensor_restrictions import product_witness, recover_restriction, search_restrictions
from .tensors import classical, tensor_power


def explore_support(q=2, power=2, *, candidate_budget=4096, seed_budget=2048,
                    max_parts=2, steps=1000, restarts=10, search_seeds=(0, 1),
                    max_height=1, reduction_limit=1,
                    max_source_dimension=64, max_source_terms=4096):
    """Execute the support and cyclic experiments in order under declared caps.

    Each support target/scope has its own attempt budget. Seed-prefix setup has
    a separate cap, as do rank walks per seed and restart. At most max_parts maps
    per target survive; at most reduction_limit nonunit parts enter rank walks.
    Retained recipes pay exact polynomial recovery before any reduction. A miss
    in the pooled scope does not suppress the full-space search or erase a hit.
    """
    positive = (q, power, max_parts, restarts, max_height, reduction_limit,
                max_source_dimension, max_source_terms)
    if any(type(n) is not int or n < 1 for n in positive):
        raise ValueError('support parameters and resource caps must be positive integers')
    if any(type(n) is not int or n < 0 for n in (candidate_budget, seed_budget, steps)):
        raise ValueError('support, seed and rank-walk budgets must be nonnegative integers')
    search_seeds = tuple(search_seeds)
    if not search_seeds or any(type(s) is not int for s in search_seeds):
        raise ValueError('choose at least one integer rank-walk seed')
    if (q+2)**power > max_source_dimension or (3*q+3)**power > max_source_terms:
        raise ValueError('coarse catalog exceeds the declared materialization budget')
    catalog, _ = coarse_catalog(q, power)
    contexts, seed_records = [], []
    for component in catalog:
        if component.layout is not None:
            continue
        dimensions = tuple(d*d for d in component.tensor.dimensions)
        if (max(dimensions) > max_source_dimension or component.tensor.rank**2 > max_source_terms
                or (q+2)**(2*power) > max_source_terms):
            seed_records.append({'sector': list(component.sector), 'status': 'skipped by materialization budget'})
            continue
        base, evidence = search_restrictions(component.tensor, candidate_budget=seed_budget, max_parts=2)
        seeds = tuple(product_witness(a, b) for a, b in product(base, repeat=2))
        pools = tuple(tuple(sorted({i for w in seeds for i in w.selections[leg]})) for leg in range(3))
        contexts.append({'component': component, 'source': tensor_power(component.tensor, 2),
                         'seeds': seeds, 'pools': pools})
        seed_records.append({'sector': list(component.sector), 'status': 'prepared',
                             'prefix': evidence, 'squared_seed_maps': [w.document() for w in seeds]})

    step_one, step_two, candidates = [], [], []
    # Complete the pooled experiment for every orientation before broadening.
    for context in contexts:
        witnesses, evidence = search_support(context['source'], (2, 2, 2), seeds=context['seeds'],
            pools=context['pools'], candidate_budget=candidate_budget, max_parts=max_parts)
        step_one.append({'sector': list(context['component'].sector), 'pool': [list(p) for p in context['pools']],
                         'search': evidence, 'witnesses': [w.document() for w in witnesses]})
        context['pooled_hits'] = witnesses
        for witness in witnesses:
            candidates.append((context, witness, 'step-one pooled support'))
    targets = ((2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2))
    for context in contexts:
        learned = list(context['seeds'])+list(context['pooled_hits'])
        for target in targets:
            witnesses, evidence = search_support(context['source'], target, seeds=learned,
                candidate_budget=candidate_budget, max_parts=max_parts)
            step_two.append({'sector': list(context['component'].sector), 'shape': list(target),
                             'search': evidence, 'witnesses': [w.document() for w in witnesses]})
            for witness in witnesses:
                if witness not in learned:
                    learned.append(witness)
                if not any(c is context and w == witness for c, w, _ in candidates):
                    candidates.append((context, witness, 'step-two full support'))

    parts, recoveries, campaigns, refined = [], [], [], []
    for index, (context, witness, source_kind) in enumerate(candidates):
        component = context['component']
        if 'polynomial' not in context:
            _, polynomial, _, leading = cw_family(q, 2*power)
            ambient = powered_selections(component.selections, (q+2)**power, 2)
            context.update(polynomial=restrict_polynomial(polynomial, ambient), leading=leading, ambient=ambient)
        tensor, recovery = recover_restriction(context['source'], witness, context['polynomial'], context['leading'])
        name = f'support-part-{index}'
        description = {'name': name, 'q': q, 'source_cw_power': power,
            'sector': list(component.sector), 'component_power': 2, 'source_kind': source_kind,
            'shape': list(witness.shape), 'products': tensor.rank, 'classical_products': prod(witness.shape),
            'coordinates_outside_seed_pool': [[i for i in selected if i not in context['pools'][leg]]
                                              for leg, selected in enumerate(witness.selections)],
            'witness': witness.document(), 'ambient_source_coordinates_in_matrix_order':
                [[context['ambient'][leg][i] for i in selected] for leg, selected in enumerate(witness.selections)],
            'recovery': recovery, 'execution': execution_evidence(tensor, (witness,))}
        parts.append((tensor, description))
        recoveries.append({'part': name, 'shape': list(witness.shape), 'recovery': recovery})
        if min(witness.shape) < 2 or len(campaigns) >= reduction_limit:
            continue
        campaign = {'part': name, 'initial_products': tensor.rank, 'target_rank': tensor.rank-1,
                    'shape': list(witness.shape), 'runs': []}
        campaigns.append(campaign)
        if height(tensor) > max_height:
            campaign['status'] = 'skipped: starting coefficients exceed the declared height cap'
            continue
        best = tensor
        for seed in search_seeds:
            result, evidence = search(tensor, witness.shape, target_rank=tensor.rank-1,
                steps=steps, restarts=restarts, seed=seed, max_height=max_height)
            if replay(tensor, evidence['events']) != result or result.expand() != classical(witness.shape).expand():
                raise AssertionError('support rank walk failed exact replay or complete identity')
            campaign['runs'].append(evidence)
            if result.rank < best.rank:
                best = result
                campaign['best_run_index'] = len(campaign['runs'])-1
            if evidence['target_reached']:
                break
        campaign.update(status='completed', best_products=best.rank, improved=best.rank < tensor.rank)
        if best.rank < tensor.rank:
            refined_name = name+'-reduced'
            refined_description = {'name': refined_name, 'q': q, 'source_cw_power': power,
                'shape': list(witness.shape), 'products': best.rank, 'classical_products': prod(witness.shape),
                'source_kind': 'replayed exact rank reduction of recovered support part',
                'parent_part': name, 'campaign_index': len(campaigns)-1,
                'ambient_source_coordinates_in_matrix_order': description['ambient_source_coordinates_in_matrix_order'],
                'complete_identity_verified': True, 'execution': execution_evidence(best, (witness,))}
            parts.append((best, refined_description))
            refined.append((best, witness.shape, refined_name))
    recombinations = []
    if refined:
        tensor, shape, name = refined[0]
        combined, shape = recombine_parts(tensor, shape, tensor, shape)
        description = {'name': 'support-recombined-square', 'shape': list(shape), 'products': combined.rank,
            'classical_products': prod(shape), 'source_kind': 'tensor square of refined support part in separate source copies',
            'input_parts': [name, name], 'complete_identity_verified': True,
            'execution': execution_evidence(combined, (identity_witness(combined, shape),))}
        parts.append((combined, description))
        recombinations.append(description)
    symmetry_parts = ()
    symmetry = {'status': 'follow-up scoped to CW₂ squared'}
    if (q, power) == (2, 2):
        recognized = [record['witness']['source_coordinates_in_matrix_order']
                      for entry in step_two if (entry['sector'], entry['shape']) == ([1, 1, 2], [2, 2, 2])
                      for record in entry['search']['candidates'] if 'witness' in record]
        refined_for_transport = [(t, d) for t, d in parts if 'parent_part' in d]
        symmetry_parts, symmetry = explore_symmetry(
            [c['component'] for c in contexts], recognized, refined_for_transport,
            max_source_dimension=max_source_dimension, max_source_terms=max_source_terms)
    report = {'schema': 'cw-three-step-support-v1', 'q': q, 'cw_power': power,
        'settings': {'candidate_budget_per_target_scope': candidate_budget, 'seed_prefix_budget': seed_budget,
            'step_one_target': [2, 2, 2], 'step_two_targets': [list(s) for s in targets],
            'max_parts_per_target': max_parts, 'steps_per_restart': steps, 'restarts_per_seed': restarts,
            'search_seeds': list(search_seeds), 'max_height': max_height, 'reduction_limit': reduction_limit,
            'max_source_dimension': max_source_dimension, 'max_source_terms': max_source_terms},
        'seed_setup': seed_records, 'step_one': step_one, 'step_two': step_two,
        'step_three': {'recoveries': recoveries, 'campaigns': campaigns, 'recombinations': recombinations},
        'step_four': symmetry,
        'parts': [d for _, d in parts],
        'limitation': 'Bounded coordinate support searches and exact rational walks only. New relative to earlier controls is not mathematical novelty. No stronger bound than the verified Smirnov baseline is claimed.'}
    return tuple(parts)+symmetry_parts, report


def compact_support_report(report):
    """Keep readable scope/cost evidence in HTML while full attempts remain JSON."""
    result = {**report, 'seed_setup': [], 'step_one': [], 'step_two': []}
    for item in report['seed_setup']:
        result['seed_setup'].append({**item, **({'prefix': {k: v for k, v in item['prefix'].items() if k != 'candidates'}}
                                               if 'prefix' in item else {})})
    for step in ('step_one', 'step_two'):
        result[step] = [{**item, 'search': {k: v for k, v in item['search'].items() if k != 'candidates'}}
                        for item in report[step]]
    result['step_four'] = {k: v for k, v in report['step_four'].items() if k not in ('maps', 'pairs')}
    return result


def support_summary(report):
    """Explain the three sequential results without equating a known recipe to a record."""
    rows = ''
    for step in ('step_one', 'step_two'):
        for entry in report[step]:
            data = entry['search']
            rows += (f'<tr><td>{"Pooled" if step == "step_one" else "Full source"}</td>'
                f'<td>{escape(str(entry["sector"]))}</td><td>{escape(str(data["shape"]))}</td>'
                f'<td>{data["input_pair_screens"]}</td><td>{data["layout_proposals"]}</td>'
                f'<td>{data["recognized_candidates"]}</td></tr>')
    costs = ''.join(f'<tr><td>{escape(p["name"])}</td><td>{escape(str(p["shape"]))}</td>'
                   f'<td>{p["products"]}</td><td>{p["classical_products"]}</td></tr>' for p in report['parts'])
    return ('<h2>Three sequential support experiments</h2>'
        '<p>First we search only the coordinate pool of earlier verified parts for a 2×2 matrix product. '
        'Second we allow the entire squared interior component and nearby balanced matrix shapes, '
        'reusing any new maps to guide later targets. Third we recover the finite polynomial recipe, '
        'try exact rational rank reductions, replay their algebra, and recombine an improved part.</p>'
        '<p>For shape (m,k,n), keep mk input coordinates from A, kn from B and mn outputs. Every output '
        'needs exactly k supported products; each A entry must occur n times and each B entry m times. '
        'These necessary conditions remove unsuitable proposals. Every admitted layout still passes '
        'the complete coefficient identity and a separate check against the original source tensor. '
        'The input-subset streams are interleaved so that a budget does not stay on one A subset.</p>'
        '<table><thead><tr><th>Scope</th><th>Sector</th><th>Target shape</th><th>A/B screens</th>'
        f'<th>Layouts proposed</th><th>Verified layouts</th></tr></thead><tbody>{rows}</tbody></table>'
        '<p>A screen and an output-layout proposal each consume one attempt unit. Setup expansion '
        'and source-coefficient arithmetic are separate and not counted as matrix execution. A cap '
        'leaves stream exhaustion unknown; a failed bounded prefix is not an exclusion.</p>'
        '<h2>Paid recovery, reduction and recombination</h2><table><thead><tr><th>Part</th><th>Shape</th>'
        f'<th>Executable products</th><th>Classical target</th></tr></thead><tbody>{costs}</tbody></table>'
        '<p>Interpolation is compared with exact coefficient expansion and replayed dependencies. '
        'The later rank walk starts from that paid recipe and preserves every coefficient after each '
        'recorded algebraic change. Failed seeds remain in the evidence. A refined part is combined '
        'using separate source copies and explicit matrix index maps, then executed on rational inputs. '
        'A recovered 2×2 seven-product identity reproduces the known rank-seven case; its exponent '
        'is not a new bound and does not improve our Smirnov baseline. The reusable extraction maps, '
        'exact recipe transformations and failed larger-shape targets remain useful research evidence.</p>'
        + symmetry_summary(report['step_four']))


def symmetry_summary(report):
    """Explain which pairs actually ran and what their recovery costs compare."""
    if report['status'] != 'completed':
        return f'<h2>Cyclic follow-up</h2><p>{escape(report["status"])}</p>'
    rows = ''.join(f'<tr><td>{escape(scope)}</td><td>{escape(status)}</td><td>{count}</td></tr>'
                   for scope, outcomes in report['outcomes'].items() for status, count in outcomes.items())
    return ('<h2>Cyclic transport and joint recovery</h2>'
        '<p>Matrix multiplication uses three coordinate spaces, for the left input, right input and output. '
        'Cycling them transposes the old output and old left input in their new roles. '
        'We check that rotating every coefficient of a squared CW sector gives the corresponding sector. '
        f'All {len(report["maps"])} maps are also checked and recovered from the polynomial in the shared CW source.</p>'
        '<p>We test all pairs inside the original sector and all pairs from that sector to the next one. '
        'The shared source retains any cross coefficients, so disjoint coordinate sets alone '
        'cannot establish independence. Every accepted pair is recovered as one target.</p>'
        f'<table><thead><tr><th>Pair scope</th><th>Outcome</th><th>Pairs</th></tr></thead><tbody>{rows}</tbody></table>'
        f'<p>Joint recovery used fewer products than separate polynomial recovery in {report["joint_recovery_savings"]} '
        f'cases, and fewer than the best known separate recipes in {report["savings_against_best_known_separate"]}. '
        f'A dense union has its own classical comparator, with {report["dense_union_savings"]} savings observed. '
        'Full maps, rejection reasons, recovery costs '
        'and executed recipes are available in the JSON files.</p>')
