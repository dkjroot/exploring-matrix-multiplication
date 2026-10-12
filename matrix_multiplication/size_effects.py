"""Explain discrete size effects without confusing a recipe with its size policy.

An equal-block formula fixes the divisions at a recursive node. Padding arbitrary
inputs to a schedule prefix and choosing a classical cutoff are separate choices
made by our reference interpreter. They preserve exact identities and fixed-cutoff
asymptotic exponents, but can change finite counts substantially. These choices
are portable and are not imposed by Python, a particular GPU or a home computer.

All evidence here is derived from recorded discrete budgets, not a fitted curve.
Observed field-product comparisons are separate. Operand-size word costs and
memory peaks can vary even when the scalar schedule is identical. Unknown callbacks
receive only declared-model evidence: matching totals do not prove their traversal.
"""

from fractions import Fraction
from math import prod


def describe_sample(name, size, cutoff, predicted, model, *, known_policy=False):
    """Describe one square request's planner choices and inherent recipe divisions.

    Vector-valued shapes are kept for rectangular schedules. A full-cycle radix
    must not be mistaken for the shape at every level or the minimum padding
    size. The recorded level order determines the child shapes explicitly.
    """
    if not predicted:
        return {'status': 'unavailable', 'reason': 'no discrete budget was produced'}
    padded = tuple(predicted['padded_dimensions']) if 'padded_dimensions' in predicted else (predicted['padded_dimension'],)*3
    leaf = tuple(predicted.get('composition_budget', {}).get('leaf_dimensions',
                 (predicted['effective_leaf_dimension'],)*3))
    budget = predicted.get('composition_budget')
    levels = []
    if budget:
        for level in budget['levels']:
            parent = tuple(level['size']) if isinstance(level['size'], list) else (level['size'],)*3
            radix = tuple(level['radix']) if isinstance(level['radix'], list) else (level['radix'],)*3
            levels.append({'stage': level['stage'], 'parent_dimensions': list(parent),
                           'block_counts': list(radix), 'child_dimensions': [p//r for p, r in zip(parent, radix)],
                           'child_products_per_node': level['rank']})
    elif model.get('base_dimension') is not None:
        radix = model['base_dimension']
        parent = padded
        for _ in range(predicted['recursion_levels']):
            child = tuple(p//radix for p in parent)
            levels.append({'stage': name, 'parent_dimensions': list(parent), 'block_counts': [radix]*3,
                           'child_dimensions': list(child), 'child_products_per_node': model['base_bilinear_rank']})
            parent = child
    ratio = Fraction(prod(padded), size**3)
    recursive = model.get('base_dimension') is not None
    return {'status': 'known_reference_policy' if known_policy else 'declared_model_only',
            'source': 'discrete recurrence budget; execution counters are checked separately',
            'requested_dimensions': [size]*3, 'padded_dimensions': list(padded),
            'padding_added': padded != (size,)*3,
            'extra_zero_input_cells': padded[0]*padded[1]+padded[1]*padded[2]-2*size**2,
            'padded_volume_ratio': [ratio.numerator, ratio.denominator],
            'volume_ratio_is_multiplication_overhead': False,
            'configured_cutoff': cutoff, 'leaf_dimensions': list(leaf),
            'recursion_levels': len(levels), 'levels': levels,
            'padding_precedes_cutoff': recursive,
            'requested_fits_cutoff': size <= cutoff,
            'padded_fits_cutoff': max(padded) <= cutoff,
            'padding_triggered_recursion': recursive and size <= cutoff and max(padded) > cutoff,
            'recipe_constraint': 'integer equal-block divisions prescribed by each stage' if recursive else 'direct row-by-column products; no recursive division',
            'padding_policy': 'zero-pad to the first covering prefix of the repeating stage schedule' if recursive else 'none',
            'padding_is_universal_mathematical_requirement': False,
            'cutoff_policy': 'stop when every block dimension is at most the cutoff' if recursive else 'ignored',
            'platform_constraint': 'none identified; these are portable reference size-handling choices',
            'alternatives': ['handle fringes separately', 'peel rows/columns', 'use uneven/generalized blocks',
                             'dispatch to another verified recipe'] if recursive else [],
            'alternatives_implemented': False,
            'padding_and_cropping_are_exact': True if known_policy else None,
            'output_verification_scope': 'sample.verification supplies the actual product check',
            'fixed_cutoff_family_exponent_preserved': True if known_policy else None,
            'ideal_curve_scope': 'smooth unpadded continuation of the recurrence, not an executable fractional matrix size'}


def describe_transition(previous, current):
    """Attribute adjacent sampled budgets without asserting unobserved boundaries.

    A coarse input grid can skip many integer sizes. Report transitions between
    those samples, not the exact location of an unsampled threshold. Matching
    padded dimensions and leaf/stage sequences establish a planner plateau;
    matching observed field counts separately corroborates it. No inference is
    made about elapsed time, bit costs or memory from these field-count plateaus.
    """
    before, after = previous['size_effects'], current['size_effects']
    if before.get('status') != 'known_reference_policy' or after.get('status') != 'known_reference_policy':
        return {'kind': 'unattributed', 'reason': 'reference traversal is not established for both samples'}
    same_geometry = (before['padded_dimensions'] == after['padded_dimensions']
                     and before['leaf_dimensions'] == after['leaf_dimensions']
                     and before['levels'] == after['levels'])
    if after['padding_policy'] == 'none':
        kind, reason = 'schoolbook_growth', 'No padding or recursive cutoff; field work follows the requested dimensions.'
    elif same_geometry:
        kind, reason = 'padding_plateau', 'Different requested sizes use the same padded geometry, stages and leaf dimensions under our size policy.'
    elif before['padded_dimensions'] != after['padded_dimensions']:
        kind, reason = 'padded_geometry_change', 'The selected padded geometry changes; any extra recursion executes the recipe on that chosen larger problem. Padding to this grid is an implementation choice.'
    else:
        kind, reason = 'leaf_or_stage_change', 'The recorded leaf or stage schedule changes; inspect the fixed-cutoff traversal rather than inferring a new exponent.'
    observed = None
    if previous['status'] == current['status'] == 'passed':
        observed = (previous['observation']['counts'].get('field_multiplications', 0)
                    == current['observation']['counts'].get('field_multiplications', 0))
    return {'kind': kind, 'reason': reason, 'from_requested_size': previous['size'],
            'to_requested_size': current['size'], 'observed_data_products_equal': observed,
            'scope': 'adjacent sampled sizes; not every intervening size or every resource metric'}
