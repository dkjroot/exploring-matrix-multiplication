"""Project exact arithmetic categories without executing enormous matrices.

This module reads verified stage descriptions and their existing dimension
budgets. It accounts for combination additions/subtractions, coefficient scales,
normalizations and classical leaf work separately. All count arithmetic is
integer arithmetic. Logs or approximate SVG coordinates belong only to display.
Projection points are recurrence evaluations, not captured execution evidence.

No memory peak or operand width is extrapolated from small samples. Those depend
on representation, inputs, cancellations and the interpreter. The projection
retains a 64-bit INPUT/OUTPUT storage baseline, which is neither the storage of
exact Fraction objects nor an algorithm's peak-memory estimate.
"""

CATEGORIES = ('field_multiplications', 'coefficient_multiplications',
              'field_additions', 'field_subtractions', 'field_normalizations')


def stage_counts(stage, child_shape):
    """Derive one node's overhead from its explicit formula or sparse recipe.

    Sparse input/output coefficients each perform one scale and one add.
    Rectangular stages use three different areas. Negative coefficients in that
    schedule remain scales followed by additions, rather than fabricated calls
    to the subtraction primitive. Readable formulas use their actual add/sub
    schedules, retaining Winograd's shared intermediates.
    """
    result = dict.fromkeys(CATEGORIES, 0)
    m, k, n = child_shape
    scheme = getattr(stage, 'scheme', None)
    if scheme is None and hasattr(stage, 'left'):
        scheme = stage
    if scheme is not None:
        terms = sum(area*sum(len(column) for column in factor)
                    for area, factor in zip((m*k, k*n, m*n),
                                            (scheme.left, scheme.right, scheme.output)))
        result['field_additions'] = terms
        result['coefficient_multiplications'] = terms
        if scheme.divisor != 1:
            result['field_normalizations'] = m*n*scheme.shape[0]*scheme.shape[2]
    else:
        if stage.name not in ('strassen', 'strassen-winograd'):
            raise ValueError('arithmetic-category projection needs a declared formula schedule')
        adds, subs = (12, 6) if stage.name == 'strassen' else (7, 8)
        result['field_additions'], result['field_subtractions'] = adds*m*n, subs*m*n
    return result


def category_budget(name, size, cutoff, plan=None):
    """Calculate discrete category counts without running multiplication data.

    The budget's padding and leaf traversal match the existing execution plan.
    Projection and executed scalar counters remain independent: any disagreement
    is a failed budget check rather than a fit correcting the projection.
    """
    if any(type(i) is not int or i < 1 for i in (size, cutoff)):
        raise ValueError('size and cutoff must be positive integers')
    result = dict.fromkeys(CATEGORIES, 0)
    if name == 'naive' and plan is None:
        result.update(field_multiplications=size**3, field_additions=size**3)
        return result
    if plan is None:
        if name == 'smirnov':
            from .smirnov import plan as smirnov_plan
            plan = smirnov_plan()
        else:
            from .composition import Composition
            plan = Composition([name])
    budget = plan.budget(size, size, size, cutoff=cutoff)
    products = budget['field_multiplications']
    result.update(field_multiplications=products, field_additions=products)
    for level in budget['levels']:
        stage = plan.stages[level['depth'] % len(plan.stages)]
        if isinstance(level['size'], list):
            child = tuple(x//y for x, y in zip(level['size'], level['radix']))
        else:
            child = (level['size']//level['radix'],)*3
        overhead = stage_counts(stage, child)
        for category, count in overhead.items():
            result[category] += level['nodes']*count
    if sum(result.values()) != budget['field_arithmetic_operations']:
        raise AssertionError('category projection disagrees with the independent total-step budget')
    return result


def project_work(name, cutoff, model, plan=None, cycles=12):
    """Evaluate fixed-cutoff recurrences over whole cycles beyond measured sizes.

    This finite sequence illustrates an unbounded recurrence; it is not execution
    to infinity or proof from a finite graph. The exact symbolic recurrence and
    algorithm exponent in the model remain the asymptotic evidence. Unknown
    descriptors are reported unavailable rather than extrapolated from samples.
    """
    if type(cycles) is not int or cycles < 1:
        raise ValueError('projection cycles must be a positive integer')
    radix = model.get('base_dimension') or 2
    points = []
    for depth in range(cycles+1):
        size = radix**depth
        counts = category_budget(name, size, cutoff, plan)
        points.append({'size': size, 'cycle_depth': depth, 'counts': counts,
                       'field_arithmetic_operations': sum(counts.values()),
                       'input_output_64bit_baseline_bytes': 3*size*size*8})
    return {'status': 'projected', 'kind': 'exact integer recurrence evaluation; no matrix execution',
            'fixed_cutoff': cutoff, 'cycle_radix': radix, 'cycles': cycles,
            'recurrence': model['recurrence'], 'points': points,
            'operand_width_projection': 'not derived; measured widths depend on inputs and coefficients',
            'peak_memory_projection': 'not derived; 64-bit input/output baseline is not exact representation or peak memory',
            'setup_scaling': 'fixed recipe preparation is separate from per-call work; construction/search budgets are independent parameters'}
