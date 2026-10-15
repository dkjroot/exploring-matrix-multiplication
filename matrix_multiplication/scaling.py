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
        if stage.name == 'strassen':
            # Three sums and two differences on each input leg; six sums and
            # two differences on output. Their areas differ on rectangular cores.
            result['field_additions'] = 3*m*k + 3*k*n + 6*m*n
            result['field_subtractions'] = 2*m*k + 2*k*n + 2*m*n
        else:
            # S has one sum/three differences, T four differences, U six
            # sums/one difference. Preserve the authored reuse schedule.
            result['field_additions'] = m*k + 6*m*n
            result['field_subtractions'] = 3*m*k + 4*k*n + m*n
    return result


def category_budget(name, size, cutoff, plan=None, *, size_policy="fringe"):
    """Calculate discrete category counts without running multiplication data.

    The budget's padding and leaf traversal match the existing execution plan.
    Projection and executed scalar counters remain independent: any disagreement
    is a failed budget check rather than a fit correcting the projection.
    """
    if any(type(i) is not int or i < 1 for i in (size, cutoff)):
        raise ValueError('size and cutoff must be positive integers')
    from .common import validate_size_policy
    validate_size_policy(size_policy)
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
    budget = plan.budget(size, size, size, cutoff=cutoff, size_policy=size_policy)
    if "category_counts" in budget:
        return budget["category_counts"]
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


def project_work(name, cutoff, model, plan=None, cycles=12, *, size_policy="fringe"):
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
        counts = category_budget(name, size, cutoff, plan, size_policy=size_policy)
        points.append({'size': size, 'cycle_depth': depth, 'counts': counts,
                       'field_arithmetic_operations': sum(counts.values()),
                       'input_output_64bit_baseline_bytes': 3*size*size*8})
    return {'status': 'projected', 'kind': 'exact integer recurrence evaluation; no matrix execution',
            'fixed_cutoff': cutoff, 'size_policy': size_policy, 'cycle_radix': radix, 'cycles': cycles,
            'recurrence': model['recurrence'], 'points': points,
            'operand_width_projection': 'not derived; measured widths depend on inputs and coefficients',
            'peak_memory_projection': 'not derived; 64-bit input/output baseline is not exact representation or peak memory',
            'setup_scaling': 'fixed recipe preparation is separate from per-call work; construction/search budgets are independent parameters'}


def project_range(name, cutoff, sizes, plan=None, extensions=12, *, size_policy="fringe"):
    """Budget selected integer sizes for an expandable, initially measured graph.

    Each slider step doubles the largest requested dimension. Small dimensions
    are dense; larger intervals have sixteen subdivisions plus points just before,
    at and after the repeating stage-prefix boundaries on every axis. These exact
    samples expose off-grid fringes without claiming to resolve every large-size
    fluctuation. Only counts are projected; operand values and resources are not.
    One prepared plan serves the whole grid, outside any observed execution.
    """
    requested = tuple(sizes)
    if not requested or any(type(n) is not int or n < 1 for n in requested):
        raise ValueError('projection sizes must be positive integers')
    if type(extensions) is not int or extensions < 1:
        raise ValueError('projection extensions must be a positive integer')
    from .common import validate_size_policy
    validate_size_policy(size_policy)
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError('cutoff must be a positive integer')
    if plan is None and name != 'naive':
        if name == 'smirnov':
            from .smirnov import plan as smirnov_plan
            plan = smirnov_plan()
        else:
            from .composition import Composition
            plan = Composition([name])
    limits = [max(requested)*2**i for i in range(extensions+1)]
    minimum, maximum = min(requested), limits[-1]
    grid = set(requested) | set(range(minimum, min(maximum, 256)+1))
    for start, stop in zip(limits, limits[1:]):
        grid.update(start+(stop-start)*j//16 for j in range(17))
    # A rectangular schedule has several distinct prefix boundaries. Keep each
    # axis, rather than treating the full-cycle square radix as every stage.
    shapes = ([getattr(s, 'shape', (getattr(s, 'radix', 2),)*3) for s in plan.stages]
              if plan is not None else [(2, 2, 2)])
    shape, depth = (1, 1, 1), 0
    while min(shape) <= maximum:
        for boundary in shape:
            grid.update(n for n in (boundary-1, boundary, boundary+1) if minimum <= n <= maximum)
        shape = tuple(d*r for d, r in zip(shape, shapes[depth % len(shapes)]))
        depth += 1
    points = []
    for size in sorted(n for n in grid if minimum <= n <= maximum):
        if plan is None:
            counts = dict.fromkeys(CATEGORIES, 0)
            counts.update(field_multiplications=size**3, field_additions=size**3)
            atomic = 2*size**3
        else:
            budget = plan.budget(size, size, size, cutoff=cutoff, size_policy=size_policy)
            counts = budget.get('category_counts')
            if counts is None:
                counts = category_budget(name, size, cutoff, plan, size_policy=size_policy)
            atomic = budget['atomic_arithmetic_operations']
        points.append({'size': size, 'counts': counts,
                       'field_arithmetic_operations': sum(counts.values()),
                       'atomic_arithmetic_operations': atomic})
    return {'status': 'projected', 'kind': 'exact discrete schedule budgets; no matrix execution',
            'size_policy': size_policy, 'fixed_cutoff': cutoff,
            'sampled_maximum_size': max(requested), 'range_limits': limits, 'points': points,
            'sampling': 'all small integer sizes through 256, larger interval samples and stage-prefix neighbours; connecting lines do not resolve every intervening size',
            'resource_scope': 'no time, operand-width, word-cost or peak-memory projection'}
