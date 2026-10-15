"""Independent exact operation budgets for recursive cores and classical fringes.

Only dimensions and verified stage descriptions enter this calculation. It never
executes a matrix or reads observed counters. Equal core children share a shape,
so one path with node multiplicities suffices even for enormous projections.
"""

from math import prod

from .common import validate_size_policy
from .scaling import CATEGORIES, stage_counts


def fringe_budget(stages, shape, cutoff, *, gaussian=False):
    """Count a repeating stage path, including every unpadded boundary term.

    At a node of shape (m,k,n), the core rounds each axis down to a multiple of
    its stage count. Exactly m*k*n-core_m*core_k*core_n dot terms remain. Each
    contributes one data product and one accumulation add. Gaussian products
    expose six component operations; their accumulation adds two more.
    """
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    if any(type(d) is not int or d < 1 for d in shape):
        raise ValueError("dimensions must be positive integers")
    requested = shape
    nodes, depth, atomic, fringe_products = 1, 0, 0, 0
    counts = dict.fromkeys(CATEGORIES, 0)
    levels = []
    while max(shape) > cutoff:
        stage = stages[depth % len(stages)]
        radix = stage.shape if hasattr(stage, 'shape') else (stage.radix,)*3
        core = tuple(d-d % r for d, r in zip(shape, radix))
        if not all(core):
            break
        child = tuple(d//r for d, r in zip(core, radix))
        fringe = nodes*(prod(shape)-prod(core))
        fringe_products += fringe
        counts['field_multiplications'] += fringe
        counts['field_additions'] += fringe
        overhead = stage_counts(stage, child)
        for key, value in overhead.items():
            counts[key] += nodes*value
        component = (6*overhead['coefficient_multiplications']
                     + 2*(overhead['field_additions']+overhead['field_subtractions']
                          + overhead['field_normalizations'])) if gaussian else sum(overhead.values())
        atomic += nodes*component + (8 if gaussian else 2)*fringe
        levels.append({'depth': depth, 'stage': stage.name, 'nodes': nodes,
                       'size': list(shape), 'radix': list(radix), 'rank': stage.rank,
                       'core_dimensions': list(core), 'child_dimensions': list(child),
                       'fringe_data_products': fringe})
        nodes *= stage.rank
        shape, depth = child, depth+1
    products = nodes*prod(shape)
    counts['field_multiplications'] += products
    counts['field_additions'] += products
    atomic += (8 if gaussian else 2)*products
    return {'size_policy': 'fringe', 'padded_size': max(requested),
            'padded_dimensions': list(requested), 'leaf_size': max(shape),
            'leaf_dimensions': list(shape), 'levels': levels, 'classical_leaves': nodes,
            'fringe_data_products': fringe_products, 'category_counts': counts,
            'field_multiplications': counts['field_multiplications'],
            'field_arithmetic_operations': sum(counts.values()),
            'atomic_arithmetic_operations': atomic,
            'atomic_domain': 'Gaussian components' if gaussian else 'real scalars; native complex is opaque'}
