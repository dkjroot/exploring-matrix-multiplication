"""Proved value and word-work bounds for bounded integer inputs.

This is abstract arithmetic, not a fit to observations or a matrix execution.
Each scalar represents |x| <= magnitude with a reduced denominator dividing D.
Triangle inequalities and common denominators enclose every permitted input.
Maintained formulas run on one abstract entry per block; logical block areas
weight the costs. One uniform child enclosure bounds all rank children, avoiding
an exponentially large tree. Complete recipe identities bound returned products.
Python object/allocator/observer memory and division/GCD/carry costs stay excluded.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod

from . import scalars, strassen, strassen_winograd
from .common import padded_dimensions, validate_size_policy
from .operand_costs import integer_bits, words
from .instrumentation import suspended
from .scalars import Gaussian
from .schemes import bilinear_step


@dataclass(frozen=True)
class Enclosure:
    """Enclose a reduced rational by its absolute value and common denominator."""

    magnitude: Fraction
    denominator: int = 1

    def __post_init__(self):
        """Canonicalize guaranteed zero and reject invalid mathematical bounds."""
        object.__setattr__(self, 'magnitude', Fraction(self.magnitude))
        if self.magnitude < 0 or type(self.denominator) is not int or self.denominator < 1:
            raise ValueError('enclosures require nonnegative magnitude and positive denominator')
        if self.numerator_bound == 0:
            object.__setattr__(self, 'magnitude', Fraction(0))
            object.__setattr__(self, 'denominator', 1)

    @property
    def numerator_bound(self):
        """Bound the stored numerator using |numerator| <= magnitude * D."""
        return self.magnitude.numerator*self.denominator//self.magnitude.denominator

    @classmethod
    def constant(cls, value):
        """Enclose one exact real coefficient, retaining its reduced denominator."""
        value = Fraction(value)
        return cls(abs(value), value.denominator)

    def plus(self, other):
        """Enclose a sum or difference without assuming any cancellation."""
        return Enclosure(self.magnitude+other.magnitude, lcm(self.denominator, other.denominator))

    def times(self, other):
        """Enclose a product, allowing subsequent exact reduction to shrink it."""
        return Enclosure(self.magnitude*other.magnitude, self.denominator*other.denominator)

    def multiple(self, count):
        """Enclose a sum of count terms with a shared denominator, not a product."""
        return Enclosure(count*self.magnitude, self.denominator)


def join_enclosures(values):
    """Find one enclosure valid for every child, using a denominator LCM."""
    return Enclosure(max(v.magnitude for v in values), lcm(*(v.denominator for v in values)))


class BoundCosts:
    """Accumulate upper bounds on the existing declared 64-bit word-cost model."""

    def __init__(self):
        """Start one node's weighted abstract schedule, with no scalar samples."""
        self.atomic = Counter()
        self.word_products = 0
        self.word_sums = 0
        self.operand_bits = 1
        self.result_bits = 1
        self.numerator_bits = 1
        self.denominator_bits = 1
        self.repeat = 1
        self.normalization_denominator = None

    def record(self, operation, left, right, result, area):
        """Bound a scalar call using monotone operand-width bounds and its area.

        Nonunit universal denominators may hide unit actual denominators. Counting
        the extra cross products is conservative. Exact division is recorded for
        coverage and width checks, but has no invented word-work estimate.
        """
        weight = area*self.repeat
        self.atomic[operation] += weight
        for value in (left, right, result):
            self.numerator_bits = max(self.numerator_bits, integer_bits(value.numerator_bound))
            self.denominator_bits = max(self.denominator_bits, integer_bits(value.denominator))
        for value in (left, right):
            self.operand_bits = max(self.operand_bits, integer_bits(value.numerator_bound),
                                    integer_bits(value.denominator))
        self.result_bits = max(self.result_bits, integer_bits(result.numerator_bound), integer_bits(result.denominator))
        an, bn = integer_bits(left.numerator_bound), integer_bits(left.denominator)
        cn, dn = integer_bits(right.numerator_bound), integer_bits(right.denominator)
        b, d = left.denominator, right.denominator
        if operation == 'multiply':
            self.word_products += weight*(words(an)*words(cn)+(words(bn)*words(dn) if b != 1 and d != 1 else 0))
        elif operation in ('add', 'subtract'):
            self.word_products += weight*((words(an)*words(dn) if d != 1 else 0)
                +(words(cn)*words(bn) if b != 1 else 0)+(words(bn)*words(dn) if b != 1 and d != 1 else 0))
            width = max(an+dn if d != 1 else an, cn+bn if b != 1 else cn)
            self.word_sums += weight*words(width)

    def merge(self, child, count):
        """Bound sequential child work with rank copies, retaining maximum widths."""
        self.atomic.update({k: count*v for k, v in child.atomic.items()})
        self.word_products += count*child.word_products
        self.word_sums += count*child.word_sums
        for key in ('operand_bits', 'result_bits', 'numerator_bits', 'denominator_bits'):
            setattr(self, key, max(getattr(self, key), getattr(child, key)))


class BoundScalar:
    """One abstract real entry with a logical block area, never a matrix data value."""

    def __init__(self, enclosure, area, costs):
        """Attach a mathematical enclosure to this node's model accounting."""
        self.enclosure, self.area, self.costs = enclosure, area, costs

    def operate(self, other, operation):
        """Propagate one scalar inequality through a maintained formula operation."""
        if isinstance(other, BoundScalar):
            if self.area != other.area or self.costs is not other.costs:
                raise ValueError('abstract scalar operations require a common block area and node')
            right = other.enclosure
        else:
            right = Enclosure.constant(other)
        left = self.enclosure
        if operation == 'multiply':
            result = left.times(right)
        elif operation in ('add', 'subtract'):
            result = left.plus(right)
        else:
            # The only divisions in maintained stages are final normalizations.
            # The verified identity makes each returned entry an exact product
            # sum, whose denominator divides the input-product denominator.
            if not right.magnitude:
                raise ZeroDivisionError('zero normalization')
            result = Enclosure(left.magnitude/right.magnitude,
                               self.costs.normalization_denominator or left.denominator*right.numerator_bound)
        self.costs.record(operation, left, right, result, self.area)
        return BoundScalar(result, self.area, self.costs)

    def __add__(self, other):
        """Bound addition, including the schedule's literal initial zero add."""
        return self.operate(other, 'add')

    def __radd__(self, other):
        """Preserve the symmetric word-cost bound when a literal is on the left."""
        return self.__add__(other)

    def __sub__(self, other):
        """Bound a difference by the sum of absolute-value bounds."""
        return self.operate(other, 'subtract')

    def __rsub__(self, other):
        """A reversed difference has the same magnitude and symmetric word cost."""
        return self.__sub__(other)

    def __mul__(self, other):
        """Bound one data or coefficient product without skipping zero operands."""
        return self.operate(other, 'multiply')

    def __rmul__(self, other):
        """Bound an exact coefficient times an abstract data entry."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Track normalization widths while leaving division cost unmodelled."""
        return self.operate(other, 'divide')


def abstract_value(enclosures, area, costs, gaussian):
    """Wrap one or two component bounds for the actual scalar formula schedule."""
    values = tuple(BoundScalar(value, area, costs) for value in enclosures)
    return Gaussian(*values) if gaussian else values[0]


def components(value):
    """Extract real component enclosures from a one-entry abstract block value."""
    return tuple(v.enclosure for v in (value.real, value.imag)) if isinstance(value, Gaussian) else (value.enclosure,)


def product_enclosures(left, right, inner):
    """Use the complete identity to bound a returned dot product, not its expansion."""
    if len(left) == 1:
        return (left[0].times(right[0]).multiple(inner),)
    ar, ai = left
    br, bi = right
    return (ar.times(br).plus(ai.times(bi)).multiple(inner),
            ar.times(bi).plus(ai.times(br)).multiple(inner))


def dot_work(left, right, inner, terms, costs, gaussian):
    """Bound repeated dot terms without iterating over a giant inner dimension.

    Before any accumulation, its magnitude is at most (inner-1) times a product
    enclosure. This includes a core's already computed prefix and each remaining
    fringe update. Repeating the maximum per-term cost bounds every prefix.
    """
    product = product_enclosures(left, right, 1)
    costs.repeat = terms
    value = scalars.multiply(abstract_value(left, 1, costs, gaussian), abstract_value(right, 1, costs, gaussian))
    accumulator = tuple(v.multiple(inner-1) for v in product)
    scalars.add(abstract_value(accumulator, 1, costs, gaussian), value)
    costs.repeat = 1


def node_bound(stages, shape, left, right, cutoff, depth, gaussian):
    """Bound a depth-first node using one uniform child, fixed rank and true areas."""
    costs = BoundCosts()
    m, k, n = shape
    stage = stages[depth % len(stages)] if stages else None
    radix = getattr(stage, 'shape', (getattr(stage, 'radix', 2),)*3) if stage else (2,)*3
    core = tuple(d-d % r for d, r in zip(shape, radix))
    if stage is None or max(shape) <= cutoff or not all(core):
        dot_work(left, right, k, m*k*n, costs, gaussian)
        return costs
    cm, ck, cn = core
    child_shape = tuple(d//r for d, r in zip(core, radix))
    a, b, c = child_shape
    child_left, child_right = [], []
    returned = product_enclosures(left, right, k)
    costs.normalization_denominator = lcm(*(value.denominator for value in returned))

    def child_product(a_block, b_block):
        """Collect child input enclosures and return the identity's dot enclosure."""
        la, rb = components(a_block[0][0]), components(b_block[0][0])
        child_left.append(la)
        child_right.append(rb)
        return [[abstract_value(product_enclosures(la, rb, b), a*c, costs, gaussian)]]

    a_blocks = [[[abstract_value(left, a*b, costs, gaussian)]] for _ in range(radix[0]*radix[1])]
    b_blocks = [[[abstract_value(right, b*c, costs, gaussian)]] for _ in range(radix[1]*radix[2])]
    scheme = getattr(stage, 'scheme', None)
    if scheme is None and hasattr(stage, 'left'):
        scheme = stage
    if scheme is not None:
        bilinear_step(a_blocks, b_blocks, child_product, scheme)
    else:
        formula = {'strassen': strassen.step, 'strassen-winograd': strassen_winograd.step}.get(stage.name)
        if formula is None:
            raise ValueError('resource bounds require a maintained formula or verified sparse scheme')
        formula(a_blocks, b_blocks, child_product)
    if len(child_left) != stage.rank:
        raise AssertionError('abstract schedule disagrees with the declared rank')
    la = tuple(join_enclosures([v[i] for v in child_left]) for i in range(len(left)))
    rb = tuple(join_enclosures([v[i] for v in child_right]) for i in range(len(right)))
    child_cost = node_bound(stages, child_shape, la, rb, cutoff, depth+1, gaussian)
    costs.merge(child_cost, stage.rank)
    if core != shape:
        dot_work(left, right, k, prod(shape)-prod(core), costs, gaussian)
    return costs


def resource_bound(plan, size, cutoff=1, *, input_bound=4, size_policy='fringe'):
    """Prove bounds for any real integer matrices with |entry| <= input_bound.

    The plan must already have complete exact identity validation.
    """
    if any(type(v) is not int or v < 1 for v in (size, cutoff)):
        raise ValueError('size and cutoff must be positive integers')
    if type(input_bound) is not int or input_bound < 0:
        raise ValueError('input bound must be a nonnegative integer')
    validate_size_policy(size_policy)
    stages = tuple(plan.stages) if plan is not None else ()
    gaussian = bool(getattr(plan, 'gaussian', False))
    shape = (size,)*3
    if stages and size_policy == 'padded':
        shapes = [getattr(s, 'shape', (getattr(s, 'radix', 2),)*3) for s in stages]
        shape = padded_dimensions(size, size, size, shapes)
    root = (Enclosure(input_bound), Enclosure(0)) if gaussian else (Enclosure(input_bound),)
    with suspended():
        costs = node_bound(stages, shape, root, root, cutoff, 0, gaussian)
    return {'operand_bits_upper_bound': costs.operand_bits,
            'result_bits_upper_bound': costs.result_bits,
            'numerator_bits_upper_bound': costs.numerator_bits,
            'denominator_bits_upper_bound': costs.denominator_bits,
            'word_multiplications_upper_bound': costs.word_products,
            'word_additions_subtractions_upper_bound': costs.word_sums,
            'modelled_atomic_calls': dict(costs.atomic),
            'input_absolute_bound': input_bound,
            'is_observed_value_prediction': False,
            'is_python_memory_peak_bound': False}


def bound_range(name, cutoff, samples, projection, plan=None, *, input_bound=4, size_policy='fringe'):
    """Bound selected dimensions and independently check all captured executions.

    Use four interval subdivisions per doubling in addition to sampled
    dimensions and stage-prefix neighbours. Every saved point has its own proof;
    connecting segments do not assert a bound at unsaved integer dimensions.
    """
    if projection.get('status') != 'projected':
        return {'status': 'unavailable', 'reason': 'no verified reference traversal'}
    if plan is None and name != 'naive':
        if name == 'smirnov':
            from .smirnov import plan as smirnov_plan
            plan = smirnov_plan()
        else:
            from .composition import Composition
            plan = Composition([name])
    limits = projection['range_limits']
    grid = {s['size'] for s in samples} | set(limits)
    for start, stop in zip(limits, limits[1:]):
        grid.update(start+(stop-start)*i//4 for i in range(5))
    shapes = ([getattr(s, 'shape', (getattr(s, 'radix', 2),)*3) for s in plan.stages]
              if plan is not None else [(2, 2, 2)])
    shape, depth = (1, 1, 1), 0
    minimum = min(s['size'] for s in samples)
    while min(shape) <= limits[-1]:
        for boundary in shape:
            grid.update(n for n in (boundary-1, boundary, boundary+1) if minimum <= n <= limits[-1])
        shape = tuple(d*r for d, r in zip(shape, shapes[depth % len(shapes)]))
        depth += 1
    points = [{'size': n, **resource_bound(plan, n, cutoff, input_bound=input_bound,
                                         size_policy=size_policy)} for n in sorted(grid)]
    lookup = {p['size']: p for p in points}
    pairs = (('observed_max_operand_bits', 'operand_bits_upper_bound'),
             ('observed_max_result_bits', 'result_bits_upper_bound'),
             ('observed_max_denominator_bits', 'denominator_bits_upper_bound'),
             ('equivalent_word_multiplications', 'word_multiplications_upper_bound'),
             ('equivalent_word_additions_subtractions', 'word_additions_subtractions_upper_bound'))
    for sample in samples:
        if sample['status'] != 'passed':
            continue
        point, observation = lookup[sample['size']], sample['observation']
        checked = all(observation['operand_costs'][actual] <= point[upper] for actual, upper in pairs)
        checked &= observation['atomic_arithmetic_operations'] == sum(point['modelled_atomic_calls'].values())
        sample['resource_bounds_check_passed'] = checked
        if not checked:
            sample.update(status='failed', error='resource bound or abstract schedule validation failed')
    return {'status': 'bounded' if all(s.get('resource_bounds_check_passed', True) for s in samples) else 'failed',
            'kind': 'proved upper bounds, not predicted observations',
            'input_absolute_bound': input_bound, 'range_limits': limits, 'points': points,
            'assumptions': 'real integer inputs with entry magnitudes at most the declared bound; fixed verified recipe and cutoff',
            'proof': 'exact rational triangle inequalities and denominator divisibility; maintained scalar schedules; uniform enclosure of recursive children',
            'excluded': ['Python memory peaks', 'elapsed time', 'GCD and cancellation cost', 'division cost', 'carry propagation', 'allocation'],
            'sampling': 'sampled dimensions, slider boundaries, stage-prefix neighbours and four subdivisions per doubling; no guarantee on connecting segments'}
