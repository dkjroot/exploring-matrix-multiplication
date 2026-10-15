"""Model a portable exact-arithmetic schedule independently of Python internals.

Every 64-bit word product, sum, difference, carry/borrow step, comparison,
shift, and sign operation has weight one. These are declared accounting units,
not equal-duration CPU instructions. Multiplication uses schoolbook rows with
two accumulation/carry steps per word pair. Division uses binary restoring
division: one full-width shift and comparison per dividend bit, with a
subtraction/borrow pass for each set quotient bit. Euclid uses those divisions.

Rationals use denominator-GCD addition and cross-cancelled multiplication,
followed by exact reduction. We replay this mathematical schedule as observation;
the algorithm still uses its original scalar operators, in its original order.
The older unreduced word estimates remain a separate, backwards-compatible model.
"""

from collections import Counter
from fractions import Fraction

from .operand_costs import exact_parts, integer_bits, words


WORK_CATEGORIES = ('word_products', 'word_additions', 'word_subtractions',
                   'carry_borrow_steps', 'word_comparisons', 'word_shifts', 'sign_steps')
SCHEDULE_STATISTICS = WORK_CATEGORIES + ('integer_divisions', 'division_bit_steps',
    'exact_divisions', 'gcd_calls', 'gcd_iterations', 'rational_operations',
    'normalizations', 'normalization_reductions', 'cross_cancellations',
    'word_products_saved_by_cross_cancellation')
MODEL = 'exact-schoolbook-euclid-64-v1'


class ReferenceArithmetic:
    """Replay one scalar call and account for an explicitly specified schedule."""

    def __init__(self):
        """Start independent work and explanatory event counters."""
        self.counts = Counter()

    def product(self, a, b):
        """Multiply signed magnitudes, accumulating each two-word partial product."""
        pairs = words(integer_bits(a))*words(integer_bits(b))
        self.counts.update(word_products=pairs, word_additions=2*pairs,
                           carry_borrow_steps=2*pairs, sign_steps=1)
        return a*b

    def combine(self, a, b, subtract=False):
        """Model a full limb comparison and sum/difference with carry or borrow."""
        width = words(max(integer_bits(a), integer_bits(b)))
        # Opposite signs turn an addition into magnitude subtraction, and vice versa.
        difference = (a < 0) != (b < 0)
        key = 'word_subtractions' if difference != subtract else 'word_additions'
        self.counts.update({key: width, 'carry_borrow_steps': width,
                            'word_comparisons': width, 'sign_steps': 1})
        return a-b if subtract else a+b

    def division(self, a, b, *, exact=False):
        """Account for binary restoring magnitude division and signed correction.

        A remainder register has divisor_bits+1 bits. Every dividend bit gets a
        full-register shift and comparison; a set quotient bit gets subtraction
        and borrow passes. Quotient bit writes and input-bit access are represented
        by one additional word shift per bit. This deliberately simple schedule
        also charges division by one; it is not Python's division implementation.
        """
        if not b:
            raise ZeroDivisionError('reference division by zero')
        q, r = divmod(abs(a), abs(b))
        bits = integer_bits(a)
        width = words(integer_bits(b)+1)
        subtractions = q.bit_count()*width
        self.counts.update(integer_divisions=1, division_bit_steps=bits,
                           word_shifts=bits*(width+1), word_comparisons=bits*width,
                           word_subtractions=subtractions, carry_borrow_steps=subtractions,
                           sign_steps=1)
        if exact:
            self.counts['exact_divisions'] += 1
            if r:
                raise ArithmeticError('reference quotient must be exact')
        return divmod(a, b)

    def gcd(self, a, b):
        """Run ordinary Euclid, exposing every remainder division and zero test."""
        self.counts['gcd_calls'] += 1
        a, b = abs(a), abs(b)
        while b:
            self.counts['gcd_iterations'] += 1
            self.counts['word_comparisons'] += words(integer_bits(b))
            a, b = b, self.division(a, b)[1]
        self.counts['word_comparisons'] += 1
        return a

    def quotient(self, a, b):
        """Skip identity reductions; otherwise perform checked exact division."""
        return a if b == 1 else self.division(a, b, exact=True)[0]

    def rational(self, operation, left, right):
        """Return a reduced rational pair using a documented common schedule."""
        a, b = left
        c, d = right
        self.counts['rational_operations'] += 1
        if operation in ('add', 'subtract'):
            g = self.gcd(b, d)
            s, t = self.quotient(b, g), self.quotient(d, g)
            numerator = self.combine(a if t == 1 else self.product(a, t),
                                     c if s == 1 else self.product(c, s), operation == 'subtract')
            denominator = d if s == 1 else self.product(s, d)
            reduction = self.gcd(numerator, denominator)
            self.counts['normalizations'] += 1
            self.counts['normalization_reductions'] += int(reduction != 1)
            return self.quotient(numerator, reduction), self.quotient(denominator, reduction)
        if operation == 'divide':
            if c == 0:
                raise ZeroDivisionError('rational divisor is zero')
            c, d = d, c
        elif operation != 'multiply':
            raise ValueError('unsupported rational operation')
        if d < 0:
            c, d = -c, -d
            self.counts['sign_steps'] += 1
        g, h = self.gcd(a, d), self.gcd(c, b)
        raw_pairs = words(integer_bits(a))*words(integer_bits(c))
        raw_pairs += words(integer_bits(b))*words(integer_bits(d)) if b != 1 and d != 1 else 0
        a, d = self.quotient(a, g), self.quotient(d, g)
        c, b = self.quotient(c, h), self.quotient(b, h)
        reduced_pairs = words(integer_bits(a))*words(integer_bits(c))
        reduced_pairs += words(integer_bits(b))*words(integer_bits(d)) if b != 1 and d != 1 else 0
        self.counts['cross_cancellations'] += int(g != 1)+int(h != 1)
        self.counts['word_products_saved_by_cross_cancellation'] += max(0, raw_pairs-reduced_pairs)
        numerator = self.product(a, c)
        denominator = d if b == 1 else b if d == 1 else self.product(b, d)
        self.counts['normalizations'] += 1
        return numerator, denominator


class MathematicalWork:
    """Aggregate checked reference schedules without modifying algorithm counts."""

    def __init__(self):
        """Keep bounded aggregates, with unknown operations explicitly recorded."""
        self.counts = Counter()
        self.by_role = Counter()
        self.modelled_calls = Counter()
        self.unmodelled = Counter()

    def observe(self, name, operands, result, role, status):
        """Replay supported exact arithmetic and admit costs only after equality."""
        parts = [exact_parts(value) for value in operands]
        operation = name.rsplit('.', 1)[-1]
        if (status != 'ok' or len(parts) != 2 or any(p is None for p in parts)
                or (operation != 'divmod' and exact_parts(result) is None)):
            self.unmodelled[name] += 1
            return
        ref = ReferenceArithmetic()
        a, b = operands
        if operation == 'divmod' and type(a) is int and type(b) is int:
            expected = ref.division(a, b, exact=isinstance(result, tuple) and result[1] == 0)
            ref.counts['normalizations'] += int(role == 'normalization')
        elif operation in ('multiply', 'add', 'subtract') and type(a) is int and type(b) is int:
            expected = ref.product(a, b) if operation == 'multiply' else ref.combine(a, b, operation == 'subtract')
        elif operation in ('multiply', 'add', 'subtract', 'divide'):
            numerator, denominator = ref.rational(operation, *parts)
            # Comparison uses canonical pairs, avoiding a second Fraction arithmetic call.
            expected = Fraction(numerator, denominator)
        else:
            self.unmodelled[name] += 1
            return
        if expected != result:
            self.unmodelled[name+':reference_mismatch'] += 1
            return
        self.counts.update(ref.counts)
        self.by_role[role or 'combination'] += sum(ref.counts[key] for key in WORK_CATEGORIES)
        self.modelled_calls[name] += 1

    def snapshot(self):
        """Export exact work integers and coverage; partial totals are not scores."""
        total = sum(self.counts[key] for key in WORK_CATEGORIES)
        status = 'partial' if self.unmodelled else 'modelled' if self.modelled_calls else 'no_operations'
        return {'schema': MODEL, 'status': status, 'word_bits': 64,
                'total_work_units': total if status == 'modelled' else None,
                'covered_work_units': total, 'work_categories': {key: self.counts[key] for key in WORK_CATEGORIES},
                'schedule_statistics': {key: self.counts[key] for key in SCHEDULE_STATISTICS},
                'work_by_role': dict(self.by_role),
                'modelled_calls': dict(self.modelled_calls), 'unmodelled_calls': dict(self.unmodelled),
                'weights': {key: 1 for key in WORK_CATEGORIES},
                'scope': 'replayed exact arithmetic schedule; all word categories have weight one; not execution time',
                'excluded_work': ['allocation', 'data movement', 'control flow outside arithmetic', 'setup', 'verification'],
                'is_hardware_instruction_count': False}


def compare_work(candidate, baseline, *, input_hash):
    """Normalize only complete, same-model work against a same-input baseline."""
    result = {'status': 'unavailable', 'input_repr_sha256': input_hash,
              'baseline': 'classical multiplication on identical inputs',
              'meaning': 'classical work / candidate work; higher is better; not a speedup prediction'}
    if (candidate.get('schema') != baseline.get('schema')
            or candidate.get('weights') != baseline.get('weights')
            or candidate.get('word_bits') != baseline.get('word_bits')):
        result['reason'] = 'different arithmetic models'
    elif candidate.get('status') != 'modelled' or baseline.get('status') != 'modelled':
        result['reason'] = 'incomplete exact-arithmetic coverage'
    elif not candidate.get('total_work_units') or not baseline.get('total_work_units'):
        result['reason'] = 'no positive work total'
    else:
        ratio = Fraction(baseline['total_work_units'], candidate['total_work_units'])
        result.update(status='compared', model=candidate['schema'],
                      candidate_work_units=candidate['total_work_units'], baseline_work_units=baseline['total_work_units'],
                      relative_efficiency={'numerator': ratio.numerator, 'denominator': ratio.denominator})
    return result
