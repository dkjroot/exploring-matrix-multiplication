"""Observe exact operand growth and model explicitly defined 64-bit word work.

This module never performs the algorithm's arithmetic. It observes the operands
and result AFTER a primitive call. Integers and fractions remain unchanged.
The schoolbook word model is portable arithmetic accounting, not the interpreter's
big-integer implementation or a count of CPU/GPU instructions. Python may use
faster multiplication, and Fraction may cancel factors before multiplying.

For integers the product estimate is ceil(bits(a)/64)*ceil(bits(b)/64).
For fractions it models unreduced numerator/denominator arithmetic; addition
models cross products and an integer addition, multiplication models numerator
and denominator products. Unit denominators require no denominator arithmetic.
GCD, exact division, allocation, sign handling and carry propagation are excluded.
Both observed widths and those omissions travel with every exported report.
"""

from collections import Counter
from fractions import Fraction


WORD_BITS = 64


def integer_bits(value):
    """Return magnitude bits, giving zero one storage bit without float logs."""
    return max(1, abs(value).bit_length())


def words(bits):
    """Round a positive bit width up to the project's 64-bit word baseline."""
    return (bits+WORD_BITS-1)//WORD_BITS


def exact_parts(value):
    """Recognize exact real scalars without pretending a float is an exact input."""
    if type(value) is int:
        return value, 1
    if type(value) is Fraction:
        return value.numerator, value.denominator
    return None


def word_cost(operation, left, right):
    """Return (word products, word additions/subtractions) for an exact operation.

    Inputs are numerator/denominator pairs. The unreduced rational expression is
    used deliberately: cancellation/GCD complexity is not guessed. Estimates
    for numerator sums use conservative product-width bounds, not an approximate
    value. This is a declared reference schedule, not a minimum bit complexity.
    """
    a, b = left
    c, d = right
    an, bn, cn, dn = map(integer_bits, (a, b, c, d))
    if b == d == 1:
        if operation == 'multiply':
            return words(an)*words(cn), 0
        if operation in ('add', 'subtract'):
            return 0, words(max(an, cn))
    if operation == 'multiply':
        return words(an)*words(cn)+(words(bn)*words(dn) if b != 1 and d != 1 else 0), 0
    if operation in ('add', 'subtract'):
        # a/b +/- c/d = (a*d +/- c*b)/(b*d). A unit denominator
        # simply copies the opposite numerator, avoiding a needless product.
        products = (words(an)*words(dn) if d != 1 else 0)
        products += words(cn)*words(bn) if b != 1 else 0
        products += words(bn)*words(dn) if b != 1 and d != 1 else 0
        width = max(an+dn if d != 1 else an, cn+bn if b != 1 else cn)
        return products, words(width)
    return None


class OperandCosts:
    """Bounded aggregate width histograms and word-cost estimates by scalar role."""

    def __init__(self):
        """Initialize one experiment's statistics without retaining scalar values."""
        self.operations = Counter()
        self.unmodelled = Counter()
        self.word_products = Counter()
        self.word_additions = Counter()
        self.width_histogram = Counter()
        self.by_primitive = {}
        self.max_operand_bits = 0
        self.max_result_bits = 0
        self.max_denominator_bits = 0

    def observe(self, name, operands, result, role, status):
        """Inspect atomic real calls, excluding unknown domains and failed results.

        Gaussian products already expose their real component operations through
        instrumentation.evaluate, so there is no second complex cost estimate.
        Width buckets are powers of two in word count; storage grows with the
        logarithm of the largest width rather than with the number of calls.
        """
        operation = name.rsplit('.', 1)[-1]
        parts = [exact_parts(value) for value in operands]
        result_parts = exact_parts(result)
        label = role or 'combination'
        if status != 'ok' or len(parts) != 2 or any(p is None for p in parts):
            self.unmodelled[name] += 1
            return
        for numerator, denominator in parts:
            bits = max(integer_bits(numerator), integer_bits(denominator))
            self.max_operand_bits = max(self.max_operand_bits, bits)
            self.max_denominator_bits = max(self.max_denominator_bits, integer_bits(denominator))
            bucket = 1 << (words(bits)-1).bit_length()
            self.width_histogram[bucket] += 1
        if result_parts is not None:
            self.max_result_bits = max(self.max_result_bits, *(integer_bits(i) for i in result_parts))
            self.max_denominator_bits = max(self.max_denominator_bits, integer_bits(result_parts[1]))
        estimate = word_cost(operation, *parts)
        if estimate is None:
            self.unmodelled[name] += 1
            return
        self.operations[operation] += 1
        self.word_products[label] += estimate[0]
        self.word_additions[label] += estimate[1]
        entry = self.by_primitive.setdefault(name, {'calls': 0, 'word_multiplications': 0, 'word_additions_subtractions': 0})
        entry['calls'] += 1
        entry['word_multiplications'] += estimate[0]
        entry['word_additions_subtractions'] += estimate[1]

    def snapshot(self):
        """Export observed widths separately from modelled equivalent word work."""
        return {"schema": "exact-operand-word-cost-v1", "word_bits": WORD_BITS,
                "status": 'partial' if self.unmodelled else 'modelled' if self.operations else 'no_operations',
                "model": 'schoolbook 64-bit word products; unreduced rational expressions; unit denominators omitted',
                "is_hardware_instruction_count": False,
                "excluded_work": ['GCD and cancellation', 'division', 'carry propagation', 'sign handling', 'allocation'],
                "observed_max_operand_bits": self.max_operand_bits,
                "observed_max_result_bits": self.max_result_bits,
                "observed_max_denominator_bits": self.max_denominator_bits,
                "operand_word_bucket_upper_bounds": {str(k): v for k, v in sorted(self.width_histogram.items())},
                "modelled_atomic_operations": dict(self.operations),
                "primitive_word_costs": {name: dict(values) for name, values in self.by_primitive.items()},
                "unmodelled_calls": dict(self.unmodelled),
                "equivalent_word_multiplications": sum(self.word_products.values()),
                "equivalent_word_additions_subtractions": sum(self.word_additions.values()),
                "word_multiplications_by_role": dict(self.word_products),
                "word_additions_subtractions_by_role": dict(self.word_additions)}
