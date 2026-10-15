"""Observe logical value traffic, independently of hardware caches and pointers.

The model counts scalar-primitive input/output ports and explicit matrix/vector
copies, initializations, and stores. Payloads use 64-bit magnitude words; a
rational carries numerator and denominator, a Gaussian pair both components.
Copying a Python reference is modelled as transferring its mathematical payload,
not as physically copying a big integer. Repeated uses are counted repeatedly.
This is a schedule-dependent data-flow model, not DRAM/cache traffic or peak live
storage. Observer replay, input validation, setup and verification are excluded.
"""

from collections import Counter

from .operand_costs import exact_parts, integer_bits, words


def payload_words(value):
    """Return exact-domain magnitude payload size, or unknown for opaque domains."""
    if type(value) is int:
        return words(integer_bits(value))
    parts = exact_parts(value)
    if parts is not None:
        return sum(words(integer_bits(part)) for part in parts)
    if type(value).__name__ == 'Gaussian' and hasattr(value, 'real') and hasattr(value, 'imag'):
        sizes = [payload_words(value.real), payload_words(value.imag)]
        return sum(sizes) if all(size is not None for size in sizes) else None
    return None


class DataMovement:
    """Aggregate modelled traffic by primitive ports and explicit structural work."""

    def __init__(self):
        """Start bounded aggregates without retaining any observed values."""
        self.categories = {}

    def record(self, value, action, *, category='structural', repeat=1):
        """Count logical reads/writes; a copy has one of each per payload word."""
        if action not in ('read', 'write', 'copy') or type(repeat) is not int or repeat < 0:
            raise ValueError('traffic needs read/write/copy and a nonnegative integer repeat')
        size = payload_words(value)
        entry = self.categories.setdefault(category, Counter())
        for direction in (('reads', 'writes') if action == 'copy' else (action+'s',)):
            entry['scalar_'+direction] += repeat
            if size is None:
                entry['unknown_payload_'+direction] += repeat
            else:
                entry['word_'+direction] += repeat*size

    def observe(self, operands, result, status):
        """Count every original scalar input; failed calls have no successful output."""
        for value in operands:
            self.record(value, 'read', category='arithmetic_ports')
        if status == 'ok':
            for value in result if isinstance(result, tuple) else (result,):
                self.record(value, 'write', category='arithmetic_ports')

    def snapshot(self):
        """Export exact logical word totals, with partial payload coverage explicit."""
        totals = Counter()
        for entry in self.categories.values():
            totals.update(entry)
        known = totals['word_reads']+totals['word_writes']
        unknown = totals['unknown_payload_reads']+totals['unknown_payload_writes']
        return {'schema': 'logical-value-traffic-64-v1', 'word_bits': 64,
                'status': 'partial' if unknown else 'modelled' if self.categories else 'no_operations',
                'logical_word_transfers': None if unknown else known,
                'covered_word_transfers': known, 'counts': dict(totals),
                'by_category': {key: dict(entry) for key, entry in self.categories.items()},
                'is_hardware_memory_traffic': False,
                'scope': 'scalar ports plus explicit matrix/vector copies, initializations and stores; mathematical payloads',
                'excluded_work': ['cache behavior', 'pointers and object headers', 'internal scalar temporaries',
                                  'setup', 'validation', 'observer replay', 'verification']}
