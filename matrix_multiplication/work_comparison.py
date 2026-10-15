"""Bind normalized work scores to a verified classical run on identical inputs.

Baseline execution is separate reference-accounting work. It is never included
in a candidate's timers, allocation probes, arithmetic counters, or projections.
"""

from hashlib import sha256

from .instrumentation import Instrumentation, suspended, session
from .common import classical, validate
from .mathematical_work import compare_work
from .verification import check_product


def classical_reference(a, b):
    """Execute and verify the classical baseline without an enclosing observer."""
    observer = Instrumentation(profile=False)
    with suspended():
        validate(a, b)
        with session(observer):
            result = classical(a, b)
        verification = check_product(a, b, result)
    snapshot = observer.snapshot()
    return {'status': 'passed' if verification.passed else 'failed',
            'input_repr_sha256': sha256(repr((a, b)).encode('utf-8')).hexdigest(),
            'scope': 'separate classical reference; excluded from candidate measurements',
            'mathematical_work': snapshot['mathematical_work'],
            'data_movement': snapshot['data_movement']}


def attach_comparison(report, reference, *, input_hash):
    """Attach a score only to verified candidate and same-input reference evidence."""
    if (report.get('status') != 'passed' or reference.get('status') != 'passed'
            or reference.get('input_repr_sha256') != input_hash):
        report['work_comparison'] = {'status': 'unavailable', 'reason': 'verification or input binding failed'}
    else:
        report['work_comparison'] = compare_work(report.get('mathematical_work', {}),
                                                reference['mathematical_work'], input_hash=input_hash)
