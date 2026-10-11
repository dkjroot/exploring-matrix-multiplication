"""Observe reference algorithms without changing their arithmetic decisions.

Counters distinguish algorithm-level field operations from the real operations
used to implement them. Timings are inclusive and exclusive wall-clock timings
of the instrumented Python code, not predictions of optimized CPU/GPU speed.
Context-local activation lets nested block/scalar primitives share one observer.
Counters update once per operation; they never walk all recursive ancestors.
Optional traces use parent IDs rather than copying a growing call-stack path.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter


@dataclass
class _Frame:
    """One active primitive; parent IDs and child time keep nesting explicit."""

    identifier: int
    depth: int
    child_seconds: float = 0.0


_ACTIVE = ContextVar("matrix_multiplication_instrumentation", default=(None, None))


class Instrumentation:
    """Accumulate counters/profiles and optionally retain or stream trace events.

    A fresh instance describes one experiment. Reusing it deliberately aggregates
    experiments. ``trace_limit`` bounds retained events; a ``sink`` receives every
    event and can stream them elsewhere without retaining them here. Operand/result
    representations are optional because printing huge exact numbers is expensive.
    Each instance is intended for one executing context, not concurrent workers.
    """

    def __init__(self, *, profile=True, trace=False, trace_limit=10000,
                 capture_values=False, sink=None):
        """Configure observation; reject invalid retention bounds and sinks."""
        if type(trace_limit) is not int or trace_limit < 0:
            raise ValueError("trace_limit must be a nonnegative integer")
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable")
        self.profile = profile
        self.trace = trace
        self.trace_limit = trace_limit
        self.capture_values = capture_values
        self.sink = sink
        self.counts = Counter()
        self.primitive_calls = Counter()
        self.profiles = {}
        self.products_by_depth = Counter()
        self.events = []
        self.dropped_events = 0
        self.maximum_depth = 0
        self._next_span = 0
        self._next_event = 0

    def _emit(self, event):
        """Retain a bounded prefix and/or stream an event, with stable ordering."""
        if not (self.trace or self.sink is not None):
            return
        event = {"sequence": self._next_event, **event}
        self._next_event += 1
        if self.trace:
            if len(self.events) < self.trace_limit:
                self.events.append(event)
            else:
                self.dropped_events += 1
        if self.sink is not None:
            # Observer callbacks belong to measurement machinery, not algorithm
            # arithmetic. Suspending observation also prevents recursive tracing.
            with suspended():
                self.sink(event)

    def _profile(self, name, elapsed, exclusive, operations=None):
        """Aggregate one primitive call without storing a record for every call."""
        self.primitive_calls[name] += 1
        entry = self.profiles.setdefault(name, {
            "calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0,
            "inclusive_counts": Counter(),
        })
        entry["calls"] += 1
        entry["inclusive_seconds"] += elapsed
        entry["exclusive_seconds"] += max(0.0, exclusive)
        if operations:
            entry["inclusive_counts"].update(operations)

    def snapshot(self):
        """Return a JSON-compatible report; inclusive profiles overlap by design."""
        atomic = sum(value for key, value in self.counts.items()
                     if key.startswith(("real_", "native_complex_")) or key == "integer_divmods")
        return {
            "counts": dict(self.counts), "atomic_arithmetic_operations": atomic,
            "field_arithmetic_operations": sum(self.counts[key] for key in (
                "field_multiplications", "coefficient_multiplications", "field_additions",
                "field_subtractions", "field_normalizations")),
            "counter_convention": "omitted arithmetic categories have zero observed calls",
            "primitive_calls": dict(self.primitive_calls),
            "profiles": {name: {**entry, "inclusive_counts": dict(entry["inclusive_counts"])}
                         for name, entry in self.profiles.items()},
            "products_by_depth": dict(self.products_by_depth),
            "maximum_depth": self.maximum_depth,
            "retained_trace_events": len(self.events), "dropped_trace_events": self.dropped_events,
        }


@contextmanager
def suspended():
    """Temporarily exclude setup, verification, or observer callbacks from counts.

    ``session(None)`` means inherit the current observer, which is useful when
    recursive primitives cooperate. It cannot disable an enclosing experiment.
    This explicit operation clears both observer and parent span, then restores
    them even if unobserved work raises. Nesting suspensions is safe because each
    ContextVar token restores precisely the context it replaced.
    """
    token = _ACTIVE.set((None, None))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@contextmanager
def session(observer=None):
    """Activate an observer, or inherit an enclosing primitive experiment.

    The token is restored on success and failure. Supplying another observer
    starts an independent observation context with no foreign parent span.
    """
    current, frame = _ACTIVE.get()
    if observer is not None and not isinstance(observer, Instrumentation):
        raise TypeError("instrumentation must be an Instrumentation instance")
    chosen = current if observer is None else observer
    token = _ACTIVE.set((chosen, frame if chosen is current else None))
    try:
        yield chosen
    finally:
        _ACTIVE.reset(token)


@contextmanager
def scope(name, *, depth=None, **metadata):
    """Observe a block, recursive node, or experiment with nested parent IDs.

    Snapshotting a fixed set of counter categories gives inclusive counts in
    constant work per scope. No counter is propagated up a recursive stack.
    """
    observer, parent = _ACTIVE.get()
    if observer is None:
        yield
        return
    identifier = observer._next_span
    observer._next_span += 1
    level = depth if depth is not None else (parent.depth if parent else 0)
    frame = _Frame(identifier, level)
    observer.maximum_depth = max(observer.maximum_depth, level)
    before = observer.counts.copy()
    start = perf_counter() if observer.profile else 0.0
    token = _ACTIVE.set((observer, frame))
    status = "ok"
    try:
        observer._emit({"type": "enter", "span": identifier,
                        "parent": parent.identifier if parent else None,
                        "primitive": name, "depth": level, "metadata": metadata})
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        elapsed = perf_counter() - start if observer.profile else 0.0
        operations = observer.counts - before
        _ACTIVE.reset(token)
        if parent is not None:
            parent.child_seconds += elapsed
        observer._profile(name, elapsed, elapsed - frame.child_seconds, operations)
        observer._emit({"type": "exit", "span": identifier, "primitive": name,
                        "depth": level, "status": status, "seconds": elapsed,
                        "counts": dict(operations)})


def count(category, amount=1):
    """Record a semantic operation or structural cost, with no timing wrapper."""
    observer, frame = _ACTIVE.get()
    if observer is not None:
        observer.counts[category] += amount
        if category == "field_multiplications":
            observer.products_by_depth[frame.depth if frame else 0] += amount


def evaluate(name, category, function, *operands, role=None):
    """Execute and observe one atomic operation, including failed attempts.

    Complex built-in operations stay opaque; the interpreter's internal real
    work is not invented. Our exact pair representation exposes its real work.
    """
    observer, frame = _ACTIVE.get()
    if observer is None:
        return function(*operands)
    start = perf_counter() if observer.profile else 0.0
    status, result = "error", None
    try:
        result = function(*operands)
        status = "ok"
        return result
    finally:
        elapsed = perf_counter() - start if observer.profile else 0.0
        observer.counts[category] += 1
        observer._profile(name, elapsed, elapsed, {category: 1})
        if frame is not None:
            frame.child_seconds += elapsed
        if observer.trace or observer.sink is not None:
            event = {"type": "arithmetic", "parent": frame.identifier if frame else None,
                     "primitive": name, "category": category, "role": role,
                     "depth": frame.depth if frame else 0, "seconds": elapsed, "status": status}
            if observer.capture_values:
                event["operands"] = [repr(value) for value in operands]
                event["result"] = repr(result)
            observer._emit(event)
