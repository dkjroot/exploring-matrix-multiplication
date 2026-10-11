"""Separate mathematical models from secondary execution measurements.

An exponent belongs to an algorithm family and its recurrence, not a particular
timed run. Execution probes surround the public multiplication call and exclude
the independent correctness oracle. Memory tracking is optional and reports only
Python allocations that tracemalloc can see; hardware metrics remain unmeasured.
"""

from math import isfinite, log
from time import perf_counter
import tracemalloc


def arithmetic_model(name):
    """Describe the implemented square family under unit-cost field arithmetic.

    These are formula-derived models, not fitted exponents or claims of exact
    floating-point evaluation. Fixed leaf cutoffs preserve the recursive exponent;
    fixing recursion depth while matrix size grows would instead yield cubic work.
    """
    radix, rank = {
        "naive": (None, None), "strassen": (2, 7),
        "strassen-winograd": (2, 7), "alphaevolve": (4, 48),
    }[name]
    return {
        "square_arithmetic_exponent": 3.0 if radix is None else log(rank, radix),
        "exponent_expression": "3" if radix is None else f"log_{radix}({rank})",
        "parameter_source": "declared algorithm scheme specification; not inferred from executions",
        "derivation": "classical cubic arithmetic" if radix is None else "log(base product count) / log(base dimension)",
        "base_dimension": radix,
        "base_bilinear_rank": rank,
        "recurrence": "Theta(n^3)" if radix is None else f"T(n) = {rank} T(n/{radix}) + O(n^2)",
        "arithmetic_domain": "complex field" if name == "alphaevolve" else "real or complex field",
        "assumptions": ["unit-cost field arithmetic", "fixed classical leaf cutoff",
                        "square inputs grow; recursive depth may grow"],
        "scope": "algorithm family; not an exponent estimated from this run",
        "identity_validation": "complete base identity checked by maintained verification/tests",
    }


class ExecutionMeasurement:
    """Time a public algorithm call and optionally observe traced Python memory.

The measured call includes validation, padding, cold setup, and instrumentation.
It excludes input parsing, the correctness oracle, printing, and report writing.
An existing tracemalloc session is left untouched; in that case a fresh per-call
memory peak is unavailable rather than borrowing a process-wide historical peak.
"""

    def __init__(self, *, measure_memory=False):
        """Configure the optional allocation probe without starting either clock."""
        self.measure_memory = measure_memory
        self.elapsed_seconds = None
        self.python_peak_bytes = None
        self.memory_status = "not_requested"
        self._owns_tracer = False

    def __enter__(self):
        """Start a fresh memory session when possible, then start the call clock."""
        self.elapsed_seconds = None
        self.python_peak_bytes = None
        self.memory_status = "not_requested"
        if self.measure_memory:
            if tracemalloc.is_tracing():
                self.memory_status = "unavailable_existing_tracer"
            else:
                tracemalloc.start()
                self._owns_tracer = True
                self.memory_status = "measured"
        self._started = perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Record partial calls on failure and always restore our memory session."""
        self.elapsed_seconds = perf_counter() - self._started
        if self._owns_tracer:
            try:
                _, self.python_peak_bytes = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
                self._owns_tracer = False
        return False

    def snapshot(self, classical_products=None, verification=None):
        """Export measured diagnostics without inventing hardware observations.

The equivalent operation rate uses the conventional 2*M*N*K field-operation
baseline, not this algorithm's actual work or the chip's floating throughput.
Nonfinite display errors become strings so the report remains strict JSON.
"""
        equivalent = None
        if classical_products is not None and self.elapsed_seconds:
            equivalent = 2 * classical_products / self.elapsed_seconds
        accuracy = {"status": "unavailable", "maximum_absolute_error": None}
        if verification is not None:
            error = verification.max_absolute_error
            accuracy = {
                "status": "measured", "passed": verification.passed,
                "mode": "exact equality" if verification.exact else "floating-point tolerance",
                "maximum_absolute_error": error if isfinite(error) else str(error),
                "failures": verification.failures,
            }
        return {
            "elapsed_algorithm_call_seconds": self.elapsed_seconds,
            "timing_scope": "public Python algorithm call including setup and observation; excluding verification",
            "classical_equivalent_field_operations_per_second": equivalent,
            "equivalent_rate_is_hardware_throughput": False,
            "accuracy": accuracy,
            "python_traced_peak_bytes": {"status": self.memory_status, "value": self.python_peak_bytes},
            "memory_scope": "Python-traced allocations including setup and observation; excludes pre-existing inputs",
            "process_peak_memory_bytes": {"status": "not_measured", "value": None},
            "gpu_peak_memory_bytes": {"status": "not_measured", "value": None},
            "hardware_memory_traffic_bytes": {"status": "not_measured", "value": None},
            "hardware_instruction_count": {"status": "not_measured", "value": None},
            "memory_tracking_enabled": self.measure_memory,
        }
