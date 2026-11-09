"""Exact input fixtures and comparable measurements for batch experiments."""

from fractions import Fraction

from . import Instrumentation, naive, strassen
from .experiments import ExecutionMeasurement
from .verification import check_product


def rational_inputs(size, lanes):
    """Create distinct signed rational inputs in each lane with declared formulas."""
    left = tuple([[Fraction((3*i-2*j+5*lane) % 11-5, 3+lane)
                  for j in range(size)] for i in range(size)] for lane in range(lanes))
    right = tuple([[Fraction((i+4*j+3*lane) % 13-6, 5+lane)
                   for j in range(size)] for i in range(size)] for lane in range(lanes))
    return left, right


def measure_separate(kernel, left, right, target):
    """Measure individual calls together, then compare both results exactly."""
    observer = Instrumentation(profile=False)
    with ExecutionMeasurement() as timing:
        results = tuple(kernel(a, b, instrumentation=observer) for a, b in zip(left, right))
    if results != target:
        raise AssertionError("separate calls disagree with the verified batch outputs")
    return {"observation": observer.snapshot(), "execution_measurements": timing.snapshot()}


def measure_batch(plan, size, *, compare_classical=False):
    """Check the batch budget and exact outputs outside its timed execution."""
    left, right = rational_inputs(size, plan.lanes)
    observer = Instrumentation(profile=False)
    with ExecutionMeasurement() as timing:
        actual = plan.multiply(left, right, instrumentation=observer)
    verified = [check_product(a, b, c).passed for a, b, c in zip(left, right, actual)]
    budget = plan.budget(size)
    if not all(verified) or any(observer.counts[k] != value for k, value in budget.items()):
        raise AssertionError("recursive batch fails the independent output check or arithmetic budget")

    classical_products = plan.lanes*size**3
    result = {"dimension": size, "exact_outputs_verified": verified, "budget": budget,
              "observation": observer.snapshot(), "execution_measurements": timing.snapshot(),
              "separate_strassen": measure_separate(strassen.multiply, left, right, actual),
              "classical_products": classical_products}
    if compare_classical:
        result["separate_classical"] = measure_separate(naive.multiply, left, right, actual)
        if result["separate_classical"]["observation"]["counts"]["field_multiplications"] != classical_products:
            raise AssertionError("separate classical calls disagree with the declared product count")
    return result
