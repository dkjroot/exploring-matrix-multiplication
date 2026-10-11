"""Execute size/cutoff experiments and compare counted work with recurrence models.

This is an arithmetic-work sweep, not a timing benchmark. Every reported work
count comes from running the algorithm's observed primitives. Predictions and
finite-range log-log fits occupy separate fields and never replace measurements.
Input generation, model arithmetic, fits, and exact verification are unobserved.
"""

from hashlib import sha256
from math import isfinite, log
from random import Random

from . import ALGORITHMS, Instrumentation
from .common import padded_size
from .experiments import arithmetic_model
from .verification import check_product


SCHEMA = "matrix-multiplication-work-sweep-v1"


def work_model(name):
    """Declare scalar costs for this integer sweep, without reading captured counts.

    Strassen's formulas contain 18 block additions/subtractions; Winograd has 15.
    Each nonzero sparse AlphaEvolve coefficient requires a scale and an add. Its
    Gaussian products expand to six component operations, adds to two, and each
    output normalization to two integer divmods. These costs describe our current
    primitive schedule, including operations involving zero/one coefficients.
    """
    if name != "alphaevolve":
        return {"field_leaf_factor": 2, "atomic_leaf_factor": 2,
                "field_block_factor": {"naive": 0, "strassen": 18, "strassen-winograd": 15}[name],
                "atomic_block_factor": {"naive": 0, "strassen": 18, "strassen-winograd": 15}[name],
                "source": "classical dot accumulation and explicit block formulas"}
    from .alphaevolve import _factors
    try:
        nonzeros = sum(len(column) for axis in _factors() for column in axis)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"unavailable_reason": str(exc)}
    return {"field_leaf_factor": 2, "atomic_leaf_factor": 8,
            "field_block_factor": 2 * nonzeros + 16,
            "atomic_block_factor": 8 * nonzeros + 32,
            "coefficient_nonzeros": nonzeros,
            "source": "validated sparse coefficient columns and Gaussian primitive formulas"}


def prediction(name, size, cutoff, costs=None):
    """Predict work from declared recursion and scalar costs, not observed counters.

The smooth curve uses the requested dimension and effective radix-power leaf.
The discrete recurrence rounds the dimension up, then counts classical products
in each leaf. This accounts for padding and cutoffs without evaluating a matrix.
Total-work predictions include additions, scaling, and normalization, so their
smooth curves include lower-order terms rather than only a normalized n**p.
"""
    costs = work_model(name) if costs is None else costs
    model = arithmetic_model(name)
    radix, rank = model["base_dimension"], model["base_bilinear_rank"]
    if radix is None:
        result = {"ideal_data_multiplications": size ** 3,
                "padded_data_multiplications": size ** 3, "padded_dimension": size,
                "effective_leaf_dimension": size, "recursion_levels": 0}
        for layer in ("field", "atomic"):
            result[f"ideal_{layer}_arithmetic_operations"] = costs[f"{layer}_leaf_factor"] * size ** 3
            result[f"padded_{layer}_arithmetic_operations"] = costs[f"{layer}_leaf_factor"] * size ** 3
        return result
    effective = 1
    while effective * radix <= cutoff:
        effective *= radix
    padded = padded_size(size, size, size, radix)
    leaf, levels = padded, 0
    while leaf > cutoff:
        leaf //= radix
        levels += 1
    exponent = model["square_arithmetic_exponent"]
    ideal = size ** 3 if size <= effective else effective ** 3 * (size / effective) ** exponent
    result = {"ideal_data_multiplications": ideal,
              "padded_data_multiplications": rank ** levels * leaf ** 3,
              "padded_dimension": padded, "effective_leaf_dimension": leaf,
              "recursion_levels": levels}
    for layer in ("field", "atomic"):
        if "unavailable_reason" in costs:
            result[f"ideal_{layer}_arithmetic_operations"] = None
            result[f"padded_{layer}_arithmetic_operations"] = None
            continue
        leaf_factor, block_factor = costs[f"{layer}_leaf_factor"], costs[f"{layer}_block_factor"]
        # Sum every level's quadratic overhead separately from its child products.
        overhead = sum(rank ** depth * (padded // radix ** (depth + 1)) ** 2
                       for depth in range(levels))
        result[f"padded_{layer}_arithmetic_operations"] = (
            leaf_factor * rank ** levels * leaf ** 3 + block_factor * overhead)
        # Extend the geometric level sum continuously; exact radix sizes agree
        # with the discrete recurrence, while requested odd sizes avoid padding.
        smooth_overhead = 0 if size <= effective else (
            size ** 2 / radix ** 2 * ((size / effective) ** (exponent - 2) - 1)
            / (rank / radix ** 2 - 1))
        result[f"ideal_{layer}_arithmetic_operations"] = leaf_factor * ideal + block_factor * smooth_overhead
    return result


def log_log_fit(points):
    """Fit log(work) against log(n), labelling the slope as finite-range evidence.

At least two distinct positive sizes are required. A perfect fit to a small
sample still does not prove an asymptotic bound. Padding and lower-order terms
can make the empirical slope differ substantially from the family exponent.
"""
    valid = [(x, y) for x, y in points if x > 0 and y > 0]
    if len({x for x, _ in valid}) < 2:
        return {"status": "insufficient_points", "slope": None, "r_squared": None}
    xs, ys = [log(x) for x, _ in valid], [log(y) for _, y in valid]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / variance
    residual = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - my) ** 2 for y in ys)
    return {"status": "fitted", "slope": slope,
            "r_squared": max(0.0, 1 - residual / total) if total else None,
            "point_count": len(valid), "minimum_size": min(x for x, _ in valid),
            "maximum_size": max(x for x, _ in valid),
            "meaning": "finite-range count-growth slope; not an asymptotic proof"}


def inputs(size, seed):
    """Generate reproducible signed integer matrices shared by every configuration."""
    random = Random(seed + size)
    return ([[random.randrange(-4, 5) for _ in range(size)] for _ in range(size)],
            [[random.randrange(-4, 5) for _ in range(size)] for _ in range(size)])


def collect_work(sizes, cutoffs, *, algorithms=None, seed=1729, progress=None):
    """Run all requested configurations with counting only and exact verification.

Algorithms receive identical inputs at each size. No primitive clocks, memory
probe, or wall-clock fit is enabled. Failures retain their attempted counters and
diagnostics; they are excluded from fitted growth slopes. Sizes have no arbitrary
upper ceiling, but full recursive execution can require substantial resources.
"""
    sizes, cutoffs = sorted(set(sizes)), sorted(set(cutoffs))
    if not sizes or not cutoffs or any(type(value) is not int or value < 1 for value in sizes + cutoffs):
        raise ValueError("sizes and cutoffs must contain positive integers")
    names = list(ALGORITHMS) if algorithms is None else list(dict.fromkeys(algorithms))
    if not names or any(name not in ALGORITHMS for name in names):
        raise ValueError("choose at least one registered algorithm")
    costs = {name: work_model(name) for name in names}
    series = [{"algorithm": name, "cutoff": cutoff, "mathematical_model": arithmetic_model(name),
               "work_model": costs[name], "samples": []}
              for name in names for cutoff in cutoffs]
    for size in sizes:
        left, right = inputs(size, seed)
        input_hash = sha256(repr((left, right)).encode("utf-8")).hexdigest()
        for line in series:
            name, cutoff = line["algorithm"], line["cutoff"]
            if progress:
                progress(f"Counting {name}, n={size}, cutoff={cutoff}")
            observer = Instrumentation(profile=False)
            sample = {"size": size, "input_repr_sha256": input_hash,
                      "prediction": prediction(name, size, cutoff, costs[name])}
            try:
                result = ALGORITHMS[name](left, right, cutoff=cutoff, instrumentation=observer)
                verification = check_product(left, right, result)
                sample.update(status="passed" if verification.passed else "failed",
                              verification={"passed": verification.passed, "mode": "exact equality",
                                            "maximum_absolute_error": (verification.max_absolute_error
                                                if isfinite(verification.max_absolute_error) else str(verification.max_absolute_error)),
                                            "failures": verification.failures})
            except (ArithmeticError, ValueError, TypeError, OSError) as exc:
                sample.update(status="error", error=str(exc))
            sample["observation"] = observer.snapshot()
            sample["prediction_matches_observed_data_multiplications"] = (
                observer.counts["field_multiplications"] == sample["prediction"]["padded_data_multiplications"]
                if sample["status"] == "passed" else None)
            sample["prediction_matches_observed_arithmetic_steps"] = (
                all(sample["observation"][metric] == sample["prediction"][f"padded_{metric}"]
                    for metric in ("field_arithmetic_operations", "atomic_arithmetic_operations"))
                if sample["status"] == "passed" and "unavailable_reason" not in costs[name] else None)
            line["samples"].append(sample)
    for line in series:
        minimum, maximum = sizes[0], sizes[-1]
        curve_sizes = sorted(set(sizes + [minimum + (maximum - minimum) * index / 80 for index in range(81)]))
        line["ideal_curve"] = []
        for size in curve_sizes:
            predicted = prediction(line["algorithm"], size, line["cutoff"], line["work_model"])
            line["ideal_curve"].append({"size": size,
                "data_multiplications": predicted["ideal_data_multiplications"],
                "field_arithmetic_operations": predicted["ideal_field_arithmetic_operations"],
                "atomic_arithmetic_operations": predicted["ideal_atomic_arithmetic_operations"]})
        line["empirical_fits"] = {}
        for metric in ("field_multiplications", "field_arithmetic_operations", "atomic_arithmetic_operations"):
            points = [(sample["size"], sample["observation"]["counts"].get(metric, 0)
                       if metric == "field_multiplications" else sample["observation"][metric])
                      for sample in line["samples"] if sample["status"] == "passed"]
            line["empirical_fits"][metric] = log_log_fit(points)
    return {"schema": SCHEMA, "configuration": {"sizes": sizes, "cutoffs": cutoffs, "seed": seed,
            "scalar_domain": "signed Python integers; complex intermediates for AlphaEvolve",
            "primitive_timing_enabled": False, "focus": "executed arithmetic work, not elapsed time"},
            "series": series}
