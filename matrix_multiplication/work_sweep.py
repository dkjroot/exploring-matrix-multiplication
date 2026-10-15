"""Execute size/cutoff experiments and compare counted work with recurrence models.

This is an arithmetic-work sweep, not a timing benchmark. Every reported work
count comes from running the algorithm's observed primitives. Predictions and
finite-range log-log fits occupy separate fields and never replace measurements.
Input generation, model arithmetic, fits, and exact verification are unobserved.
Cutoffs with the same known traversal reuse one verified measurement, explicitly
labelled as reused evidence rather than an independently executed experiment.
"""

from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite, log
from random import Random

from . import ALGORITHMS, Instrumentation
from .common import padded_size
from .experiments import ExecutionMeasurement, arithmetic_model
from .scaling import CATEGORIES, category_budget, project_work, project_range
from .size_effects import describe_sample, describe_transition
from .verification import check_product
from .work_comparison import classical_reference, attach_comparison


SCHEMA = "matrix-multiplication-work-sweep-v1"
INPUT_MAGNITUDE_BOUND = 4


def positive_integer_grid(values, name):
    """Validate each requested value before deduplicating and ordering a grid.

    True equals 1 in Python sets. Validating after set conversion can therefore
    hide an invalid boolean when a legitimate integer appeared first. Validate
    the original entries, then canonicalize them so API and CLI runs agree.
    """
    requested = list(values)
    if not requested or any(type(value) is not int or value < 1 for value in requested):
        raise ValueError(f"{name} must contain positive integers")
    return sorted(set(requested))


def schedule_work_model(plan):
    """Describe per-stage budgets without treating a cycle as one uniform split.

    Rectangular stages have different block areas; square mixed-radix plans have
    different cutoff phases. Both use the plan's exact discrete budget instead
    of inventing a homogeneous smooth curve from the cycle's square exponent.
    """
    return {"source": "verified repeating stage schedule; level-specific arithmetic budgets",
            "stages": [{"name": s.name, "rank": s.rank, "radix": s.radix,
                        "dimensions": list(s.shape) if hasattr(s, "shape") else None,
                        "field_block_factor": getattr(s, "field_block_cost", None)} for s in plan.stages],
            "ideal_curve_status": "omitted: repeating stage cutoffs have phase-dependent constants"}


def schedule_prediction(plan, size, cutoff, *, size_policy="fringe"):
    """Translate an independently calculated stage budget into sweep fields."""
    budget = plan.budget(size, size, size, cutoff=cutoff, size_policy=size_policy)
    result = {"size_policy": size_policy,
              "padded_dimension": budget["padded_size"],
              "effective_leaf_dimension": budget["leaf_size"],
              "recursion_levels": len(budget["levels"]),
              "padded_data_multiplications": budget["field_multiplications"],
              "padded_field_arithmetic_operations": budget["field_arithmetic_operations"],
              "padded_atomic_arithmetic_operations": budget["atomic_arithmetic_operations"],
              "composition_budget": budget}
    if "padded_dimensions" in budget:
        result["padded_dimensions"] = budget["padded_dimensions"]
    return result


def work_model(name):
    """Declare scalar costs for this integer sweep, without reading captured counts.

    Strassen's formulas contain 18 block additions/subtractions; Winograd has 15.
    Each nonzero sparse AlphaEvolve coefficient requires a scale and an add. Its
    Gaussian products expand to six component operations, adds to two, and each
    output normalization to two integer divmods. These costs describe our current
    primitive schedule, including operations involving zero/one coefficients.
    """
    if name == "smirnov":
        from .smirnov import plan
        try:
            return schedule_work_model(plan())
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return {"unavailable_reason": str(exc),
                    "ideal_curve_status": "omitted: rectangular stage coefficients unavailable"}
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


def prediction(name, size, cutoff, costs=None, *, size_policy="fringe"):
    """Predict work from declared recursion and scalar costs, not observed counters.

The smooth curve uses the requested dimension and effective radix-power leaf.
The default discrete budget rounds cores down and counts their boundary terms.
The baseline rounds the entire input up. Both count leaves without executing data.
Total-work predictions include additions, scaling, and normalization, so their
smooth curves include lower-order terms rather than only a normalized n**p.
"""
    from .common import validate_size_policy
    validate_size_policy(size_policy)
    costs = work_model(name) if costs is None else costs
    if name == "smirnov":
        from .smirnov import plan
        return schedule_prediction(plan(), size, cutoff, size_policy=size_policy)
    if size_policy == "fringe" and name != "naive":
        from .composition import Composition
        result = prediction(name, size, cutoff, costs, size_policy="padded")
        result.update(schedule_prediction(Composition([name]), size, cutoff, size_policy=size_policy))
        return result
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
    return ([[random.randrange(-INPUT_MAGNITUDE_BOUND, INPUT_MAGNITUDE_BOUND+1) for _ in range(size)] for _ in range(size)],
            [[random.randrange(-INPUT_MAGNITUDE_BOUND, INPUT_MAGNITUDE_BOUND+1) for _ in range(size)] for _ in range(size)])


def cutoff_execution_structure(name, implementation, predicted, plan=None):
    """Identify equivalent cutoffs only for our known deterministic traversals.

    Within one algorithm and input, cutoff controls only the leaf predicate. The
    same padding, ordered stages, and leaf shape therefore imply the same scalar
    operations in the same order. Matching totals alone would not establish this.
    Replaced registry callbacks and unfamiliar plan classes return None, so they
    still execute independently even if they advertise identical work budgets.
    """
    from . import alphaevolve, naive, smirnov, strassen, strassen_winograd
    from .composition import Composition
    from .rectangular import RectangularCycle

    known = {"naive": naive.multiply, "strassen": strassen.multiply,
             "strassen-winograd": strassen_winograd.multiply,
             "alphaevolve": alphaevolve.multiply, "smirnov": smirnov.multiply}
    known_callback = implementation is known.get(name)
    known_plan = (type(plan) in (Composition, RectangularCycle)
                  and getattr(implementation, "__self__", None) is plan
                  and getattr(implementation, "__func__", None) is type(plan).multiply)
    if not (known_callback or known_plan):
        return None
    budget = predicted.get("composition_budget")
    if budget is not None:
        return {"size_policy": budget.get("size_policy", "padded"),
                "padded_dimensions": budget.get("padded_dimensions", [budget["padded_size"]] * 3),
                "leaf_dimensions": budget.get("leaf_dimensions", [budget["leaf_size"]] * 3),
                "stages": [{key: level[key] for key in ("depth", "stage", "size", "radix", "rank")}
                           for level in budget["levels"]]}
    return {"padded_dimensions": [predicted["padded_dimension"]] * 3,
            "leaf_dimensions": [predicted["effective_leaf_dimension"]] * 3,
            "recursion_levels": predicted["recursion_levels"]}


def collect_work(sizes, cutoffs, *, algorithms=None, seed=1729, progress=None, compositions=None,
                 reuse_equivalent_cutoffs=True, measure_memory=True, projection_cycles=12,
                 size_policy="fringe"):
    """Run all requested configurations with counting only and exact verification.

Algorithms receive identical inputs at each size. Primitive clocks and wall-clock
fits are disabled; an optional per-call Python allocation probe surrounds only
the public execution. Failures retain their attempted counters and
diagnostics; they are excluded from fitted growth slopes. Sizes have no arbitrary
upper ceiling, but full recursive execution can require substantial resources.
By default, proven-equivalent cutoff traversals share one successful measurement
within the same algorithm/input. No result is reused across sizes or algorithms,
and failed outputs/budget mismatches are never cached. Disable reuse to request
independent executions of every cutoff, for example when checking the optimizer.
"""
    from .common import validate_size_policy
    validate_size_policy(size_policy)
    sizes = positive_integer_grid(sizes, "sizes")
    cutoffs = positive_integer_grid(cutoffs, "cutoffs")
    if type(projection_cycles) is not int or projection_cycles < 1:
        raise ValueError("projection cycles must be a positive integer")
    names = list(ALGORITHMS) if algorithms is None else list(dict.fromkeys(algorithms))
    if not names or any(name not in ALGORITHMS for name in names):
        raise ValueError("choose at least one registered algorithm")
    costs = {name: work_model(name) for name in names}
    compositions = compositions or {}
    if any(not isinstance(name, str) or not name or name in ALGORITHMS for name in compositions):
        raise ValueError("composition names must be nonempty and distinct from registered algorithms")
    implementations = {**ALGORITHMS, **{name: plan.multiply for name, plan in compositions.items()}}
    names.extend(compositions)
    # Registered staged algorithms and user compositions share the same model
    # and budget path; execution still dispatches through the public registry.
    plans = dict(compositions)
    if "smirnov" in names and "unavailable_reason" not in costs["smirnov"]:
        from .smirnov import plan as smirnov_plan
        plans["smirnov"] = smirnov_plan()
    costs.update({name: schedule_work_model(plan) for name, plan in plans.items()})
    series = [{"algorithm": name, "cutoff": cutoff, "mathematical_model":
               plans[name].model() if name in plans else arithmetic_model(name),
               "work_model": costs[name], "size_policy": size_policy, "samples": []}
              for name in names for cutoff in cutoffs]
    for line in series:
        # Projection setup must not replace successful/failed execution evidence.
        # Unavailable optional coefficient data still leaves the declared model.
        line["mathematical_model"]["exponent_notation"] = "algorithm exponent p; upper bound on omega, not omega itself"
        try:
            line["scaling_projection"] = project_work(line["algorithm"], line["cutoff"],
                line["mathematical_model"], plans.get(line["algorithm"]), projection_cycles, size_policy=size_policy)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            line["scaling_projection"] = {"status": "unavailable", "reason": str(exc)}
    for size in sizes:
        left, right = inputs(size, seed)
        input_hash = sha256(repr((left, right)).encode("utf-8")).hexdigest()
        # This cache lasts only for this input pair. It retains measured evidence,
        # not result matrices or predictions substituted for executed counters.
        measured = {}
        for line in series:
            name, cutoff = line["algorithm"], line["cutoff"]
            observer = Instrumentation(profile=False)
            sample = {"size": size, "input_repr_sha256": input_hash, "prediction": {},
                      "measurement": {"kind": "setup_error"}}
            cache_key = None
            try:
                sample["prediction"] = (schedule_prediction(plans[name], size, cutoff, size_policy=size_policy) if name in plans
                                        else prediction(name, size, cutoff, costs[name], size_policy=size_policy))
                structure = (cutoff_execution_structure(name, implementations[name], sample["prediction"],
                                                       plans.get(name))
                             if reuse_equivalent_cutoffs and "unavailable_reason" not in costs[name] else None)
                # Attribution is independent of whether equivalent cutoffs are
                # reused. A replaced callback with matching counts must not be
                # credited with our planner's actual padding/leaf policy.
                known_policy = cutoff_execution_structure(name, implementations[name], sample['prediction'], plans.get(name)) is not None
                sample['size_effects'] = describe_sample(name, size, cutoff, sample['prediction'],
                                                        line['mathematical_model'], known_policy=known_policy)
                if structure is not None:
                    cache_key = (name, input_hash, json.dumps(structure, sort_keys=True))
                source = measured.get(cache_key) if cache_key is not None else None
                if source is not None:
                    if progress:
                        progress(f"Reusing {name}, n={size}, cutoff={cutoff} from cutoff={source['cutoff']}")
                    for key in ("status", "verification", "observation", "execution_measurements"):
                        sample[key] = deepcopy(source["sample"][key])
                    sample["measurement"] = {"kind": "reused", "source_cutoff": source["cutoff"],
                                             "equivalent_execution_structure": structure}
                else:
                    if progress:
                        progress(f"Counting {name}, n={size}, cutoff={cutoff}")
                    sample["measurement"] = {"kind": "executed", "source_cutoff": cutoff}
                    if structure is not None:
                        sample["measurement"]["equivalent_execution_structure"] = structure
                    measurement = ExecutionMeasurement(measure_memory=measure_memory)
                    try:
                        with measurement:
                            result = implementations[name](left, right, cutoff=cutoff, instrumentation=observer, size_policy=size_policy)
                    finally:
                        sample["execution_measurements"] = measurement.snapshot()
                    verification = check_product(left, right, result)
                    sample.update(status="passed" if verification.passed else "failed",
                                  verification={"passed": verification.passed, "mode": "exact equality",
                                                "maximum_absolute_error": (verification.max_absolute_error
                                                    if isfinite(verification.max_absolute_error) else str(verification.max_absolute_error)),
                                                "failures": verification.failures})
            except (ArithmeticError, ValueError, TypeError, OSError) as exc:
                sample.update(status="error", error=str(exc))
            if sample["measurement"]["kind"] != "reused":
                sample["observation"] = observer.snapshot()
            sample["prediction_matches_observed_data_multiplications"] = (
                sample["observation"]["counts"].get("field_multiplications", 0) == sample["prediction"]["padded_data_multiplications"]
                if sample["status"] == "passed" else None)
            sample["prediction_matches_observed_arithmetic_steps"] = (
                all(sample["observation"][metric] == sample["prediction"][f"padded_{metric}"]
                    for metric in ("field_arithmetic_operations", "atomic_arithmetic_operations"))
                if sample["status"] == "passed" and "unavailable_reason" not in costs[name] else None)
            # Category predictions come from the formula/descriptor, never by
            # allocating a second observed matrix execution or fitting a count.
            if sample["status"] == "passed" and line["scaling_projection"]["status"] == "projected":
                predicted_counts = category_budget(name, size, cutoff, plans.get(name), size_policy=size_policy)
                sample["category_prediction"] = predicted_counts
                sample["category_prediction_matches_observed"] = all(
                    sample["observation"]["counts"].get(key, 0) == predicted_counts[key] for key in CATEGORIES)
                if not sample["category_prediction_matches_observed"]:
                    sample.update(status="failed", error="separate arithmetic category budget mismatch")
            if (cache_key is not None and sample["measurement"]["kind"] == "executed"
                    and sample["status"] == "passed"
                    and sample["prediction_matches_observed_data_multiplications"] is True
                    and sample["prediction_matches_observed_arithmetic_steps"] is True):
                measured[cache_key] = {"cutoff": cutoff, "sample": sample}
            line["samples"].append(sample)
    for line in series:
        for index, sample in enumerate(line['samples']):
            sample.setdefault('size_effects', {'status': 'unavailable', 'reason': 'size planner failed before attribution'})
            if index:
                sample['size_effects']['transition_from_previous_sample'] = describe_transition(line['samples'][index-1], sample)
        try:
            if not any(s['size_effects'].get('status') == 'known_reference_policy'
                       and s['status'] == 'passed' for s in line['samples']):
                raise ValueError('no successful known reference traversal for this series')
            line['range_projection'] = project_range(line['algorithm'], line['cutoff'], sizes,
                plans.get(line['algorithm']), projection_cycles, size_policy=size_policy)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            line['range_projection'] = {'status': 'unavailable', 'reason': str(exc)}
        from .resource_bounds import bound_range
        try:
            line['resource_bounds'] = bound_range(line['algorithm'], line['cutoff'], line['samples'],
                line['range_projection'], plans.get(line['algorithm']),
                input_bound=INPUT_MAGNITUDE_BOUND, size_policy=size_policy)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            line['resource_bounds'] = {'status': 'unavailable', 'reason': str(exc)}
        minimum, maximum = sizes[0], sizes[-1]
        curve_sizes = sorted(set(sizes + [minimum + (maximum - minimum) * index / 80 for index in range(81)]))
        if "ideal_curve_status" in line["work_model"]:
            # Do not impose a homogeneous smooth model on a mixed-radix schedule.
            # Exact padded predictions still accompany every executed sample.
            curve_sizes = []
        line["ideal_curve"] = []
        for size in curve_sizes:
            predicted = prediction(line["algorithm"], size, line["cutoff"], line["work_model"], size_policy="padded")
            line["ideal_curve"].append({"size": size,
                "data_multiplications": predicted["ideal_data_multiplications"],
                "field_arithmetic_operations": predicted["ideal_field_arithmetic_operations"],
                "atomic_arithmetic_operations": predicted["ideal_atomic_arithmetic_operations"]})
        if line['range_projection']['status'] == 'projected' and curve_sizes:
            # The continuous comparison is secondary display evidence. Discrete
            # projected counts remain exact integers in a separate collection.
            ideal_extension = []
            for point in line['range_projection']['points']:
                if point['size'] <= maximum:
                    continue
                predicted = prediction(line['algorithm'], point['size'], line['cutoff'],
                                       line['work_model'], size_policy='padded')
                ideal_extension.append({'size': point['size'],
                    'data_multiplications': predicted['ideal_data_multiplications'],
                    'field_arithmetic_operations': predicted['ideal_field_arithmetic_operations'],
                    'atomic_arithmetic_operations': predicted['ideal_atomic_arithmetic_operations']})
            line['range_projection']['ideal_points'] = ideal_extension
        line["empirical_fits"] = {}
        for metric in ("field_multiplications", "field_arithmetic_operations", "atomic_arithmetic_operations"):
            points = [(sample["size"], sample["observation"]["counts"].get(metric, 0)
                       if metric == "field_multiplications" else sample["observation"][metric])
                      for sample in line["samples"] if sample["status"] == "passed"]
            line["empirical_fits"][metric] = log_log_fit(points)
    references, reference_executions = [], 0
    # A classical denominator is required even when the user selected only a
    # generated recipe. These extra reference calls are explicit and unmeasured.
    for size in sizes:
        if progress:
            progress(f"Accounting classical work baseline, n={size}")
        from .naive import multiply as classical_multiply
        source = next((sample for line in series if line['algorithm'] == 'naive'
                       and implementations['naive'] is classical_multiply
                       for sample in line['samples'] if sample['size'] == size and sample['status'] == 'passed'
                       and sample['prediction_matches_observed_data_multiplications'] is True
                       and sample['prediction_matches_observed_arithmetic_steps'] is True), None)
        if source is not None:
            reference = {'status': 'passed', 'input_repr_sha256': source['input_repr_sha256'],
                         'scope': 'reused verified classical sample; no additional execution',
                         'mathematical_work': deepcopy(source['observation']['mathematical_work']),
                         'data_movement': deepcopy(source['observation']['data_movement'])}
        else:
            left, right = inputs(size, seed)
            reference = classical_reference(left, right)
            reference_executions += 1
        references.append({'size': size, **reference})
        for line in series:
            sample = next(sample for sample in line['samples'] if sample['size'] == size)
            target = {'status': sample['status'], **sample['observation']}
            attach_comparison(target, reference, input_hash=sample['input_repr_sha256'])
            sample['work_comparison'] = target['work_comparison']
    samples = [sample for line in series for sample in line["samples"]]
    summary = {"requested_configurations": len(samples),
               **{label: sum(sample["measurement"]["kind"] == kind for sample in samples)
                  for label, kind in (("actual_executions", "executed"), ("reused_measurements", "reused"),
                                      ("setup_errors", "setup_error"))}}
    summary['classical_reference_executions'] = reference_executions
    return {"schema": SCHEMA, "execution_summary": summary, 'classical_work_references': references,
            "configuration": {"sizes": sizes, "cutoffs": cutoffs, "seed": seed,
            "input_absolute_bound": INPUT_MAGNITUDE_BOUND,
            "reuse_equivalent_cutoffs": reuse_equivalent_cutoffs, "size_policy": size_policy,
            "memory_tracking_enabled": measure_memory, "projection_cycles": projection_cycles,
            "scalar_domain": "signed Python integers; complex intermediates where schemes require them",
            "primitive_timing_enabled": False, "focus": "executed arithmetic work, not elapsed time"},
            "series": series}
