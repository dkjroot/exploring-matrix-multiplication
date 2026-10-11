"""Run reference experiments with independent correctness and optional observation.

Literal inputs are parsed as data. Each selected algorithm gets its own observer,
and verification runs after observation ends. JSON reports separate operation
counts/profiles from bounded scalar traces; timings include instrumentation cost.
Requesting a JSON report also generates its dated offline HTML visualization.
"""

import argparse
import ast
import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from pprint import pformat
import sys
import unittest

from matrix_multiplication import ALGORITHMS, DESCRIPTIONS, Instrumentation
from matrix_multiplication.common import validate
from matrix_multiplication.experiments import ExecutionMeasurement, arithmetic_model
from matrix_multiplication.verification import check_product
from matrix_multiplication.reporting import write_data_report
from matrix_multiplication.composition import Composition


def matrix_literal(text):
    """Parse a safe Python literal; convert parsing errors into CLI diagnostics."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError("expected a matrix literal, e.g. [[1,2],[3,4]]") from exc


def experiment_metadata(args, name):
    """Describe a reproducible run outside its measured arithmetic, without input copies."""
    metadata = {"algorithm": name, "cutoff": args.cutoff,
                "atol": args.atol if isfinite(args.atol) else str(args.atol),
                "rtol": args.rtol if isfinite(args.rtol) else str(args.rtol),
                "input_repr_sha256": sha256(repr((args.a, args.b)).encode("utf-8")).hexdigest()}
    try:
        rows, inner, columns = validate(args.a, args.b, args.cutoff)
        metadata["input_shapes"] = [[rows, inner], [inner, columns]]
        metadata["classical_data_product_baseline"] = rows * inner * columns
    except (ValueError, TypeError):
        metadata["input_shapes"] = None
    return metadata


def main(argv=None):
    """Run algorithms, validate results, optionally export measurements; return status."""
    parser = argparse.ArgumentParser(description=__doc__)
    registry = ALGORITHMS
    parser.add_argument("--algorithm", choices=["all", *registry], default="all")
    parser.add_argument("--composition", help="repeating comma-separated cycle of built-in names or scheme JSON paths; replaces --algorithm")
    parser.add_argument("--a", type=matrix_literal, default=[[1, 2], [3, 4]], help="left matrix literal")
    parser.add_argument("--b", type=matrix_literal, default=[[5, 6], [7, 8]], help="right matrix literal")
    parser.add_argument("--cutoff", type=int, default=1, help="classical leaf size; default 1 exercises the full recursion")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--list", action="store_true", help="list the implemented algorithms")
    parser.add_argument("--self-test", action="store_true", help="run the reproducible regression suite")
    parser.add_argument("--instrument", action="store_true", help="count and profile shared primitives")
    parser.add_argument("--counts-only", action="store_true", help="count primitives without per-primitive clocks")
    parser.add_argument("--measure-memory", action="store_true", help="measure peak traced Python allocations during each algorithm call")
    parser.add_argument("--report", type=Path, help="write JSON and automatically generate its dated HTML report")
    parser.add_argument("--output-root", type=Path, help="parent of automatic HTML reports; defaults to outputs/")
    parser.add_argument("--trace", type=Path, help="write a bounded primitive/scalar trace as JSON")
    parser.add_argument("--trace-limit", type=int, default=10000, help="retained events per algorithm; default 10000")
    parser.add_argument("--trace-values", action="store_true", help="include scalar operand/result representations")
    args = parser.parse_args(argv)
    if args.trace_limit < 0:
        parser.error("--trace-limit must be nonnegative")
    if args.list:
        for name, description in DESCRIPTIONS.items():
            print(f"{name}: {description}")
        print("OpenAI 2.25: an existence bound without a supplied finite base scheme; not implemented.")
        return 0
    if args.self_test:
        tests = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "tests"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
    implementations = ALGORITHMS if args.algorithm == "all" else {args.algorithm: registry[args.algorithm]}
    plan = None
    if args.composition is not None:
        if args.algorithm != "all":
            parser.error("--composition replaces --algorithm; specify only one")
        try:
            plan = Composition.from_spec(args.composition)
        except (ValueError, TypeError, OSError) as exc:
            print(f"ERROR: cannot load composition: {exc}", file=sys.stderr)
            return 1
        implementations = {"composition": plan.multiply}
    failed = False
    reports, traces = {}, {}
    for name, implementation in implementations.items():
        description = " -> ".join(s.name for s in plan.stages) + " (repeating cycle)" if plan else DESCRIPTIONS[name]
        print(f"{name}: {description}")
        observer = (Instrumentation(profile=not args.counts_only,
                                    trace=args.trace is not None, trace_limit=args.trace_limit,
                                    capture_values=args.trace_values)
                    if args.instrument or args.report or args.trace or args.counts_only or args.measure_memory else None)
        execution = ExecutionMeasurement(measure_memory=args.measure_memory)
        check = None
        try:
            with execution:
                result = implementation(args.a, args.b, cutoff=args.cutoff, instrumentation=observer)
            # The exact oracle is deliberately outside the measured algorithm.
            check = check_product(args.a, args.b, result, atol=args.atol, rtol=args.rtol)
        except (ValueError, TypeError, ArithmeticError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failed = True
            if observer is not None:
                reports[name] = {**observer.snapshot(), "status": "error", "error": str(exc)}
                reports[name]["execution_measurements"] = execution.snapshot()
                traces[name] = observer.events
            continue
        print(pformat(result))
        mode = "exact equality" if check.exact else "floating-point tolerance"
        print(f"{'PASS' if check.passed else 'FAIL'}: {mode}; maximum absolute error {check.max_absolute_error:g}")
        for failure in check.failures:
            print(f"  {failure}")
        failed |= not check.passed
        if observer is not None:
            reports[name] = {**observer.snapshot(), "status": "passed" if check.passed else "failed"}
            metadata = experiment_metadata(args, name)
            reports[name]["execution_measurements"] = execution.snapshot(
                metadata.get("classical_data_product_baseline"), check)
            traces[name] = observer.events
            counts = observer.counts
            print(f"WORK: {counts['field_multiplications']} data field products; "
                  f"{counts['coefficient_multiplications']} coefficient products; "
                  f"{reports[name]['atomic_arithmetic_operations']} atomic arithmetic operations")
            memory = reports[name]["execution_measurements"]["python_traced_peak_bytes"]
            print(f"SECONDARY: algorithm call {execution.elapsed_seconds:.6g} seconds; "
                  f"traced Python peak {memory['value']} bytes ({memory['status']})")
    # Files contain measurements, not copied inputs or potentially non-JSON results.
    for name, report in reports.items():
        report["experiment"] = experiment_metadata(args, name)
        report_plan = plan
        if name == "smirnov" and report["status"] != "error":
            from matrix_multiplication.smirnov import plan as smirnov_plan
            report_plan = smirnov_plan()
        report["mathematical_model"] = report_plan.model() if report_plan else arithmetic_model(name)
        if report_plan and report["experiment"]["input_shapes"]:
            shapes = report["experiment"]["input_shapes"]
            budget = report_plan.budget(shapes[0][0], shapes[0][1], shapes[1][1], cutoff=args.cutoff)
            report["composition_budget"] = budget
            report["composition_budget_matches"] = {
                "field_multiplications": report["counts"].get("field_multiplications", 0) == budget["field_multiplications"],
                **{key: report[key] == budget[key] for key in
                   ("field_arithmetic_operations", "atomic_arithmetic_operations")}}
            if report["status"] == "passed" and not all(report["composition_budget_matches"].values()):
                report["status"] = "failed"
                report["error"] = "composition counts disagree with recurrence budget"
                failed = True
        report["experiment"]["primitive_timing_enabled"] = not args.counts_only
    try:
        if args.report:
            raw = (json.dumps(reports, indent=2, allow_nan=False) + "\n").encode("utf-8")
            args.report.write_bytes(raw)
            destination = write_data_report(raw, source_name=args.report.name, output_root=args.output_root)
            print(f"HTML REPORT: {destination.resolve()}")
        if args.trace:
            args.trace.write_text(json.dumps({"reports": reports, "events": traces}, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8")
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"ERROR: cannot write experiment output: {exc}", file=sys.stderr)
        return 1
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
