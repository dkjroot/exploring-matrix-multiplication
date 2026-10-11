"""Run all research report families, including measured arithmetic growth.

This command executes our algorithms and counts their arithmetic, rather than
estimating wall-clock complexity. The default suite also generates construction,
search/replay, CW recovery, exact certificates, and separate pair diagnostics.
All output shares one dated ignored directory. --measurements-only selects the
original count-only report. Research budgets are independent of matrix sizes.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

from matrix_multiplication import ALGORITHMS
from matrix_multiplication.work_sweep import collect_work
from matrix_multiplication.reporting import write_data_report
from matrix_multiplication.composition import Composition
from matrix_multiplication.research_suite import SuiteSettings, run_suite


def main(argv=None):
    """Collect a reproducible sweep and retain both JSON and its visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--algorithms", choices=list(ALGORITHMS), nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeat-equivalent-cutoffs", action="store_true",
                        help="execute every cutoff independently instead of reusing identical traversals")
    parser.add_argument("--composition", action="append", default=[],
                        help="add a repeating cycle of built-in names or JSON paths; repeatable")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--reference-report", type=Path, action="append", default=[],
                        help="include an earlier single-pair JSON report as separately labelled evidence; repeatable")
    parser.add_argument("--measurements-only", action="store_true", help="produce only the original arithmetic growth report")
    parser.add_argument("--search-size", type=int, default=2)
    parser.add_argument("--search-start", help="start from a verified real rational scheme JSON or maintained Strassen formula")
    search_target = parser.add_mutually_exclusive_group()
    search_target.add_argument("--search-target-rank", type=int)
    search_target.add_argument("--search-target-exponent", type=Fraction)
    parser.add_argument("--search-steps", type=int, default=1000, help="attempted flips per restart")
    parser.add_argument("--search-restarts", type=int, default=10)
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--search-max-height", type=int, default=1)
    parser.add_argument("--search-extra-rank", type=int, default=0)
    parser.add_argument("--search-split-every", type=int, default=50)
    parser.add_argument("--cw-parameters", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--cw-powers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--pair-trace-limit", type=int, default=200, help="retained scalar/primitive events per algorithm and diagnostic pair")
    args = parser.parse_args(argv)
    try:
        compositions = {f"composition-{i+1}": Composition.from_spec(spec)
                        for i, spec in enumerate(args.composition)}
        if not args.measurements_only:
            settings = SuiteSettings(search_size=args.search_size, search_start=args.search_start,
                                     search_target_rank=args.search_target_rank,
                                     search_target_exponent=args.search_target_exponent,
                                     search_steps=args.search_steps, search_restarts=args.search_restarts,
                                     search_seeds=tuple(args.search_seeds), search_max_height=args.search_max_height,
                                     search_extra_rank=args.search_extra_rank, search_split_every=args.search_split_every,
                                     cw_parameters=tuple(args.cw_parameters),
                                     cw_powers=tuple(args.cw_powers), trace_limit=args.pair_trace_limit)
            destination, status = run_suite(args.sizes, args.cutoffs, algorithms=args.algorithms,
                                           compositions=compositions, settings=settings, seed=args.seed,
                                           output_root=args.output_root, reference_reports=args.reference_report,
                                           progress=print_progress,
                                           reuse_equivalent_cutoffs=not args.repeat_equivalent_cutoffs)
            print(destination.resolve())
            return status
        data = collect_work(args.sizes, args.cutoffs, algorithms=args.algorithms,
                            seed=args.seed, progress=print_progress,
                            reuse_equivalent_cutoffs=not args.repeat_equivalent_cutoffs,
                            compositions=compositions)
        raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")
        destination = write_data_report(raw, source_name="work-sweep.json",
                                        output_root=args.output_root, title="Measured arithmetic growth",
                                        reference_reports=args.reference_report)
        print(destination.resolve())
    except (ValueError, TypeError, ArithmeticError, OSError) as exc:
        print(f"ERROR: cannot collect work sweep: {exc}", file=sys.stderr)
        return 1
    return int(any(sample["status"] != "passed" or not sample["prediction_matches_observed_data_multiplications"]
                   or sample["prediction_matches_observed_arithmetic_steps"] is False
                   for line in data["series"] for sample in line["samples"]))


def print_progress(message):
    """Flush progress before potentially large recursive experiments."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
