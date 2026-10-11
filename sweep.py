"""Measure arithmetic growth over matrix sizes and leaf blocks, then render HTML.

This command executes our algorithms and counts their arithmetic, rather than
estimating wall-clock complexity. Theoretical predictions and measured finite-
range slopes are clearly separated. Output goes to a dated ignored directory.
"""

import argparse
import json
from pathlib import Path
import sys

from matrix_multiplication import ALGORITHMS
from matrix_multiplication.work_sweep import collect_work
from matrix_multiplication.reporting import write_data_report


def main(argv=None):
    """Collect a reproducible sweep and retain both JSON and its visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--algorithms", choices=list(ALGORITHMS), nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--reference-report", type=Path, action="append", default=[],
                        help="include an earlier single-pair JSON report as separately labelled evidence; repeatable")
    args = parser.parse_args(argv)
    try:
        data = collect_work(args.sizes, args.cutoffs, algorithms=args.algorithms,
                            seed=args.seed, progress=print_progress)
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
