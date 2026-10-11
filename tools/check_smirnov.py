"""Run a bounded Smirnov comparison, independently of sweep.py defaults.

The default uses only 2x2 and 4x4 matrices. An explicit --full-cycle exercises
one complete 54x54 square base, capturing the actual 64000 leaf products. JSON
and readable HTML are generated together under the ignored outputs directory.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from matrix_multiplication.work_sweep import collect_work
from matrix_multiplication.reporting import write_data_report


def main(argv=None):
    """Compare exact outputs and captured budgets at deliberately bounded sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cycle", action="store_true", help="explicitly execute one 54x54 base")
    args = parser.parse_args(argv)
    sizes = [54] if args.full_cycle else [2, 4]
    algorithms = ["naive", "smirnov"] if args.full_cycle else ["naive", "strassen", "alphaevolve", "smirnov"]
    try:
        data = collect_work(sizes, [1], algorithms=algorithms, progress=print_progress)
        if any(s["status"] != "passed" or not s["prediction_matches_observed_data_multiplications"]
               or not s["prediction_matches_observed_arithmetic_steps"]
               for line in data["series"] for s in line["samples"]):
            raise ValueError("an exact output or arithmetic budget failed")
        raw = (json.dumps(data, indent=2, allow_nan=False)+"\n").encode("utf-8")
        destination = write_data_report(raw, source_name="smirnov-check.json",
                                        title="Smirnov verified lower-exponent construction")
        print(f"PASS: exact results and executed arithmetic budgets; {destination.resolve()}")
    except (ValueError, ArithmeticError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


def print_progress(message):
    """Flush configuration progress before an explicitly requested full base."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
