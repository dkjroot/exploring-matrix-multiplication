"""Import pinned Smirnov coefficients as data, never execute upstream code.

The BSD source and license remain in ignored Background Research. The converter
checks all coefficients of the exact rectangular identity before writing JSON.
Our execution engine is independent of the upstream C++ and code generator.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
from itertools import product
from pathlib import Path
import re
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from matrix_multiplication.schemes import BilinearScheme
from matrix_multiplication.scheme_io import save_scheme, verify_scheme

REVISION = "132358be7a6ab57c83dd5ce70ece460bd1ec5dfa"
BASE = f"https://raw.githubusercontent.com/arbenson/fast-matmul/{REVISION}"
SOURCE_URL = BASE + "/codegen/algorithms/smirnov336-40-960"
SOURCE_SHA256 = "41724202b7ef0dc9f11665be6e7b158e55d080127e3bb6611ce61d6773c9bcb5"
DATA = ROOT / "Background Research"


def convert(raw):
    """Read the three exact factor tables and resolve ordering by full verification."""
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("coefficient file differs from the pinned source fingerprint")
    groups, current = [], []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if current:
                groups.append(current)
                current = []
            continue
        tokens = line.split()
        if len(tokens) != 40 or any(not re.fullmatch(r"[+-]?\d+(?:/\d+)?", t) for t in tokens):
            raise ValueError("unexpected coefficient table encoding")
        current.append([Fraction(t) for t in tokens])
    if current:
        groups.append(current)
    if [len(g) for g in groups] != [9, 18, 18]:
        raise ValueError("expected factor tables with 9, 18, 18 rows")
    # Source conventions must never be guessed without checking the identity.
    for transposes in product((False, True), repeat=3):
        factors = []
        for table, (rows, cols), transpose in zip(groups, ((3, 3), (3, 6), (3, 6)), transposes):
            columns = []
            for term in range(40):
                columns.append(tuple(((i % rows)*cols+i//rows if transpose else i, row[term])
                                     for i, row in enumerate(table) if row[term]))
            factors.append(tuple(columns))
        scheme = BilinearScheme("smirnov-336", None, *factors, dimensions=(3, 3, 6))
        try:
            verify_scheme(scheme)
        except ValueError:
            continue
        return scheme, transposes
    raise ValueError("no supported row/column convention satisfies the exact identity")


def main(argv=None):
    """Download or read pinned tables, verify, and retain their attribution/notices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        DATA.mkdir(exist_ok=True)
        source = args.source or DATA / "smirnov336-40-960.txt"
        if args.download:
            source.write_bytes(urlopen(SOURCE_URL, timeout=30).read())
            (DATA / "fast-matmul-LICENSE.txt").write_bytes(urlopen(BASE + "/LICENSE", timeout=30).read())
        raw = source.read_bytes()
        scheme, transposes = convert(raw)
        license_path = DATA / "fast-matmul-LICENSE.txt"
        license_text = license_path.read_text(encoding="utf-8")
        provenance = {"source_url": SOURCE_URL, "source_revision": REVISION,
                      "source_sha256": SOURCE_SHA256,
                      "author": "Alexey V. Smirnov; coefficient encoding by Austin R. Benson and Grey Ballard",
                      "license": "BSD-2-Clause", "license_url": BASE + "/LICENSE",
                      "license_notice": license_text,
                      "conversion": f"Exact Fraction tables; transpose flags {transposes}; row-major output"}
        destination = DATA / "smirnov-336.json"
        save_scheme(scheme, destination, provenance=provenance)
        print(f"Verified rank {scheme.rank}, dimensions {scheme.shape}: {destination}")
    except (ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
