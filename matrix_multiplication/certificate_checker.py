"""Standalone exact certificate checker, independent of the algorithm libraries.

This file can be copied beside certificate.json and run with ordinary Python.
It imports only the standard library. Dense basis evaluation deliberately differs
from the production verifier's sparse tensor expansion. Stored pass flags are
never trusted: every coefficient is recomputed with exact Fraction arithmetic.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from math import prod
from pathlib import Path


def fingerprint(document):
    """Hash canonical JSON, including provenance; formatting does not affect it."""
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def exact_pair(value):
    """Decode coefficients independently, admitting no floating approximation."""
    if type(value) is int:
        return Fraction(value), Fraction(0)
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise ValueError("invalid exact coefficient")
    components = []
    for key in ("real", "imag"):
        pair = value[key]
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int for x in pair) or pair[1] <= 0):
            raise ValueError("invalid rational coefficient")
        components.append(Fraction(pair[0], pair[1]))
    return tuple(components)


def multiply_pair(a, b):
    """Use exact component arithmetic without the library's scalar primitives."""
    return (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])


def check_recipe(document):
    """Evaluate every pair of matrix units and compare every output entry.

    E_ij has one at row i, column j and zero elsewhere. Their product is E_il
    when the inner indices match and zero otherwise. Bilinearity makes these
    basis pairs sufficient for the entire identity over the declared exact field.
    """
    rectangular = isinstance(document, dict) and document.get("schema_version") == 2
    required = {"schema_version", "name", "dimensions" if rectangular else "radix", "left", "right", "output"}
    if (not isinstance(document, dict) or not required <= set(document)
            or set(document) - required - {"divisor", "provenance"}
            or type(document["schema_version"]) is not int or document["schema_version"] not in (1, 2)):
        raise ValueError("invalid recipe schema")
    d, divisor = document.get("radix"), document.get("divisor", 1)
    dimensions = document.get("dimensions") if rectangular else [d, d, d]
    if (not isinstance(dimensions, list) or len(dimensions) != 3
            or any(type(x) is not int or x < 2 for x in dimensions)
            or type(divisor) is not int or divisor < 1):
        raise ValueError("invalid radix or divisor")
    m, inner, n = dimensions
    if not isinstance(document["name"], str) or not document["name"]:
        raise ValueError("invalid recipe name")
    if not isinstance(document.get("provenance", {}), dict):
        raise ValueError("invalid provenance")
    factors, rank = [], None
    for name, bound in zip(("left", "right", "output"), (m*inner, inner*n, m*n)):
        source = document[name]
        if not isinstance(source, list) or not source:
            raise ValueError("empty or malformed factor")
        if rank is not None and len(source) != rank:
            raise ValueError("factor ranks differ")
        rank = len(source)
        dense = []
        for column in source:
            if not isinstance(column, list):
                raise ValueError("invalid sparse column")
            entries = [(Fraction(0), Fraction(0)) for _ in range(bound)]
            seen = set()
            for term in column:
                if (not isinstance(term, list) or len(term) != 2
                        or type(term[0]) is not int or not 0 <= term[0] < bound
                        or term[0] in seen):
                    raise ValueError("invalid or duplicate block index")
                seen.add(term[0])
                entries[term[0]] = exact_pair(term[1])
            dense.append(entries)
        factors.append(dense)
    u, v, w = factors
    for left in range(m*inner):
        i, j = divmod(left, inner)
        for right in range(inner*n):
            k, l = divmod(right, n)
            products = [multiply_pair(u[t][left], v[t][right]) for t in range(rank)]
            for output in range(m*n):
                real, imag = Fraction(0), Fraction(0)
                for t in range(rank):
                    contribution = multiply_pair(products[t], w[t][output])
                    real += contribution[0]
                    imag += contribution[1]
                target = divisor if j == k and output == i*n+l else 0
                if real != target or imag != 0:
                    raise ValueError(f"basis identity failed at left={left}, right={right}, output={output}")
    return {**({"dimensions": dimensions} if rectangular else {"radix": d}),
            "rank": rank, "coefficient_checks": (m*inner*n)**2,
            "basis_pairs": m*inner*inner*n, "identity": "exact", "recipe_sha256": fingerprint(document)}


def check_certificate(certificate):
    """Recheck recipes, binding, and the ordered cycle's exact integer recurrence.

    Composition relies on the documented induction argument, not a materialized
    giant tensor. No claim here certifies the execution interpreter or floating
    numerical stability. The exponent is specified exactly as log_radix(rank).
    """
    if (not isinstance(certificate, dict) or set(certificate) !=
            {"schema", "field", "conventions", "stages", "cycle"}
            or certificate["schema"] != "matrix-multiplication-certificate-v1"
            or certificate["field"] != "Q(i)"
            or certificate["conventions"] != "row-major; output divided by recipe divisor"):
        raise ValueError("invalid certificate schema or conventions")
    stages = certificate["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("certificate needs a nonempty ordered stage cycle")
    results = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"recipe", "recipe_sha256", "evidence"}:
            raise ValueError("invalid stage record")
        if fingerprint(stage["recipe"]) != stage["recipe_sha256"]:
            raise ValueError("recipe fingerprint mismatch")
        result = check_recipe(stage["recipe"])
        if stage["evidence"] != result:
            raise ValueError("stored evidence disagrees with independent recomputation")
        results.append(result)
    shapes = [r.get("dimensions", [r.get("radix")]*3) for r in results]
    totals = [prod(s[axis] for s in shapes) for axis in range(3)]
    if len(set(totals)) != 1:
        raise ValueError("cycle dimensions do not form a square construction")
    radix, rank = totals[0], prod(r["rank"] for r in results)
    expected = {"radix": radix, "rank": rank, "repeats": True,
                "exponent_expression": f"log_{radix}({rank})",
                "recurrence": f"T(n) = {rank} T(n/{radix}) + O(n^2)"}
    cycle = certificate["cycle"]
    if (not isinstance(cycle, dict) or cycle != expected
            or type(cycle.get("radix")) is not int or type(cycle.get("rank")) is not int
            or cycle.get("repeats") is not True):
        raise ValueError("cycle declaration disagrees with ordered verified stages")
    return {"certificate_sha256": fingerprint(certificate), "stages": results, "cycle": expected}


def read_json(path):
    """Read strict JSON data, rejecting duplicate keys and nonfinite constants."""
    def object_pairs(pairs):
        """Prevent ambiguous repeated keys in a supposedly exact certificate."""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        """Reject NaN/infinity tokens admitted by Python's permissive defaults."""
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def main(argv=None):
    """Check a portable package with only Python; return nonzero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path, nargs="?", default=Path(__file__).with_name("certificate.json"))
    args = parser.parse_args(argv)
    try:
        result = check_certificate(read_json(args.certificate))
    except (ValueError, TypeError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    print("PASS: all exact basis identities and cycle declarations rechecked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
