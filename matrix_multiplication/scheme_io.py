"""Exact, data-only interchange for square and rectangular bilinear schemes.

Version 1 stores a square radix; version 2 stores (rows, inner, columns).
Both use row-major block indices in all three factors. Every product has
one sparse column in left, right, and output. Coefficients are integers or exact
Gaussian rationals, never floating approximations. Loading checks the complete
tensor identity before a recipe can enter the composition engine. Verification
uses its own pair arithmetic, outside the observed execution primitives.
"""

from fractions import Fraction
import json
from math import isfinite
from pathlib import Path

from .scalars import Gaussian
from .schemes import BilinearScheme


def coefficient_pair(value):
    """Return exact rational components, rejecting approximate coefficient data."""
    if isinstance(value, Gaussian):
        parts = (value.real, value.imag)
    else:
        parts = (value, 0)
    if any(type(x) not in (int, Fraction) for x in parts):
        raise ValueError("scheme coefficients must be exact integers or Fractions")
    return tuple(Fraction(x) for x in parts)


def pair_product(a, b):
    """Multiply verification-only rational pairs independently of instrumentation."""
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def verify_scheme(scheme):
    """Check every coefficient of U*V*W against the square product tensor.

    Sparse expansion stores only reached tensor entries, including cancellations.
    Expected nonzeros are checked separately, so a missing output is detected.
    This is a complete finite identity check, not random numerical sampling.
    """
    tensor = {}
    for left, right, output in zip(scheme.left, scheme.right, scheme.output):
        for factor in (left, right, output):
            indices = [index for index, _ in factor]
            if len(indices) != len(set(indices)):
                raise ValueError("duplicate block indices in a sparse column")
            for _, value in factor:
                coefficient_pair(value)
        for i, u in left:
            for j, v in right:
                uv = pair_product(coefficient_pair(u), coefficient_pair(v))
                for k, w in output:
                    real, imag = pair_product(uv, coefficient_pair(w))
                    old = tensor.get((i, j, k), (0, 0))
                    tensor[i, j, k] = old[0] + real, old[1] + imag
    m, k, n = scheme.shape
    expected = {(i*k+j, j*n+l, i*n+l) for i in range(m)
                for j in range(k) for l in range(n)}
    for key in set(tensor) | expected:
        target = (scheme.divisor if key in expected else 0, 0)
        if tensor.get(key, (0, 0)) != target:
            raise ValueError(f"incorrect tensor coefficient at {key}")
    return scheme


def decode_coefficient(value):
    """Parse an integer or explicit real/imaginary numerator-denominator pairs."""
    if type(value) is int:
        return value
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise ValueError("coefficient must be an integer or real/imag rational object")
    components = []
    for name in ("real", "imag"):
        pair = value[name]
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int for x in pair) or pair[1] <= 0):
            raise ValueError("rational components need integer numerator and positive denominator")
        components.append(Fraction(*pair))
    return Gaussian(*components) if components[1] else components[0]


def scheme_document(scheme, *, provenance=None):
    """Export a verified canonical recipe, with optional source/license metadata."""
    verify_scheme(scheme)

    def encode(value):
        """Keep integers readable and serialize other exact values losslessly."""
        real, imag = coefficient_pair(value)
        if not imag and real.denominator == 1:
            return real.numerator
        return {"real": [real.numerator, real.denominator],
                "imag": [imag.numerator, imag.denominator]}

    shape_fields = ({"schema_version": 2, "dimensions": list(scheme.shape)}
                    if scheme.dimensions else {"schema_version": 1, "radix": scheme.radix})
    return {**shape_fields, "name": scheme.name,
            "divisor": scheme.divisor,
            **{name: [[[index, encode(value)] for index, value in column]
                       for column in getattr(scheme, name)]
               for name in ("left", "right", "output")},
            "provenance": provenance or {}}


def scheme_from_document(document):
    """Validate the versioned format and return a fully identity-checked recipe."""
    rectangular = isinstance(document, dict) and document.get("schema_version") == 2
    required = {"schema_version", "name", "dimensions" if rectangular else "radix", "left", "right", "output"}
    if (not isinstance(document, dict) or not required <= set(document)
            or set(document) - required - {"divisor", "provenance"}
            or type(document["schema_version"]) is not int or document["schema_version"] not in (1, 2)):
        raise ValueError("expected a version-1 or version-2 sparse scheme document")
    if not isinstance(document.get("provenance", {}), dict):
        raise ValueError("provenance must be an object")
    factors = []
    for name in ("left", "right", "output"):
        factor = document[name]
        if not isinstance(factor, list) or not factor:
            raise ValueError("each factor must be a nonempty list of sparse columns")
        columns = []
        for column in factor:
            if not isinstance(column, list):
                raise ValueError("sparse columns must be lists")
            terms = []
            for term in column:
                if not isinstance(term, list) or len(term) != 2:
                    raise ValueError("sparse terms must contain index and coefficient")
                terms.append((term[0], decode_coefficient(term[1])))
            columns.append(tuple(terms))
        factors.append(tuple(columns))
    dimensions = document.get("dimensions")
    if rectangular and (not isinstance(dimensions, list) or len(dimensions) != 3):
        raise ValueError("dimensions must be a three-element array")
    return verify_scheme(BilinearScheme(document["name"], document.get("radix"),
                                        *factors, divisor=document.get("divisor", 1),
                                        dimensions=tuple(dimensions) if rectangular else None))


def load_scheme_document(path):
    """Read and verify one unambiguous JSON recipe, retaining its source notices.

    Python's default JSON reader silently keeps the last duplicate object key.
    A proof artifact must not have two competing meanings for its rank, factors,
    or attribution. Reject duplicate keys at every nesting depth and nonstandard
    NaN/Infinity constants. Coefficient parsing separately rejects finite floats.
    Callers receive the same document that was checked, avoiding a second file
    read that could bind provenance to a different version of the recipe.
    """
    def unique_object(pairs):
        """Build one JSON object, failing before a duplicate can overwrite data."""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        """Reject JSON extensions that cannot belong to reproducible exact data."""
        raise ValueError(f"nonfinite JSON constant: {value}")

    def finite_float(value):
        """Reject finite-looking literals whose exponent overflows Python's float."""
        parsed = float(value)
        if not isfinite(parsed):
            raise ValueError(f"nonfinite JSON number: {value}")
        return parsed

    document = json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object, parse_constant=reject_constant,
                          parse_float=finite_float)
    scheme_from_document(document)
    return document


def load_scheme(path):
    """Load a fully checked JSON identity; no imported source code executes."""
    return scheme_from_document(load_scheme_document(path))


def save_scheme(scheme, path, *, provenance=None):
    """Save a verified scheme for later import, preserving caller-supplied notices."""
    document = scheme_document(scheme, provenance=provenance)
    Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
