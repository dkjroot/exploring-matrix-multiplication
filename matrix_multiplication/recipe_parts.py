"""Reload exact rational matrix parts or independent sums, including unit axes.

Parts are generic exact tensors with proved matrix shapes, source maps and
construction provenance. They can be useful before they qualify as recursive
matrix schemes. Loading rechecks the complete identity; a saved success flag
is never trusted. A direct-sum part lists component_shapes and must pass the
whole independent identity, including all cross coefficients. Tensor products
of ordinary matrix parts use separate copies of the source spaces,
not an unproved independence assumption about overlapping restrictions.
"""

from fractions import Fraction
import json
from pathlib import Path

from .tensor_restrictions import RestrictionWitness, product_witness, verify_restriction
from .tensors import Tensor, classical, direct_sum, tensor_product


def identity_witness(tensor, shape):
    """Describe a part already in matrix order without asserting its identity."""
    return RestrictionWitness(tensor.dimensions, shape,
                              tuple(tuple(range(d)) for d in tensor.dimensions))


def recombine_parts(left, left_shape, right, right_shape):
    """Tensor verified executable parts and explicitly reindex rows and columns.

    Unit axes remain valid: an outer-product part and a dot-product part may
    combine into a shape with all axes at least two. Actual recovered terms are
    retained; no classical specification replaces the proposed recipe's cost.
    """
    first, second = identity_witness(left, left_shape), identity_witness(right, right_shape)
    verify_restriction(left, first)
    verify_restriction(right, second)
    witness = product_witness(first, second)
    paired = tensor_product(left, right)
    return verify_restriction(paired, witness), witness.shape


def tensor_from_document(document):
    """Decode exact rational terms, rejecting floats, booleans and bad indices."""
    if not isinstance(document, dict) or document.get("schema") != "rational-tensor-v1":
        raise ValueError("unsupported rational tensor document")
    dimensions, terms = document.get("dimensions"), document.get("terms")
    if not isinstance(dimensions, list) or not isinstance(terms, list):
        raise ValueError("tensor dimensions and terms must be arrays")
    decoded = []
    for term in terms:
        if not isinstance(term, list) or len(term) != 3:
            raise ValueError("tensor terms require three array factors")
        factors = []
        for factor in term:
            if not isinstance(factor, list):
                raise ValueError("tensor factors must be arrays")
            entries = []
            for entry in factor:
                if (not isinstance(entry, list) or len(entry) != 2
                        or not isinstance(entry[1], list) or len(entry[1]) != 2):
                    raise ValueError("tensor entries need coordinate and rational numerator/denominator")
                index, (numerator, denominator) = entry
                if type(index) is not int or any(type(n) is not int for n in (numerator, denominator)) or denominator <= 0:
                    raise ValueError("tensor coordinates and rational coefficients must be exact integers with positive denominator")
                entries.append((index, Fraction(numerator, denominator)))
            factors.append(tuple(entries))
        decoded.append(tuple(factors))
    return Tensor(tuple(dimensions), tuple(decoded))


def unique_object(pairs):
    """Reject duplicate keys rather than let one evidence field hide another."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate recipe-parts key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    """Reject nonfinite JSON extensions at the artifact boundary."""
    raise ValueError(f"nonfinite recipe-parts value: {value}")


def load_parts(path):
    """Return checked (tensor, evidence) pairs and inherited library provenance.

    Each tensor must implement its stated shape or independent component shapes
    on every coefficient. Source
    maps are retained as derivation evidence, not certified by a saved flag:
    replaying a derivation additionally needs its exact source tensor. This
    loader certifies the executable part identity, not the entire derivation.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    if (not isinstance(document, dict) or document.get("schema") != "exact-recipe-parts-v1"
            or not isinstance(document.get("parts"), list) or not isinstance(document.get("provenance"), dict)):
        raise ValueError("unsupported exact recipe-parts library")
    parts, names = [], set()
    for item in document["parts"]:
        if not isinstance(item, dict) or not isinstance(item.get("evidence"), dict):
            raise ValueError("recipe part requires evidence and an exact tensor")
        evidence = item["evidence"]
        name, shape, shapes = evidence.get("name"), evidence.get("shape"), evidence.get("component_shapes")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("recipe parts need unique names and matrix shapes")
        tensor = tensor_from_document(item.get("tensor"))
        if shape is not None:
            if not isinstance(shape, list) or shapes is not None and shapes != [shape]:
                raise ValueError("recipe part has ambiguous matrix/component shapes")
            verify_restriction(tensor, identity_witness(tensor, shape))
        else:
            if (not isinstance(shapes, list) or len(shapes) < 2
                    or any(not isinstance(s, list) or len(s) != 3 for s in shapes)):
                raise ValueError("recipe parts need a matrix shape or independent component shapes")
            expected = classical(shapes[0])
            for component_shape in shapes[1:]:
                expected = direct_sum(expected, classical(component_shape))
            if tensor.dimensions != expected.dimensions or tensor.expand() != expected.expand():
                raise ValueError("recipe part fails the complete independent-sum identity")
        if type(evidence.get("products")) is not int or evidence["products"] != tensor.rank:
            raise ValueError("recipe-part product count disagrees with its actual decomposition")
        parts.append((tensor, evidence))
        names.add(name)
    return tuple(parts), document["provenance"]
