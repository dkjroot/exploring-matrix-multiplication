"""Compose verified square schemes through the existing observable traversal.

A plan is a repeating cycle of stages, selected by recursion depth. A cycle's
radices and ranks multiply: it is a single larger base construction without
materializing its coefficient tensor. Readable built-in formulas retain their
shared intermediates; imported sparse schemes use the generic evaluator. Neither
composition nor observation adds data products to the declared recurrence.
"""

from dataclasses import dataclass
import json
from math import log, prod

from . import alphaevolve, strassen, strassen_winograd
from .common import padded_size, recursive_kernel, recursive_product, validate
from .scalars import Gaussian
from .scheme_io import coefficient_pair, load_scheme_document, scheme_from_document, scheme_document, verify_scheme
from .schemes import BilinearScheme, bilinear_step


@dataclass(frozen=True)
class Stage:
    """A formula's execution callback and explicit per-block arithmetic budget."""

    name: str
    radix: int
    rank: int
    step: object
    field_block_cost: int
    atomic_block_cost: int
    gaussian: bool = False
    scheme: object = None


def sparse_stage(scheme):
    """Verify a sparse recipe and count its literal scale/add/normalize schedule."""
    if scheme.dimensions is not None:
        raise ValueError("rectangular recipes need RectangularCycle; use the smirnov plan for its cyclic construction")
    verify_scheme(scheme)
    # Canonicalize exact real coefficients out of Gaussian wrappers. Otherwise
    # a zero imaginary component could silently trigger pair arithmetic in a
    # supposedly real plan, making its declared atomic budget inconsistent.
    factors = []
    for factor in (scheme.left, scheme.right, scheme.output):
        columns = []
        for column in factor:
            terms_in_column = []
            for index, value in column:
                real, imag = coefficient_pair(value)
                real = real.numerator if real.denominator == 1 else real
                imag = imag.numerator if imag.denominator == 1 else imag
                terms_in_column.append((index, Gaussian(real, imag) if imag else real))
            columns.append(tuple(terms_in_column))
        factors.append(tuple(columns))
    scheme = BilinearScheme(scheme.name, scheme.radix, *factors, divisor=scheme.divisor)
    terms = sum(len(column) for factor in (scheme.left, scheme.right, scheme.output)
                for column in factor)
    gaussian = any(coefficient_pair(value)[1] for factor in
                   (scheme.left, scheme.right, scheme.output)
                   for column in factor for _, value in column)
    normalization = scheme.radix ** 2 if scheme.divisor != 1 else 0

    def step(a, b, product):
        """Execute this verified descriptor via the shared sparse evaluator."""
        return bilinear_step(a, b, product, scheme)

    return Stage(scheme.name, scheme.radix, scheme.rank, step,
                 2 * terms + normalization,
                 (8 if gaussian else 2) * terms + (2 if gaussian else 1) * normalization,
                 bool(gaussian), scheme)


def builtin_stage(name):
    """Adapt maintained formulas, retaining Winograd's intermediate reuse."""
    if name == "strassen":
        return Stage(name, 2, 7, strassen._step, 18, 18)
    if name == "strassen-winograd":
        return Stage(name, 2, 7, strassen_winograd._step, 15, 15)
    if name == "alphaevolve":
        return sparse_stage(alphaevolve._scheme(alphaevolve._factors()))
    raise ValueError(f"unknown composition stage {name!r}")


class Composition:
    """A repeating schedule with the shared multiply interface.

    Pass built-in names or BilinearScheme descriptors. Validation and tensor
    checking happen at construction, outside multiplication counts. Plans use
    exact Gaussian pairs throughout if any stage needs complex coefficients;
    this keeps both the scalar domain and atomic cost model consistent.
    """

    def __init__(self, stages):
        """Freeze and validate the ordered cycle before any inputs are executed."""
        requested = tuple(stages)
        self.stages = tuple(sparse_stage(s) if isinstance(s, BilinearScheme)
                            else builtin_stage(s) for s in requested)
        if not self.stages:
            raise ValueError("composition needs at least one stage")
        self.radices = tuple(s.radix for s in self.stages)
        self.gaussian = any(s.gaussian for s in self.stages)
        self.documents = tuple(scheme_document(s.scheme) if s.scheme else None for s in self.stages)
        if "alphaevolve" in requested:
            data = json.loads(alphaevolve.DATA_PATH.read_text(encoding="utf-8"))
            provenance = {key: value for key, value in data.items() if key != "factors_doubled"}
            provenance["execution_transformation"] = "Doubled exact factors; source W transposed to row-major output; common divisor 8."
            for source, document in zip(requested, self.documents):
                if source == "alphaevolve" and document is not None:
                    document["provenance"] = provenance

    @classmethod
    def from_spec(cls, specification):
        """Parse comma-separated built-in names or paths to verified JSON schemes."""
        if specification.strip() == "smirnov":
            from .smirnov import plan
            return plan()
        stages = []
        documents = []
        for item in specification.split(","):
            item = item.strip()
            if not item:
                raise ValueError("composition stage names must not be empty")
            if item in ("strassen", "strassen-winograd", "alphaevolve"):
                stages.append(item)
                documents.append(None)
            else:
                document = load_scheme_document(item)
                stages.append(scheme_from_document(document))
                documents.append(document)
        plan = cls(stages)
        # Retain imported provenance/notices verbatim in the experiment JSON.
        plan.documents = tuple(d if d is not None else fallback
                               for d, fallback in zip(documents, plan.documents))
        return plan

    def model(self):
        """Derive the repeating-cycle recurrence; never infer an exponent from samples."""
        radix, rank = prod(self.radices), prod(s.rank for s in self.stages)
        return {"algorithm": "composition", "square_arithmetic_exponent": log(rank) / log(radix),
                "exponent_expression": f"log_{radix}({rank})",
                "base_dimension": radix, "base_bilinear_rank": rank,
                "parameter_source": "verified stage ranks; repeating cycle, not a measured fit",
                "recurrence": f"T(n) = {rank} T(n/{radix}) + O(n^2)",
                "arithmetic_domain": "exact Gaussian pairs" if self.gaussian else "real or complex field",
                "scope": "fixed repeating cycle and fixed classical cutoff",
                "schedule": [s.name for s in self.stages], "schedule_repeats": True,
                "identity_validation": "imported sparse identities checked completely; built-in formulas tested",
                "assumptions": ["unit-cost field arithmetic", "quadratic combination overhead",
                                "rank equals radix squared can add an n^2 log(n) term"],
                "scheme_documents": list(self.documents)}

    def budget(self, rows, inner, columns, *, cutoff=1, size_policy="fringe"):
        """Predict executed counts including core/fringe or baseline padding and leaves.

        This independent recurrence calculation never runs the multiplication.
        Gaussian pairs make each combination addition two component additions,
        each scale six component operations, and each leaf product six operations.
        The leaf accumulation adds two more, giving eight atomic steps per dot term.
        """
        from .size_policy import fringe_budget, validate_size_policy
        validate_size_policy(size_policy)
        if size_policy == "fringe":
            return fringe_budget(self.stages, (rows, inner, columns), cutoff, gaussian=self.gaussian)
        if type(cutoff) is not int or cutoff < 1:
            raise ValueError("cutoff must be a positive integer")
        if any(type(x) is not int or x < 1 for x in (rows, inner, columns)):
            raise ValueError("dimensions must be positive integers")
        size = padded_size(rows, inner, columns, self.radices[0], radices=self.radices)
        padded, nodes, depth, field, atomic = size, 1, 0, 0, 0
        levels = []
        while size > cutoff:
            stage = self.stages[depth % len(self.stages)]
            block_size = size // stage.radix
            # At this depth there are `nodes` identical-shape subproblems.
            # A block addition touches block_size squared entries, so multiply
            # the per-entry formula cost by that area and by the node count.
            # Data products are deferred to the leaf calculation below.
            field += nodes * block_size ** 2 * stage.field_block_cost
            # Even real-coefficient stages act on pairs in a complex plan.
            if self.gaussian:
                if stage.scheme is None:
                    cost = 2 * stage.field_block_cost
                else:
                    terms = sum(len(c) for factor in (stage.scheme.left, stage.scheme.right,
                                                      stage.scheme.output) for c in factor)
                    cost = 8 * terms + (2 * stage.radix ** 2 if stage.scheme.divisor != 1 else 0)
            else:
                cost = stage.atomic_block_cost
            atomic += nodes * block_size ** 2 * cost
            levels.append({"depth": depth, "stage": stage.name, "nodes": nodes,
                           "size": size, "rank": stage.rank, "radix": stage.radix})
            nodes *= stage.rank
            size, depth = block_size, depth + 1
        # Every remaining leaf is classical: size^2 dot products, each of length
        # size. Each data product is accumulated with one additional field add.
        # Gaussian pairs expose four component multiplies plus two sums per
        # product, and two more sums for accumulation: eight atomic operations.
        products = nodes * size ** 3
        return {"padded_size": padded, "leaf_size": size, "classical_leaves": nodes,
                "levels": levels, "field_multiplications": products,
                "field_arithmetic_operations": field + 2 * products,
                "atomic_arithmetic_operations": atomic + (8 if self.gaussian else 2) * products,
                "atomic_domain": "Gaussian components" if self.gaussian else "real scalars; native complex is opaque"}

    def multiply(self, a, b, *, cutoff=1, instrumentation=None, size_policy="fringe"):
        """Execute the cycle through shared size handling, recursion, and scalar primitives."""
        validate(a, b, cutoff)
        complex_input = any(isinstance(x, complex) for m in (a, b) for row in m for x in row)

        def select(depth):
            """Choose this depth's formula without altering its child product count."""
            stage = self.stages[depth % len(self.stages)]
            return stage.radix, stage.step

        def kernel(left, right, leaf_size):
            """Bind the cycle to the same recursive kernel used by fixed algorithms."""
            return recursive_kernel(left, right, leaf_size, self.radices[0],
                                    self.stages[0].step, name="composition", select_step=select, size_policy=size_policy)

        def decode(value):
            """Restore exact real results or native complex floating semantics."""
            value = Gaussian.from_number(value)
            if complex_input:
                return complex(value.real, value.imag)
            from numbers import Rational
            if isinstance(value.imag, Rational) and value.imag != 0:
                raise ArithmeticError("unexpected imaginary component for exact real inputs")
            return value.real

        return recursive_product(a, b, cutoff, self.radices[0], kernel,
                                 name="composition", instrumentation=instrumentation,
                                 radices=self.radices,
                                 encode=Gaussian.from_number if self.gaussian else None,
                                 decode=decode if self.gaussian else None, size_policy=size_policy)
