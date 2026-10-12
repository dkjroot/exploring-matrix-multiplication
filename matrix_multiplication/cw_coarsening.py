"""Merge CW group words, classify their tensors, and pack proved matrix pieces.

The fine partition keeps a separate space for every word of groups 0,1,2.
Here words with the same SUM of group indices share a coarse space. At power p,
its weights run from 0 to 2p, and supported sector weights sum to 2p. Merging
can expose larger components. It can also create tensors that are not ordinary
matrix products: a larger space is not automatically a larger multiplication.

This finite reference layer implements the partition and exact identities, a
conservative basis-permutation classifier, and induced selection of recognized
matrix components. Their polynomial recovery and dense packing use existing
primitives with all finite costs paid. Unrecognized components retain their
complete sparse coefficient support for further extraction research. Nothing
here implements recursive laser analysis or the asymptotic sum inequality.

Reference: Alman et al., More Asymmetry Yields Faster Matrix Multiplication,
sections 2.2 and 3.7, https://arxiv.org/abs/2404.16349. For powers of two this is
the paper's sum-index coarsening; the same finite partition also makes sense for
other positive integer powers. All code below is independently authored.
"""

from dataclasses import dataclass
from itertools import product
from math import prod

from .cw_extraction import (digits, execution_evidence, optimize_recovery,
                            restrict_polynomial, validate_parameters)
from .degenerations import cw_family, polynomial_degree_bound
from .tensor_classification import classify_matrix_tensor
from .tensors import (classical, coppersmith_winograd, direct_sum,
                      polynomial_coefficient, restrict_coordinates, tensor_power)


def weight_groups(q, power):
    """Partition all flattened source coordinates by summed CW group index.

    Digit zero belongs to group zero, digits 1..q to group one, and digit q+1
    to group two. The sum forgets their order, deliberately merging fine words.
    Each coordinate occurs exactly once; its original flattening is retained.
    """
    validate_parameters(q, power)
    groups = [[] for _ in range(2*power+1)]
    for coordinate in range((q+2)**power):
        word = digits(coordinate, (q+2,)*power)
        weight = sum(0 if digit == 0 else 2 if digit == q+1 else 1 for digit in word)
        groups[weight].append(coordinate)
    return tuple(tuple(group) for group in groups)


@dataclass(frozen=True)
class CoarseComponent:
    """A coarse sector, its original maps, and its proved layout when recognized."""

    sector: tuple
    selections: tuple
    tensor: object
    layout: object
    classification: dict

    @property
    def shape(self):
        """Expose matrix dimensions only after successful complete classification."""
        if self.layout is None:
            raise ValueError("unrecognized coarse components have no proved matrix shape")
        return self.layout.shape

    @property
    def matrix_selections(self):
        """Compose local basis permutations with the original CW source maps."""
        if self.layout is None:
            raise ValueError("unrecognized coarse components have no proved matrix coordinate order")
        return tuple(tuple(self.selections[leg][i] for i in selected)
                     for leg, selected in enumerate(self.layout.selections))

    def document(self):
        """Keep every coefficient and classification witness, including misses.

        Coefficients describe the tensor in its local sorted source basis, not
        the matrix permutation. They are exact numerator/denominator pairs and
        allow reconstruction of unrecognized tensors without an upstream file.
        This support representation is evidence, never a replacement for the
        polynomial recipe used by the executable extraction below.
        """
        support = self.tensor.expand()
        result = {"sector_weights": list(self.sector),
                  "source_coordinates": [list(s) for s in self.selections],
                  "dimensions": list(self.tensor.dimensions),
                  "classification": self.classification,
                  "support_coefficients": [[*indices, [c.numerator, c.denominator]]
                                           for indices, c in sorted(support.items())]}
        if self.layout:
            result.update(shape=list(self.shape), classical_products=prod(self.shape),
                          source_coordinates_in_matrix_order=[list(s) for s in self.matrix_selections])
        return result


def coarse_catalog(q, power):
    """Build every coarse component and independently reconstruct the full power.

    Components in this catalog share spaces and are NOT all independent matrix
    products. Recombination embeds each local support into its source coordinates
    and adds its coefficients, then compares the complete original CW tensor.
    This catches missing sectors, repeated sectors and bad coordinate maps.
    """
    groups = weight_groups(q, power)
    source = tensor_power(coppersmith_winograd(q), power)
    components, recombined = [], {}
    for x in range(2*power+1):
        for y in range(2*power-x+1):
            sector = (x, y, 2*power-x-y)
            selections = tuple(groups[weight] for weight in sector)
            tensor = restrict_coordinates(source, selections)
            layout, evidence = classify_matrix_tensor(tensor)
            item = CoarseComponent(sector, selections, tensor, layout, evidence)
            components.append(item)
            for indices, coefficient in tensor.expand().items():
                source_indices = tuple(selections[leg][i] for leg, i in enumerate(indices))
                recombined[source_indices] = recombined.get(source_indices, 0)+coefficient
    if recombined != source.expand():
        raise AssertionError("coarse CW partition failed complete source tensor reconstruction")
    return tuple(components), groups


def coarse_conflicts(sectors, power):
    """Reject shared coarse spaces and supported unintended cross-components.

    All supported aggregate triples sum to 2p. Disjoint weights alone do not
    prevent a cross-component joining X, Y and Z from different chosen slots.
    A final complete identity check still independently validates selection.
    """
    validate_parameters(1, power)
    sectors = tuple(tuple(sector) for sector in sectors)
    if not sectors or any(len(s) != 3 or any(type(i) is not int or i < 0 for i in s)
                          or sum(s) != 2*power for s in sectors):
        raise ValueError("coarse sectors need nonnegative integer weights summing to twice the power")
    weights = tuple(tuple(s[leg] for s in sectors) for leg in range(3))
    shared = [leg for leg, values in enumerate(weights) if len(set(values)) != len(values)]
    intended = set(sectors)
    unexpected = [candidate for candidate in product(*(sorted(set(w)) for w in weights))
                  if sum(candidate) == 2*power and candidate not in intended]
    return {"shared_legs": shared, "unexpected_sectors": unexpected}


def choose_coarse_components(catalog, power):
    """Greedily select only proved matrix products forming an induced sum.

    Prefer volume, then balanced shape, then sector for reproducibility. Retain
    all rejected/unrecognized components in the catalog rather than treating
    this limited selector as an optimal extraction or impossibility result.
    """
    candidates = sorted((item for item in catalog if item.layout is not None),
                        key=lambda item: (-prod(item.shape), max(item.shape)-min(item.shape), item.sector))
    chosen = []
    for item in candidates:
        conflicts = coarse_conflicts([c.sector for c in (*chosen, item)], power)
        if not conflicts["shared_legs"] and not conflicts["unexpected_sectors"]:
            chosen.append(item)
    if not chosen:
        raise ValueError("no recognized coarse matrix component is available")
    return tuple(chosen)


def extract_coarse_cw(q, power=2):
    """Recover and execute an induced matrix sum from the merged partition.

    Literal coefficients, exact interpolation, direct coefficient expansion,
    and replayed reductions must all agree. Merging does not waive finite
    recovery costs. Raw coefficient support of every catalog component remains
    available even when it cannot enter this first-stage matrix-only packing.
    """
    catalog, groups = coarse_catalog(q, power)
    chosen = choose_coarse_components(catalog, power)
    selections = tuple(tuple(i for item in chosen for i in item.matrix_selections[leg])
                       for leg in range(3))
    expected = classical(chosen[0].shape)
    for item in chosen[1:]:
        expected = direct_sum(expected, classical(item.shape))
    source = tensor_power(coppersmith_winograd(q), power)
    literal = restrict_coordinates(source, selections)
    if literal.dimensions != expected.dimensions or literal.expand() != expected.expand():
        raise AssertionError("coarse induced selection failed its complete independent-product identity")
    _, terms, original_degree, leading = cw_family(q, power)
    restricted = restrict_polynomial(terms, selections)
    degree = polynomial_degree_bound(restricted)
    interpolated = polynomial_coefficient(expected.dimensions, restricted, degree, leading)
    if interpolated.expand() != expected.expand():
        raise AssertionError("coarse polynomial recovery failed its complete independent-product identity")
    recovered, recovery = optimize_recovery(interpolated, restricted, leading)
    recovery["execution"] = execution_evidence(recovered, chosen)
    report = {"schema": "cw-coarse-extraction-v1", "q": q, "power": power,
              "partition_rule": "sum of fine CW group indices in each leg",
              "fine_sector_words": 6**power, "coarse_sector_count": len(catalog),
              "coarse_space_dimensions": [len(group) for group in groups],
              "partition_reconstruction_verified": True,
              "catalog": [item.document() for item in catalog],
              "recognized_matrix_components": sum(item.layout is not None for item in catalog),
              "components_needing_further_work": sum(item.layout is None for item in catalog),
              "selected_sectors": [list(item.sector) for item in chosen],
              "selected_component_shapes": [list(item.shape) for item in chosen],
              "retained_source_coordinates": [list(s) for s in selections],
              "selection_method": "greedy induced matching of proved matrix components; no optimality claim",
              "no_unwanted_cross_sectors_verified": True,
              "literal_direct_sum_identity_verified": True,
              "recovered_direct_sum_identity_verified": True,
              "classical_direct_sum_products": expected.rank,
              "original_degree_bound": original_degree, "degree_bound": degree,
              "coefficient": leading, "optimized_recovery": recovery,
              "source": "https://arxiv.org/abs/2404.16349",
              "limitation": "Only recognized coefficient-one basis permutations enter packing. Other components need further extraction; no general isomorphism or recursive laser search is performed."}
    return recovered, report
