"""Generate and execute Pan's exact aggregation of two independent products.

The three correction families follow equations (1)–(3) of Solomonik and
Bläser, arXiv:2610.26563v2, which give Pan's trilinear construction. The code
generates the sparse factors directly and uses our own recursive scheduler.
"""

from itertools import product

from .batch_experiments import measure_batch
from .recursive_batching import RecursiveBatch, verify_batch_tensor
from .tensors import Tensor, classical, direct_sum


REFERENCE = "https://arxiv.org/html/2610.26563v2#S1"
PAN_REFERENCE = "https://doi.org/10.1137/0209027"


def pan_pair(size):
    """Build the aggregate followed by three negative correction families.

    Write the two outputs as AB and UV. The trace variables d[h,i] and
    w[i,j] pair with output entries AB[i,h] and UV[j,i], respectively.
    All three legs store lane first, then the actual row and column.
    """
    if type(size) is not int or size < 1:
        raise ValueError("Pan dimension must be a positive integer")

    def coordinate(lane, row, column):
        """Return the lane-major, row-major coordinate for any of the three legs."""
        return (lane*size+row)*size+column

    indices = range(size)
    aggregate = [(((coordinate(0, i, j), 1), (coordinate(1, j, h), 1)),
                  ((coordinate(0, j, h), 1), (coordinate(1, h, i), 1)),
                  ((coordinate(0, i, h), 1), (coordinate(1, j, i), 1)))
                 for i, j, h in product(indices, repeat=3)]

    # Fix i,j. These products cancel ABW and AVW, summing over h in the right form.
    right_correction = [(((coordinate(0, i, j), 1),),
                         tuple(entry for h in indices
                               for entry in ((coordinate(0, j, h), 1), (coordinate(1, h, i), 1))),
                         ((coordinate(1, j, i), -1),))
                        for i, j in product(indices, repeat=2)]
    # Fix j,h. These cancel UBD and UBW, summing over i in the output factor.
    output_correction = [(((coordinate(1, j, h), 1),), ((coordinate(0, j, h), 1),),
                          tuple(entry for i in indices
                                for entry in ((coordinate(0, i, h), -1), (coordinate(1, j, i), -1))))
                         for j, h in product(indices, repeat=2)]
    # Fix h,i. These cancel AVD and UVD, summing over j in the left form.
    left_correction = [(tuple(entry for j in indices
                              for entry in ((coordinate(0, i, j), 1), (coordinate(1, j, h), 1))),
                        ((coordinate(1, h, i), 1),), ((coordinate(0, i, h), -1),))
                       for h, i in product(indices, repeat=2)]
    return Tensor((2*size**2,)*3, aggregate+right_correction+output_correction+left_correction)


def correction_evidence(tensor, size):
    """Independently bind the aggregate's excess coefficients to the paid corrections."""
    split = size**3
    aggregate = Tensor(tensor.dimensions, tensor.terms[:split]).expand()
    correction = Tensor(tensor.dimensions, tensor.terms[split:]).expand()
    classical_product = classical((size,)*3)
    target = direct_sum(classical_product, classical_product).expand()
    excess = {key: target.get(key, 0)-aggregate.get(key, 0) for key in aggregate.keys() | target.keys()
              if target.get(key, 0) != aggregate.get(key, 0)}
    if correction != excess or len(excess) != 6*split or any(c != -1 for c in excess.values()):
        raise AssertionError("the paid Pan corrections do not exactly cancel the six unwanted monomial families")
    return {"aggregate_terms": split, "correction_terms_by_family": dict.fromkeys(("right", "output", "left"), size**2),
            "cancelled_mixed_coefficients": len(excess)}


def run_pan_batch(power=2):
    """Run the fixed radix-four experiment, with a base-only scope for small sweeps."""
    if type(power) is not int or power not in (1, 2):
        raise ValueError("Pan control supports one or two levels")
    radix, lanes = 4, 2
    base = pan_pair(radix)
    corrections = correction_evidence(base, radix)
    plan = RecursiveBatch(base, radix, lanes)
    source = {"formula": REFERENCE, "original_method": PAN_REFERENCE, "base_dimension": radix}
    base_identity = verify_batch_tensor(base, radix, lanes)
    parts = [(base, {"name": "pan-batch-base", "component_shapes": [[radix]*3]*lanes,
                     "products": base.rank, "source": source, "complete_identity": base_identity})]
    evidence = {"schema": "pan-recursive-batch-v1", "source": source, "radix": radix, "lanes": lanes,
                "base_products": base.rank, "base_identity": base_identity, "corrections": corrections,
                "child_calls_per_node": len(plan.batches), "parent_terms_by_child_slot": plan.batches,
                "product_recurrence": "P(1) = 2, P(n) = 56 P(n/4)",
                "execution_scope": "instrumented public calls with validation, excluding construction and independent checks",
                "input_pattern": "batch_experiments.rational_inputs, zero-based row, column and lane indices",
                "executions": []}
    if power == 2:
        expanded = plan.second_level_tensor()
        size = radix**2
        identity = verify_batch_tensor(expanded, size, lanes)
        if expanded.rank != plan.budget(size)["field_multiplications"]:
            raise AssertionError("second-level Pan terms disagree with the scheduled budget")
        evidence["second_level_identity"] = identity
        parts.append((expanded, {"name": "pan-batch-second-level", "component_shapes": [[size]*3]*lanes,
                                 "products": expanded.rank, "parent_part": "pan-batch-base",
                                 "parent_terms_by_child_slot": plan.batches, "complete_identity": identity}))
    for level in range(1, power+1):
        evidence["executions"].append(measure_batch(plan, radix**level, compare_classical=True))
    return tuple(parts), evidence


def pan_summary(report):
    """Describe why the mixed intermediate products still give independent answers."""
    rows = "".join(f'<tr><td>{e["dimension"]}</td><td>{e["budget"]["field_multiplications"]:,}</td>'
                   f'<td>{e["observation"]["counts"]["field_multiplications"]:,}</td>'
                   f'<td>{e["separate_strassen"]["observation"]["counts"]["field_multiplications"]:,}</td>'
                   f'<td>{e["separate_classical"]["observation"]["counts"]["field_multiplications"]:,}</td>'
                   f'<td>{e["observation"]["counts"]["field_additions"]:,}</td>'
                   f'<td>{e["observation"]["counts"]["coefficient_multiplications"]:,}</td></tr>'
                   for e in report["executions"])
    identity = report.get("second_level_identity", report["base_identity"])
    return ('<h2>Pan’s exact aggregation of two products</h2>'
            '<p>Each aggregate term multiplies a left sum by a right sum. Both sums contain entries '
            'from the two input pairs, and the product contributes to both outputs. '
            'Expanding gives two wanted monomials and six unwanted ones. Three families of correction products '
            'remove those unwanted terms in pairs by summing along a shared index. For size four we pay for '
            '64 aggregate products and all 48 correction products. Together they give two independent answers.</p>'
            '<p>At a recursive node the 112 block products go to 56 two-slot child calls. Each returned block '
            'goes back to the parent term that requested it, including any negative correction coefficients. '
            f'The largest expanded identity checks {identity["coefficient_checks"]:,} coefficients, including '
            'all zeros between outputs. Sparse coefficient equality checks the zero entries without allocating a dense cube.</p>'
            '<table><thead><tr><th>Dimension</th><th>Predicted data products</th><th>Observed data products</th>'
            '<th>Separate Strassen</th><th>Separate classical</th><th>Pan additions</th>'
            f'<th>Pan coefficient scalings</th></tr></thead><tbody>{rows}</tbody></table>'
            '<p>Each output also passed a rational dot-product check. The reduction against classical '
            'concerns scalar data products. Separate Strassen uses fewer data products at these sizes. '
            'The JSON retains the observations for all three schedules.</p>')
