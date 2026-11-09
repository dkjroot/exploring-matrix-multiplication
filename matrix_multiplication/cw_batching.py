"""Run the recursive batching control on the archived CW pair."""

from hashlib import sha256
from pathlib import Path

from .batch_experiments import measure_batch, rational_inputs
from .recipe_parts import load_parts
from .recursive_batching import RecursiveBatch, verify_batch_tensor
from .tensors import direct_sum


SOURCE = Path("reports/2026-11-09-cw-symmetry/parts.json")
REFERENCE = "https://arxiv.org/html/2610.26563v2#A1"


def load_control():
    """Load the two rank-seven parts and bind their lane order to the checked CW union."""
    path = Path(__file__).resolve().parents[1] / SOURCE
    parts, _ = load_parts(path)
    by_name = {evidence["name"]: (tensor, evidence) for tensor, evidence in parts}
    names = ("support-part-0-reduced", "symmetry-map-8-reduced")
    (left, left_evidence), (right, right_evidence) = (by_name[name] for name in names)
    union, joint_evidence = by_name["symmetry-joint-0-8"]
    key = "ambient_source_coordinates_in_matrix_order"
    coordinates = [a+b for a, b in zip(left_evidence[key], right_evidence[key])]
    tensor = direct_sum(left, right)
    if (left.rank != 7 or right.rank != 7 or tensor.expand() != union.expand()
            or coordinates != joint_evidence[key]):
        raise ValueError("the archived rank-seven parts do not match the checked joint CW restriction")
    return tensor, {"parts_library": SOURCE.as_posix(), "parts_library_sha256": sha256(path.read_bytes()).hexdigest(),
                    "components": list(names), "checked_joint_part": joint_evidence["name"], key: coordinates}


def run_control():
    """Verify two levels symbolically and measure actual batches at three sizes."""
    base, source = load_control()
    plan = RecursiveBatch(base, 2, 2)
    expanded = plan.second_level_tensor()
    identity = verify_batch_tensor(expanded, 4, 2)
    if expanded.rank != plan.budget(4)["field_multiplications"]:
        raise AssertionError("second-level tensor length disagrees with the scheduled budget")

    executions = [measure_batch(plan, size) for size in (2, 4, 8)]
    if any(e["separate_strassen"]["observation"]["counts"]["field_multiplications"] != e["budget"]["field_multiplications"]
           for e in executions):
        raise AssertionError("separate Strassen control disagrees with the batched product count")

    evidence = {"schema": "cw-recursive-batch-v1", "source": source, "radix": plan.radix, "lanes": plan.lanes,
                "base_products": base.rank, "child_calls_per_node": len(plan.batches),
                "parent_terms_by_child_slot": plan.batches, "second_level_identity": identity,
                "product_recurrence": "P(1) = 2, P(n) = 7 P(n/2)",
                "execution_scope": "public calls with input validation and instrumentation, excluding independent checks",
                "input_pattern": "cw_batching.rational_inputs, zero-based row, column and lane indices",
                "executions": executions}
    parts = ((base, {"name": "cw-batch-base", "component_shapes": [[2]*3]*2, "products": base.rank, "source": source}),
             (expanded, {"name": "cw-batch-second-level", "component_shapes": [[4]*3]*2,
                         "products": expanded.rank, "parent_part": "cw-batch-base",
                         "parent_terms_by_child_slot": plan.batches, "complete_identity": identity}))
    return parts, evidence


def batch_summary(report):
    """Explain the child routing and show the counts separately from the timings."""
    rows = "".join(f'<tr><td>{e["dimension"]}</td><td>{e["budget"]["field_multiplications"]}</td>'
                   f'<td>{e["observation"]["counts"]["field_multiplications"]}</td>'
                   f'<td>{e["separate_strassen"]["observation"]["counts"]["field_multiplications"]}</td>'
                   f'<td>{e["classical_products"]}</td></tr>' for e in report["executions"])
    checks = report["second_level_identity"]["coefficient_checks"]
    return ('<h2>Computing independent products in recursive batches</h2>'
            '<p>Each call returns two ordinary matrix products on separate inputs. Above size one, '
            'the base recipe gives fourteen products of smaller blocks. We put successive pairs of '
            'those block products into seven child calls. Each returned block goes back to its own '
            'parent term, which tells us where to add it and with what coefficient. For example, '
            'the fourth call returns the products for terms 6 and 7, which contribute to different '
            'parent outputs. Terms and child slots are numbered from zero in the saved schedule.</p>'
            f'<p>The expanded two-level recipe was checked on all {checks:,} coefficients, including '
            'those that must vanish between the two products. The execution checks below compare '
            'each output with a separate rational dot-product reference.</p>'
            '<table><thead><tr><th>Matrix dimension</th><th>Predicted products</th><th>Observed products</th>'
            f'<th>Separate Strassen</th><th>Classical</th></tr></thead><tbody>{rows}</tbody></table>'
            '<p>These counts are scalar data products. The JSON also retains the predicted and observed '
            'coefficient scalings and additions for the batched schedule. The two rank-seven bases '
            'give the same product recurrence as separate Strassen calls.</p>')
