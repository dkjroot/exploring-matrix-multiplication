"""CLI for constructing, searching, replaying, and exporting exact recipes.

Run ``python -m matrix_multiplication.constructions --help``. Each construction
writes a fresh dated directory below outputs, with data and an offline HTML
summary. Matrix recipes enter the same composition, measurement, and portable
certificate paths as imported schemes. CW extraction can execute independent
matrix products; a direct sum is not labelled a dense matrix algorithm. No
third-party implementation executes here.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from hashlib import sha256
from html import escape
import json
from math import log
from pathlib import Path

from .certificates import formula_recipe, write_bundle
from .composition import Composition
from .decomposition_search import beats_exponent, replay, search
from .degenerations import recover_cw
from .cw_batching import REFERENCE as BATCH_REFERENCE, batch_summary, run_control as run_batch_control
from .pan_batching import pan_summary, run_pan_batch
from .cw_extraction import extract_cw
from .cw_interior import explore_interiors, interior_summary
from .cw_neighborhood import explore_neighborhoods, neighborhood_summary, compact_neighborhood_report
from .cw_support import explore_support, support_summary, compact_support_report
from .dense_packing import pack_cw, pack_coarse_cw
from .report_explanations import construction_guide, terminology_guide
from .scheme_io import load_scheme_document, save_scheme, scheme_document, scheme_from_document
from .tensors import from_scheme, classical, matrix_product, to_scheme


@dataclass(frozen=True)
class ConstructionRequest:
    """A reproducible construction's parameters, shared by CLI and suite runner."""

    command: str
    output_root: Path = Path("outputs")
    size: int = 2
    start: str = None
    target_rank: int = None
    target_exponent: Fraction = None
    steps: int = 1000
    restarts: int = 10
    seed: int = 0
    max_height: int = 1
    extra_rank: int = 0
    split_every: int = 50
    certificate: bool = False
    left: str = None
    right: str = None
    q: int = 2
    power: int = 1
    sector_words: tuple = None
    report: Path = None
    candidate_budget: int = 2048
    max_parts: int = 2
    max_source_dimension: int = 64
    max_source_terms: int = 4096
    neighborhood_budget: int = 4096
    neighborhood_max_witnesses: int = 64
    support_budget: int = 4096
    support_search_seeds: tuple = (0, 1)
    support_reduction_limit: int = 1


def recipe(spec):
    """Read an exact scheme path or extract a maintained seven-product formula."""
    return scheme_from_document(recipe_document(spec))


def recipe_document(spec):
    """Retain the exact source document and its attribution through constructions."""
    if spec in ("strassen", "strassen-winograd"):
        return formula_recipe(spec)
    return load_scheme_document(spec)


def target_rank(dimension, exponent):
    """Find the largest integer rank strictly beating a rational square exponent."""
    if type(dimension) is not int or dimension < 2:
        raise ValueError("target dimension must be an integer at least two")
    if type(exponent) not in (int, Fraction) or exponent <= 0:
        raise ValueError("target exponent must be an exact positive rational")
    exponent = Fraction(exponent)
    low, high = 0, 1
    while beats_exponent(high, dimension, exponent.numerator, exponent.denominator):
        high *= 2
    while low+1 < high:
        mid = (low+high)//2
        if beats_exponent(mid, dimension, exponent.numerator, exponent.denominator):
            low = mid
        else:
            high = mid
    if low < 1:
        raise ValueError("requested exponent admits no positive finite rank")
    return low


def write_json(path, document):
    """Write deterministic readable data with no approximate nonfinite values."""
    path.write_text(json.dumps(document, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def tensor_document(tensor):
    """Serialize generic rational terms separately from matrix recipe interchange."""
    return {"schema": "rational-tensor-v1", "dimensions": list(tensor.dimensions),
            "terms": [[[[i, [c.numerator, c.denominator]] for i, c in factor]
                       for factor in term] for term in tensor.terms]}


def write_summary(directory, report):
    """Create a readable offline research summary with a labelled rank trace."""
    evidence = report.get("search", {})
    curve = ""
    if evidence:
        # Replaying the algebra, rather than predicting rank from event names,
        # captures zero factors that a flip can eliminate as well as reductions.
        start = from_scheme(scheme_from_document(report["starting_recipe"]))
        ranks = [start.rank]
        for event in evidence["events"]:
            start = replay(start, [event])
            ranks.append(start.rank)
        maximum = max(ranks)+1
        points = " ".join(f"{70+590*i/max(1,len(ranks)-1):.2f},{260-220*r/maximum:.2f}"
                          for i, r in enumerate(ranks))
        curve = (f'<svg viewBox="0 0 720 330" role="img" aria-label="Verified decomposition products along accepted search path">'
                 '<path d="M70 30 V260 H660" fill="none" stroke="black"/>'
                 f'<polyline points="{points}" fill="none" stroke="#0072B2" stroke-width="3"/>'
                 '<text x="350" y="310" text-anchor="middle">Accepted transformation number</text>'
                 '<text transform="translate(20,150) rotate(-90)" text-anchor="middle">Products in decomposition</text>'
                 f'<text x="52" y="277">0</text><text x="630" y="277">{len(ranks)-1}</text>'
                 f'<text x="35" y="45">{maximum}</text><text x="45" y="260">0</text></svg>'
                 '<p>The search tries algebraic changes that preserve the computation while looking for a recipe '
                 'with fewer products. This curve shows only the sequence of accepted changes leading to the best '
                 'recipe it found. Other trial paths and restarts may have failed to improve it; those attempts '
                 'are included in the counts below, rather than in this curve. We can replay each recorded change '
                 'without repeating the random search. Failing to reach a target within this budget does not '
                 'prove that a better recipe is impossible.</p>')
    cw = report.get('cw')
    if cw:
        curve += ('<h2>Recovering an exact CW building block</h2>'
                  '<p>The CW formula used here has coefficients that are polynomials in a formal parameter. '
                  'We want the coefficient of one particular power of that parameter. Instead of choosing a '
                  'tiny floating approximation, we recover that coefficient exactly by evaluating the polynomial '
                  'at enough rational points and combining the results. That procedure is called interpolation. '
                  'Its extra products belong to the finite executable recipe and cannot be ignored.</p>'
                  '<p>The q parameter controls the size of the basic CW structure. The power parameter combines '
                  'that many copies into a larger tensor; it is not the matrix-growth exponent p. This page checks '
                  'the whole recovered tensor against an independent literal definition. It has not yet converted '
                  'that tensor into one ordinary square matrix multiplication.</p>'
                  '<table><thead><tr><th>Quantity</th><th>Recorded value</th></tr></thead><tbody>'
                  + ''.join(f'<tr><td>{label}</td><td>{cw[key]}</td></tr>' for label, key in (
                      ('CW size parameter q', 'q'), ('Number of copies combined', 'power'),
                      ('Polynomial rank-one products before recovery', 'polynomial_rank'),
                      ('Interpolation evaluation points', 'interpolation_points'),
                      ('Products in the finite recovered recipe', 'finite_recipe_products'),
                      ('Products in the independent literal recipe', 'literal_recipe_products')))
                  + '</tbody></table>')
    packing = report.get("cw_packing")
    extraction = report.get("cw_extraction") or (packing["extraction"] if packing else None)
    if extraction:
        # Expose the mathematical result before the raw evidence. In particular,
        # a reader should see the paid finite cost, rather than mistake the much
        # smaller polynomial rank for a working multiplication's product count.
        rows = ''.join('<tr><td>'+escape(' / '.join(item['word']))+'</td><td>'
                       +escape(' × '.join(map(str, item['shape'])))+'</td><td>'
                       +str(item['classical_products'])+'</td></tr>' for item in extraction['components'])
        curve += ('<h2>Verified independent matrix products</h2>'
                  '<p>Here we keep selected coordinates of the CW structure and set the others to zero, looking '
                  'for separate ordinary matrix products inside it. Each shape lists the number of rows, shared '
                  'inner entries, and output columns. For example, (2,3,4) describes a 2×3 input multiplied by a '
                  '3×4 input. The selected products must remain independent: reusing the wrong coordinate groups '
                  'could accidentally leave cross-products between different components. We check the complete '
                  'algebraic identity to exclude those terms, then run actual rational-input examples too.</p>'
                  '<p>The “sector word” identifies a sequence of coordinate-group choices in the tensor power. '
                  'It is a mathematical label, unrelated to the 64-bit machine words used in our cost model.</p>'
                  '<table><thead><tr><th>Sector word</th><th>Product shape</th><th>Classical data products</th></tr></thead>'
                  f'<tbody>{rows}</tbody></table>'
                  f'<p>Classical sum: <strong>{extraction["classical_direct_sum_products"]}</strong> data products. '
                  f'Finite recovered recipe, measured: <strong>{extraction["finite_recipe_products"]}</strong>. '
                  f'Interpolation upper bound: <strong>{extraction["finite_rank_bound"]}</strong>. '
                  'The first count is the cost of doing the selected products by ordinary multiplication. '
                  'The measured count pays for our exact coefficient-recovery recipe too. The upper bound '
                  'is a conservative construction estimate, not a smaller measured cost. Additions and '
                  'fixed-coefficient products are retained in the execution counters below.</p>'
                  + ('<p>The next section uses these products in verified finite dense packing.</p>' if packing else
                     '<p>We can now execute several independent products. That is a useful extraction milestone, '
                     'but it is not yet an efficient multiplier for one larger, unrestricted matrix pair. '
                     'Connecting those products to a single dense multiplication requires another construction.</p>'))
    if packing:
        recovery = extraction["optimized_recovery"]
        stages = (("Exact interpolation baseline", recovery["interpolation_products"]),
                  ("Direct coefficient expansion", recovery["coefficient_expansion_products"]),
                  ("Expansion after replayed reductions", recovery["reduced_expansion_products"]),
                  ("Chosen direct-sum recipe per batch", recovery["selected_products"]),
                  ("Mapped dense recipe before packing reductions", packing["mapped_products_before_reductions"]),
                  ("Packed dense recipe", packing["packed_products"]),
                  ("Classical dense baseline", packing["classical_dense_products"]))
        cost_rows = ''.join(f'<tr><td>{escape(label)}</td><td>{count}</td></tr>' for label, count in stages)
        curve += ('<h2>Exact recovery and dense packing</h2>'
                  '<p>We compare two ways to obtain the exact polynomial coefficient. Interpolation combines '
                  'evaluations at several parameter values. Direct coefficient expansion instead multiplies '
                  'the relevant coefficient vectors from the polynomial factors. Each resulting rank-one term '
                  'describes one product of an input combination from A and one from B, with weights assigning '
                  'its contribution to the output. Some terms have exact algebraic dependencies, which let us '
                  'remove products while preserving every tensor coefficient. We record and replay those '
                  'reductions, then use the less expensive verified recipe.</p>'
                  '<table><thead><tr><th>Construction stage</th><th>Data products</th></tr></thead>'
                  f'<tbody>{cost_rows}</tbody></table>'
                  f'<p>The dense target is {escape(str(packing["target_shape"]))}. '
                  f'We use {packing["batch_count"]} batch(es) of the available component products. '
                  'Think of ordinary multiplication as contributions A[i,j]×B[j,k] to output C[i,k]. '
                  'Our packing assigns rectangular groups of these contributions to the component slots. '
                  'Every (row, inner, column) triple must occur exactly once: a missing triple loses a term, '
                  'and a repeated one double-counts it. A partly used slot receives zeros in its unused entries. '
                  'Explicit coordinate maps reuse the appropriate inputs and add contributions into shared '
                  'outputs. We check this coverage and the final complete identity, then separately check '
                  'actual rational outputs.</p>'
                  '<p>This finite tiling construction is executable, so its exported square recipe can be '
                  'repeated in recursion and measured. The more powerful asymptotic sum inequality is a '
                  'theoretical bridge from many independent products to a long-term multiplication bound. '
                  'Our current tiling does not implement that full bridge. Its exponent comes from the dense '
                  'recipe we actually export, not from the stronger bound reported for a related theory.</p>')
    coarse = report.get("cw_coarse")
    if coarse:
        # Keep classification misses visible: the most interesting merged
        # components can require another extraction rather than matrix packing.
        rows = ''
        selected = {tuple(s) for s in coarse["selected_sectors"]}
        for item in coarse["catalog"]:
            classification = item["classification"]
            shape = ' × '.join(map(str, item["shape"])) if "shape" in item else 'Needs further extraction'
            rows += ('<tr><td>'+escape(str(item["sector_weights"]))+'</td><td>'
                     +escape(' × '.join(map(str, item["dimensions"])))+'</td><td>'
                     +escape(shape)+'</td><td>'+escape(classification["status"])+'</td><td>'
                     +('Yes' if tuple(item["sector_weights"]) in selected else 'No')+'</td></tr>')
        recovery, packed = coarse["optimized_recovery"], coarse["packing"]
        curve += ('<h2>Merged CW components</h2>'
                  '<p>The first CW partition keeps many coordinate groups separate. In this experiment we '
                  'merge groups whose sequences of group labels have the same total. This makes a coarser '
                  'partition: fewer, larger components that may contain useful structure. We check that '
                  'putting all their coefficients back in their original positions reconstructs the complete '
                  'CW tensor. A bigger component is not automatically a bigger matrix multiplication, so '
                  'we classify each one separately. Every recognized matrix product has a recorded '
                  'reordering of its coordinates and a complete coefficient check.</p>'
                  f'<p>{coarse["fine_sector_words"]} fine sector words become '
                  f'{coarse["coarse_sector_count"]} coarse sectors: '
                  f'{coarse["recognized_matrix_components"]} recognized matrix products and '
                  f'{coarse["components_needing_further_work"]} needing further work. '
                  'The catalog shares coordinate spaces between entries; its rows cannot all be executed '
                  'as independent products at once. Only the selected induced sum is independent: that means '
                  'the selected pieces have disjoint spaces and leave no unintended cross-products between '
                  'them. Unrecognized components keep all their exact coefficients for further research.</p>'
                  '<table><thead><tr><th>Weights X,Y,Z</th><th>Space dimensions X,Y,Z</th>'
                  '<th>Matrix shape m,k,n</th><th>Classification</th><th>Selected</th></tr></thead>'
                  f'<tbody>{rows}</tbody></table>'
                  '<p>A “dimension obstruction” means the component’s three space sizes cannot be written '
                  'as (m×k, k×n, m×n) for positive integer matrix dimensions. The full component therefore '
                  'cannot simply be relabelled as an ordinary matrix product. Smaller restrictions or '
                  'further combinations may still be useful. Other non-recognition has a narrower meaning: '
                  'our test looks for coefficient-one matrix structure by reordering coordinates, not by '
                  'searching every possible linear change of basis. Failure of that test is not a general '
                  'proof that no useful transformation exists.</p>'
                  '<h2>Verified finite recovery and packing costs</h2>'
                  '<table><thead><tr><th>Stage</th><th>Data products</th></tr></thead><tbody>'
                  + ''.join(f'<tr><td>{escape(label)}</td><td>{count}</td></tr>' for label, count in (
                      ("Interpolation", recovery["interpolation_products"]),
                      ("Coefficient expansion", recovery["coefficient_expansion_products"]),
                      ("Replayed expansion reductions", recovery["reduced_expansion_products"]),
                      ("Selected independent-sum recipe", recovery["selected_products"]),
                      ("Mapped dense recipe", packed["mapped_products_before_reductions"]),
                      ("Packed dense recipe", packed["packed_products"]),
                      ("Classical dense baseline", packed["classical_dense_products"])))
                  + '</tbody></table><p>The selected independent products and the packed dense recipe '
                  'both pass actual rational-input checks, and their counted products match their saved '
                  'recipes. All finite recovery costs are included. This is finite tiling, not a recursive '
                  'laser construction or the asymptotic sum inequality conversion: the stronger theoretical '
                  'methods still need a constructive implementation. Successful classification and packing '
                  'are therefore research steps, not permission to attach the paper’s limiting exponent '
                  'to this small executable example.</p>')
    interior = report.get("cw_interior")
    if interior:
        curve += interior_summary(interior)
    neighborhood = report.get("cw_neighborhood")
    if neighborhood:
        curve += neighborhood_summary(neighborhood)
    support = report.get('cw_support')
    if support:
        curve += support_summary(support)
    batch = report.get("cw_batching")
    if batch:
        curve += batch_summary(batch)
    pan = report.get("pan_batching")
    if pan:
        curve += pan_summary(pan)
    display = {k: v for k, v in report.items() if k != "starting_recipe"}
    if interior:
        # Preserve all attempted maps in JSON; the readable HTML uses outcome
        # counts and part tables rather than thousands of failed candidate rows.
        display["cw_interior"] = {**interior, "experiments": [
            {**entry, "search": {key: value for key, value in entry.get("search", {}).items()
                                  if key != "candidates"}} for entry in interior["experiments"]]}
    if neighborhood:
        display["cw_neighborhood"] = compact_neighborhood_report(neighborhood)
    if support:
        display['cw_support'] = compact_support_report(support)
    if "search" in display:
        display["search"] = {k: v for k, v in evidence.items() if k != "events"}
    links = ' · '.join(f'<a href="{escape(path.name)}">{escape(path.name)}</a>'
                       for path in sorted(directory.glob("*.json")))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Exact construction experiment</title><style>'
            'body{font:18px/1.6 system-ui;max-width:1000px;margin:2rem auto;padding:1rem;color:#17202a;background:#fff}'
            'pre{font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem}'
            'svg{width:100%;max-width:760px}a{color:#005a8d}'
            'table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.5rem;border-bottom:1px solid #bbb}</style>'
            '<h1>Exact construction experiment</h1>'
            f'<p>{escape(report["outcome"])}</p><p>{links}</p>'
            +construction_guide()+terminology_guide()+curve+
            '<p>Building or searching for a recipe is preparation work. Applying the finished recipe to '
            'input matrices is a separate calculation, with its own arithmetic and resource measurements. '
            'The combined suite automatically measures the generated recipes it admits to comparison. '
            'For a standalone matrix recipe, its scheme.json file can be supplied to explore.py or sweep.py '
            'with the composition option; generic tensor evidence is not automatically a matrix algorithm.</p>'
            f'<pre>{escape(json.dumps(display, indent=2))}</pre></html>')
    (directory / "index.html").write_text(html, encoding="utf-8")


def run_experiment(args, directory=None):
    """Execute a typed construction request; return its directory and exact evidence.

    The CLI and combined sweep share this function. An explicit directory makes
    suite artifacts predictably named under one dated run, without nested dates.
    Existing directories are refused; callers retain earlier experiment output.
    """
    if args.command not in ("search", "product", "replay", "cw", "cw-extract", "cw-pack", "cw-coarse", "cw-interior",
                            "cw-neighborhood", "cw-support", "cw-batch", "pan-batch"):
        raise ValueError("unknown construction experiment")
    if args.command == "search" and args.target_rank is not None and args.target_exponent is not None:
        raise ValueError("choose a rank or exponent target")
    provenance = {"implementation": "independent rational tensor construction primitives",
                  "implementation_license": "GPL-3.0; repository LICENSE"}
    report = {"schema": "construction-experiment-v1", "command": args.command,
              "coefficient_field": "Q", "limitations": "Finite exact recipes only; no new exponent bound follows from a search attempt."}
    scheme = tensor = None
    interior_parts = ()
    if args.command == "search":
        if args.start:
            start_document = recipe_document(args.start)
            original = scheme_from_document(start_document)
            shape = original.shape
            # Retain attribution and license conditions from external data.
            provenance["input_provenance"] = start_document.get("provenance", {})
        else:
            if args.size < 2:
                raise ValueError("search size must be at least two")
            shape = (args.size,)*3
            original = to_scheme(classical(shape), shape, "classical-start")
            start_document = scheme_document(original)
        if args.target_exponent is not None and len(set(shape)) != 1:
            raise ValueError("exponent targeting requires a square scheme; rectangular searches use --target-rank")
        desired = (target_rank(shape[0], args.target_exponent) if args.target_exponent is not None
                   else args.target_rank if args.target_rank is not None else original.rank-1)
        best, evidence = search(from_scheme(original), shape, target_rank=desired, steps=args.steps,
                                restarts=args.restarts, seed=args.seed, max_height=args.max_height,
                                extra_rank=args.extra_rank, split_every=args.split_every)
        scheme = to_scheme(best, shape, "search-result")
        report.update({"starting_recipe": start_document, "shape": list(shape), "search": evidence,
                       "settings": {"steps_per_restart": args.steps, "restarts": args.restarts,
                                    "max_height": args.max_height,
                                    "extra_rank": args.extra_rank, "split_every": args.split_every,
                                    "target_exponent": str(args.target_exponent) if args.target_exponent else None}})
        provenance["method_reference"] = "https://arxiv.org/abs/2212.01175"
    elif args.command == "product":
        # Read each source once: execution coefficients and retained notices must
        # describe the same input document, even if the file later changes.
        input_documents = [recipe_document(args.left), recipe_document(args.right)]
        left, right = (scheme_from_document(doc) for doc in input_documents)
        shape = tuple(x*y for x, y in zip(left.shape, right.shape))
        scheme = to_scheme(matrix_product(from_scheme(left), left.shape, from_scheme(right), right.shape), shape, "tensor-product")
        report["input_recipes"] = input_documents
        provenance["input_provenance"] = [doc.get("provenance", {}) for doc in report["input_recipes"]]
    elif args.command == "replay":
        prior = json.loads(args.report.read_text(encoding="utf-8"))
        original = scheme_from_document(prior["starting_recipe"])
        shape = original.shape
        best = replay(from_scheme(original), prior["search"]["events"])
        scheme = to_scheme(best, shape, "replayed-result")
        if scheme.rank != prior["search"]["best_rank"]:
            raise ValueError("replayed rank disagrees with report")
        report.update({"starting_recipe": prior["starting_recipe"], "search": prior["search"],
                       "replayed_report_sha256": sha256(args.report.read_bytes()).hexdigest()})
        provenance["parent_report"] = prior.get("provenance", {})
    elif args.command == "cw":
        tensor, evidence = recover_cw(args.q, args.power)
        report.update({"cw": evidence, "outcome": "Polynomial CW tensor recovered and exactly verified. This is not yet a matrix multiplier."})
    elif args.command == "cw-extract":
        tensor, evidence = extract_cw(args.q, args.power, args.sector_words)
        report.update({"cw_extraction": evidence,
                       "outcome": f"Extracted and executed {evidence['selected_components']} independent matrix products; "
                                  f"the finite recovered recipe uses {tensor.rank} data products. Dense packing remains open."})
        provenance["method_reference"] = evidence["source"]
        provenance["polynomial_reference"] = "https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf"
        # A single balanced component is already an ordinary matrix identity.
        # Export only shapes supported by the existing recursive scheme engine.
        # Several independent products must never be relabelled as one larger
        # dense product, and unit-size axes remain executable tensor evidence.
        if evidence["selected_components"] == 1:
            shape = tuple(evidence["components"][0]["shape"])
            if min(shape) >= 2:
                scheme = to_scheme(tensor, shape, "cw-extracted-component")
    elif args.command == "cw-pack":
        tensor, evidence = pack_cw(args.q, args.power, args.size, args.sector_words)
        report["cw_packing"] = evidence
        scheme = to_scheme(tensor, (args.size,)*3, "cw-packed-dense")
        provenance["method_reference"] = evidence["extraction"]["source"]
        provenance["polynomial_reference"] = "https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf"
        provenance["construction"] = "independent finite volume tiling, direct coefficient recovery, and replayed exact reductions"
    elif args.command == "cw-coarse":
        if args.sector_words is not None:
            raise ValueError("fine sector words cannot select coarse components")
        tensor, evidence = pack_coarse_cw(args.q, args.power, args.size)
        report["cw_coarse"] = evidence
        scheme = to_scheme(tensor, (args.size,)*3, "cw-coarse-packed-dense")
        provenance["method_reference"] = evidence["source"]
        provenance["polynomial_reference"] = "https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf"
        provenance["construction"] = "independent sum-index coarsening, exact support classification, polynomial recovery and finite packing"
    elif args.command == "cw-interior":
        if args.sector_words is not None:
            raise ValueError("fine sector words cannot select interior components")
        interior_parts, evidence = explore_interiors(args.q, args.power,
            candidate_budget=args.candidate_budget, max_parts=args.max_parts,
            max_source_dimension=args.max_source_dimension, max_source_terms=args.max_source_terms)
        report["cw_interior"] = evidence
        report["outcome"] = (f"Retained {len(interior_parts)} exactly recovered reusable parts, including recombinations. "
                             "Bounded search and full recovery costs are recorded; no improved exponent is claimed.")
        provenance["method_reference"] = "https://arxiv.org/abs/2404.16349"
        provenance["polynomial_reference"] = "https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf"
        provenance["construction"] = "bounded coordinate restrictions, exact CW polynomial recovery, witness products and verified recombination"
        provenance["source_parameters"] = {"q": args.q, "cw_power": args.power, "component_powers": [1, 2]}
        if evidence["recombinations"]:
            tensor, description = interior_parts[-1]
            shape = tuple(description["shape"])
            if min(shape) >= 2:
                scheme = to_scheme(tensor, shape, "cw-interior-recombined")
    elif args.command == 'cw-neighborhood':
        if args.sector_words is not None:
            raise ValueError("fine sector words cannot select coordinate neighborhoods")
        interior_parts, evidence = explore_neighborhoods(args.q, args.power,
            seed_budget=args.candidate_budget, candidate_budget=args.neighborhood_budget,
            max_parts=args.max_parts, max_witnesses=args.neighborhood_max_witnesses,
            max_source_dimension=args.max_source_dimension, max_source_terms=args.max_source_terms)
        report['cw_neighborhood'] = evidence
        report['outcome'] = (f"Seeded coordinate and joint-restriction experiment completed: "
            f"{evidence['new_verified_shapes']} new verified shapes and "
            f"{evidence['strict_recovery_savings']} joint-recovery savings. Bounded misses are not exclusions.")
        provenance.update(method_reference="https://arxiv.org/abs/2404.16349",
            polynomial_reference="https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf",
            construction="exact seeded coordinate neighborhoods, checked dense unions/direct sums and paid joint recovery",
            source_parameters={"q": args.q, "cw_power": args.power, "component_powers": [1, 2]})
        # Generic sums retain their component shapes. Only an ordinary supported
        # matrix recipe, with no worse than classical rank, joins recursion.
        for candidate, description in interior_parts:
            shape = tuple(description.get('shape', ()))
            if shape and min(shape) >= 2 and candidate.rank <= shape[0]*shape[1]*shape[2]:
                tensor = candidate
                scheme = to_scheme(tensor, shape, "cw-neighborhood-matrix")
                break
    elif args.command == "cw-batch":
        interior_parts, evidence = run_batch_control()
        report["cw_batching"] = evidence
        report["outcome"] = "Verified and executed two independent square matrix products at dimensions 2, 4 and 8."
        provenance.update(method_reference=BATCH_REFERENCE,
                          construction="recursive equal-size direct-sum batching from verified CW parts",
                          input_provenance=evidence["source"])
    elif args.command == "pan-batch":
        interior_parts, evidence = run_pan_batch(args.power)
        report["pan_batching"] = evidence
        dimensions = " and ".join(str(item["dimension"]) for item in evidence["executions"])
        report["outcome"] = f"Verified Pan’s two-product aggregation and executed both outputs at dimension {dimensions}."
        provenance.update(method_reference=evidence["source"]["formula"],
                          original_method=evidence["source"]["original_method"],
                          construction="independent generation of Pan’s exact aggregate and correction terms")
    else:
        if args.sector_words is not None:
            raise ValueError('fine sector words cannot select support-search targets')
        interior_parts, evidence = explore_support(args.q, args.power,
            candidate_budget=args.support_budget, seed_budget=args.candidate_budget,
            max_parts=args.max_parts, steps=args.steps, restarts=args.restarts,
            search_seeds=args.support_search_seeds, max_height=args.max_height,
            reduction_limit=args.support_reduction_limit,
            max_source_dimension=args.max_source_dimension, max_source_terms=args.max_source_terms)
        report['cw_support'] = evidence
        report['outcome'] = 'Bounded support, recovery and cyclic follow-up outcomes are recorded below.'
        provenance.update(method_reference='https://arxiv.org/abs/2404.16349',
            polynomial_reference='https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf',
            search_reference='https://arxiv.org/abs/2212.01175',
            construction='seed-pool support, balanced targets, paid recovery, replayed rank reduction and cyclic joint selection',
            source_parameters={'q': args.q, 'cw_power': args.power, 'component_power': 2})
        # Prefer the explicit recombination; its source parts and rank walks
        # remain available in the library and report, including unsuccessful seeds.
        for candidate, description in reversed(interior_parts[:len(evidence['parts'])]):
            shape = tuple(description['shape'])
            if min(shape) >= 2 and candidate.rank <= shape[0]*shape[1]*shape[2]:
                tensor = candidate
                scheme = to_scheme(tensor, shape, 'cw-support-recombined')
                break
    if scheme is not None:
        report.update({"shape": list(scheme.shape), "rank": scheme.rank, "exact_matrix_identity_verified": True})
        if len(set(scheme.shape)) == 1:
            d = scheme.shape[0]
            report["square_recursion"] = {"dimension": d, "rank": scheme.rank,
                                          "exponent_expression": f"log_{d}({scheme.rank})",
                                          "exponent": log(scheme.rank, d),
                                          "strictly_below_2_75": beats_exponent(scheme.rank, d, 11, 4)}
        report["outcome"] = (f"Verified {scheme.shape} matrix recipe with {scheme.rank} products. "
                             "This is a finite executable identity; the exponent comparison is reported separately.")
        if "search" in report:
            result = "reached" if report["search"]["target_reached"] else "not reached"
            report["outcome"] += f" Requested rank target {report['search']['target_rank']}: {result}."
    report["provenance"] = provenance
    if args.command == "cw-extract" and args.certificate and scheme is None:
        raise ValueError("extraction certificate requires a single supported matrix component; a direct sum is tensor evidence")
    if args.command == "cw-interior" and args.certificate and scheme is None:
        raise ValueError("interior certificate requires a supported square recombination; generic parts remain tensor evidence")
    if args.command == "cw-neighborhood" and args.certificate and scheme is None:
        raise ValueError("neighborhood certificate requires a supported matrix target; generic parts remain tensor evidence")
    if args.command == 'cw-support' and args.certificate and scheme is None:
        raise ValueError('support certificate requires a supported matrix target')
    # Reject an unsupported certificate request before creating a partial output
    # directory. Rectangular certificate creation needs an explicitly balanced
    # cycle; the suite supplies that cycle in its separate integration stage.
    if args.certificate and scheme is not None and len(set(scheme.shape)) != 1:
        raise ValueError("portable square certificate requires a balanced rectangular cycle; use the rectangular API")
    directory = Path(directory) if directory is not None else args.output_root / ("construction_"+datetime.now().strftime("%Y-%m-%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    if scheme is not None:
        save_scheme(scheme, directory / "scheme.json", provenance=provenance)
        report["scheme_sha256"] = sha256((directory / "scheme.json").read_bytes()).hexdigest()
        if args.certificate:
            # Build the plan from the saved document, not its coefficient-only
            # descriptor. The document carries inherited author/license notices,
            # which must remain inside a portable certificate distributed alone.
            write_bundle(Composition.from_spec(str(directory / "scheme.json")), directory / "certificate")
            report["portable_certificate"] = "certificate/certificate.json"
    elif tensor is not None:
        write_json(directory / "tensor.json", tensor_document(tensor))
    if args.command in ("cw-interior", "cw-neighborhood", "cw-support", "cw-batch", "pan-batch"):
        # Unit-axis pieces remain generic exact tensors, with their proved
        # matrix shape and recovery provenance. Do not relax scheme contracts
        # just to label them recursive square algorithms.
        library = {"schema": "exact-recipe-parts-v1", "provenance": provenance,
                   "parts": [{"evidence": description, "tensor": tensor_document(part)}
                             for part, description in interior_parts]}
        write_json(directory / "parts.json", library)
    write_json(directory / "report.json", report)
    write_summary(directory, report)
    return directory, report


def main(argv=None):
    """Dispatch a bounded construction experiment and save exactly checked evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    commands = parser.add_subparsers(dest="command", required=True)
    search_parser = commands.add_parser("search", help="search a finite rational recipe by exact flips and reductions")
    search_parser.add_argument("--size", type=int, default=2, help="classical square starting size if --start is absent")
    search_parser.add_argument("--start", help="verified real rational scheme JSON to improve")
    search_parser.add_argument("--target-rank", type=int)
    search_parser.add_argument("--target-exponent", type=Fraction, help="strict rational target; e.g. 11/4 or 23/10")
    search_parser.add_argument("--steps", type=int, default=1000, help="attempted flips per restart")
    search_parser.add_argument("--restarts", type=int, default=10)
    search_parser.add_argument("--seed", type=int, default=0)
    search_parser.add_argument("--max-height", type=int, default=1, help="maximum coefficient numerator/denominator; raise to explore Q more broadly")
    search_parser.add_argument("--extra-rank", type=int, default=0, help="allow temporary rank increases by exact splits")
    search_parser.add_argument("--split-every", type=int, default=50, help="attempt a split every this many steps when extra rank is allowed")
    search_parser.add_argument("--certificate", action="store_true")
    product_parser = commands.add_parser("product", help="tensor two matrix recipes with correct row-major reindexing")
    product_parser.add_argument("--left", required=True)
    product_parser.add_argument("--right", required=True)
    product_parser.add_argument("--certificate", action="store_true")
    cw_parser = commands.add_parser("cw", help="generate and exactly recover a CW polynomial tensor power")
    cw_parser.add_argument("--q", type=int, default=2)
    cw_parser.add_argument("--power", type=int, default=1)
    extraction_parser = commands.add_parser("cw-extract", help="extract and execute verified independent products from a CW power")
    extraction_parser.add_argument("--q", type=int, default=2)
    extraction_parser.add_argument("--power", type=int, default=1)
    extraction_parser.add_argument("--sector-word", dest="sector_words", action="append", type=parse_sector_word,
                                   help="explicit comma-separated sectors, e.g. 101,110,011; repeat for independent components")
    extraction_parser.add_argument("--certificate", action="store_true", help="certificate when a single extracted component is a supported square base")
    packing_parser = commands.add_parser("cw-pack", help="optimize CW recovery and construct a verified dense matrix base")
    packing_parser.add_argument("--q", type=int, default=2)
    packing_parser.add_argument("--power", type=int, default=2)
    packing_parser.add_argument("--size", type=int, default=2, help="finite square base to cover with component tiles")
    packing_parser.add_argument("--sector-word", dest="sector_words", action="append", type=parse_sector_word)
    packing_parser.add_argument("--certificate", action="store_true")
    coarse_parser = commands.add_parser("cw-coarse", help="merge CW groups, classify all components, and pack verified matrix pieces")
    coarse_parser.add_argument("--q", type=int, default=2)
    coarse_parser.add_argument("--power", type=int, default=2)
    coarse_parser.add_argument("--size", type=int, default=2)
    coarse_parser.add_argument("--certificate", action="store_true")
    interior_parser = commands.add_parser("cw-interior", help="search, recover and recombine reusable restrictions of interior CW components")
    interior_parser.add_argument("--q", type=int, default=2)
    interior_parser.add_argument("--power", type=int, default=2, help="source CW power; each interior component is tried at powers one and two")
    interior_parser.add_argument("--candidate-budget", type=int, default=2048, help="candidate subsets per sector and component power")
    interior_parser.add_argument("--max-parts", type=int, default=2, help="saved search parts per sector/power, plus a known lifted witness")
    interior_parser.add_argument("--max-source-dimension", type=int, default=64)
    interior_parser.add_argument("--max-source-terms", type=int, default=4096)
    interior_parser.add_argument("--certificate", action="store_true")
    neighborhood_parser = commands.add_parser("cw-neighborhood", help="search near verified maps and check joint restrictions with paid recovery")
    neighborhood_parser.add_argument("--q", type=int, default=2)
    neighborhood_parser.add_argument("--power", type=int, default=2)
    neighborhood_parser.add_argument("--seed-budget", dest="candidate_budget", type=int, default=2048)
    neighborhood_parser.add_argument("--candidate-budget", dest="neighborhood_budget", type=int, default=4096,
                                     help="per sector/power; half single-coordinate edits and half pair tests")
    neighborhood_parser.add_argument("--max-witnesses", dest="neighborhood_max_witnesses", type=int, default=64)
    neighborhood_parser.add_argument("--max-parts", type=int, default=2)
    neighborhood_parser.add_argument("--max-source-dimension", type=int, default=64)
    neighborhood_parser.add_argument("--max-source-terms", type=int, default=4096)
    neighborhood_parser.add_argument("--certificate", action="store_true")
    support_parser = commands.add_parser('cw-support', help='search coordinate support, reduce recipes and check cyclic joint selections')
    support_parser.add_argument('--q', type=int, default=2)
    support_parser.add_argument('--power', type=int, default=2)
    support_parser.add_argument('--candidate-budget', dest='support_budget', type=int, default=4096)
    support_parser.add_argument('--seed-budget', dest='candidate_budget', type=int, default=2048)
    support_parser.add_argument('--max-parts', type=int, default=2)
    support_parser.add_argument('--steps', type=int, default=1000)
    support_parser.add_argument('--restarts', type=int, default=10)
    support_parser.add_argument('--search-seeds', dest='support_search_seeds', type=int, nargs='+', default=(0, 1))
    support_parser.add_argument('--max-height', type=int, default=1)
    support_parser.add_argument('--reduction-limit', dest='support_reduction_limit', type=int, default=1)
    support_parser.add_argument('--max-source-dimension', type=int, default=64)
    support_parser.add_argument('--max-source-terms', type=int, default=4096)
    support_parser.add_argument('--certificate', action='store_true')
    commands.add_parser("cw-batch", help="verify and execute recursive batches from the archived rank-seven CW pair")
    pan_parser = commands.add_parser("pan-batch", help="verify and execute Pan's exact two-product aggregation")
    pan_parser.add_argument("--power", type=int, choices=(1, 2), default=2,
                            help="verify and execute the base only (1) or also dimension sixteen (2)")
    replay_parser = commands.add_parser("replay", help="replay a prior report.json independently of random search")
    replay_parser.add_argument("report", type=Path)
    replay_parser.add_argument("--certificate", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "search" and args.target_rank is not None and args.target_exponent is not None:
        parser.error("choose --target-rank or --target-exponent")
    try:
        directory, report = run_experiment(ConstructionRequest(**vars(args)))
        print(report["outcome"])
        print(f"Report: {directory / 'index.html'}")
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


def parse_sector_word(value):
    """Parse explicit CLI sectors without accepting truncated digits or booleans."""
    from .cw_extraction import sector_word
    try:
        return sector_word(tuple(tuple(int(digit) for digit in token) for token in value.split(',')))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
