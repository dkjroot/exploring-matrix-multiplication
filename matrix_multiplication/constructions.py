"""CLI for constructing, searching, replaying, and exporting exact recipes.

Run ``python -m matrix_multiplication.constructions --help``. Each construction
writes a fresh dated directory below outputs, with data and an offline HTML
summary. Matrix recipes enter the same composition, measurement, and portable
certificate paths as imported schemes. CW artifacts remain labelled tensors,
not usable matrix algorithms. No third-party implementation executes here.
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
    report: Path = None


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
                 '<p>The curve shows the replay path to the best recipe, not unsuccessful walk branches. '
                 'Attempt counts below include those branches and restarts.</p>')
    display = {k: v for k, v in report.items() if k != "starting_recipe"}
    if "search" in display:
        display["search"] = {k: v for k, v in evidence.items() if k != "events"}
    links = ' · '.join(f'<a href="{escape(path.name)}">{escape(path.name)}</a>'
                       for path in sorted(directory.glob("*.json")))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Exact construction experiment</title><style>'
            'body{font:18px/1.6 system-ui;max-width:1000px;margin:2rem auto;padding:1rem;color:#17202a;background:#fff}'
            'pre{font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem}'
            'svg{width:100%;max-width:760px}a{color:#005a8d}</style>'
            '<h1>Exact construction experiment</h1>'
            f'<p>{escape(report["outcome"])}</p><p>{links}</p>{curve}'
            '<p>Construction/search attempts are setup work. To measure the resulting matrix algorithm, '
            'pass scheme.json to explore.py or sweep.py using their composition option.</p>'
            f'<pre>{escape(json.dumps(display, indent=2))}</pre></html>')
    (directory / "index.html").write_text(html, encoding="utf-8")


def run_experiment(args, directory=None):
    """Execute a typed construction request; return its directory and exact evidence.

    The CLI and combined sweep share this function. An explicit directory makes
    suite artifacts predictably named under one dated run, without nested dates.
    Existing directories are refused; callers retain earlier experiment output.
    """
    if args.command not in ("search", "product", "replay", "cw"):
        raise ValueError("unknown construction experiment")
    if args.command == "search" and args.target_rank is not None and args.target_exponent is not None:
        raise ValueError("choose a rank or exponent target")
    provenance = {"implementation": "independent rational tensor construction primitives",
                  "implementation_license": "GPL-3.0; repository LICENSE"}
    report = {"schema": "construction-experiment-v1", "command": args.command,
              "coefficient_field": "Q", "limitations": "Finite exact recipes only; no new exponent bound follows from a search attempt."}
    scheme = None
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
    else:
        tensor, evidence = recover_cw(args.q, args.power)
        report.update({"cw": evidence, "outcome": "Polynomial CW tensor recovered and exactly verified. This is not yet a matrix multiplier."})
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
    else:
        write_json(directory / "tensor.json", tensor_document(tensor))
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


if __name__ == "__main__":
    raise SystemExit(main())
