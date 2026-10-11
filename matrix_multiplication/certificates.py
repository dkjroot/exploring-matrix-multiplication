"""Build portable certificates from exact schemes and maintained formula stages.

The existing sparse verifier and an independent dense basis checker both run at
creation. Bundles include the exact data, a standalone checker, the mathematical
argument, and file fingerprints. They are reproducible algebraic verification
packages, not Lean proofs or certificates of the Python execution interpreter.
"""

from datetime import datetime
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from . import certificate_checker, strassen, strassen_winograd
from .composition import Composition
from .instrumentation import suspended
from .scheme_io import scheme_document, scheme_from_document
from .schemes import BilinearScheme


class LinearForm:
    """A formal coefficient vector used to extract readable formulas, without data."""

    def __init__(self, coefficients):
        """Retain exact integer coefficients in an immutable vector."""
        self.coefficients = tuple(coefficients)

    def __add__(self, other):
        """Expand formula additions into coefficient-vector additions."""
        return LinearForm(a+b for a, b in zip(self.coefficients, other.coefficients))

    def __sub__(self, other):
        """Expand formula subtraction, preserving intermediate reuse algebraically."""
        return LinearForm(a-b for a, b in zip(self.coefficients, other.coefficients))


def formula_recipe(name):
    """Extract U/V/W from the actual built-in step's linear combinations.

    Input blocks contain four-coordinate linear forms. The product callback
    records those input forms and returns a seven-coordinate formal product.
    Output additions therefore expose W without hard-coding a second formula.
    This extraction establishes the algebra of the step; it does not establish
    the step's runtime operation counts or general Python-program correctness.
    """
    module = {"strassen": strassen, "strassen-winograd": strassen_winograd}[name]
    blocks = [[[LinearForm(int(i == j) for j in range(4))]] for i in range(4)]
    left, right = [], []

    def product(a, b):
        """Record one bilinear product and return its formal output basis vector."""
        index = len(left)
        if index >= 7:
            raise ValueError("built-in formula exceeded its seven-product declaration")
        left.append(tuple((i, c) for i, c in enumerate(a[0][0].coefficients) if c))
        right.append(tuple((i, c) for i, c in enumerate(b[0][0].coefficients) if c))
        return [[LinearForm(int(i == index) for i in range(7))]]

    # Algebra extraction is setup, excluded even if the caller has an observer.
    with suspended():
        outputs = module.step(blocks, blocks, product)
    if len(left) != 7:
        raise ValueError("built-in formula did not use seven products")
    output = tuple(tuple((i, block[0][0].coefficients[t])
                         for i, block in enumerate(outputs) if block[0][0].coefficients[t])
                   for t in range(7))
    recipe = scheme_document(BilinearScheme(name, 2, tuple(left), tuple(right), output))
    recipe["provenance"] = {"construction": "symbolic extraction from maintained step function",
                            "source_file": Path(module.__file__).name,
                            "source_sha256": sha256(Path(module.__file__).read_bytes()).hexdigest(),
                            "implementation_license": "GPL-3.0; see repository LICENSE"}
    return recipe


def create_certificate(plan):
    """Run both exact checks and bind the ordered stage cycle to its recipes."""
    stages = []
    for stage, document in zip(plan.stages, plan.documents):
        recipe = formula_recipe(stage.name) if document is None else document
        # First verifier expands sparse columns. Second evaluates all basis pairs.
        scheme_from_document(recipe)
        evidence = certificate_checker.check_recipe(recipe)
        stages.append({"recipe": recipe, "recipe_sha256": certificate_checker.fingerprint(recipe),
                       "evidence": evidence})
    radix, rank = plan.model()["base_dimension"], prod(s.rank for s in plan.stages)
    certificate = {"schema": "matrix-multiplication-certificate-v1", "field": "Q(i)",
                   "conventions": "row-major; output divided by recipe divisor",
                   "stages": stages,
                   "cycle": {"radix": radix, "rank": rank, "repeats": True,
                             "exponent_expression": f"log_{radix}({rank})",
                             "recurrence": f"T(n) = {rank} T(n/{radix}) + O(n^2)"}}
    certificate_checker.check_certificate(certificate)
    return certificate


def write_bundle(plan, destination=None):
    """Write a fresh portable directory; refuse to overwrite an existing package."""
    if destination is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        destination = Path(__file__).resolve().parents[1] / "outputs" / f"certificate_{stamp}"
    destination = Path(destination)
    certificate = create_certificate(plan)
    checker = Path(certificate_checker.__file__).read_bytes()
    argument = (Path(__file__).resolve().parents[1] / "CERTIFICATES.md").read_bytes()
    license_text = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
    destination.mkdir(parents=True, exist_ok=False)
    files = {"certificate.json": (json.dumps(certificate, indent=2, ensure_ascii=True,
                                              allow_nan=False) + "\n").encode("utf-8"),
             "check_certificate.py": checker, "README.md": argument, "LICENSE": license_text}
    for name, content in files.items():
        (destination / name).write_bytes(content)
    checksums = {name: sha256(content).hexdigest() for name, content in files.items()}
    (destination / "file-sha256.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv=None):
    """Create a portable package or independently recheck an existing certificate."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    create = command.add_parser("create")
    create.add_argument("--composition", default="strassen,strassen-winograd,alphaevolve")
    create.add_argument("--output", type=Path)
    verify = command.add_parser("verify")
    verify.add_argument("certificate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            path = write_bundle(Composition.from_spec(args.composition), args.output)
            print(f"CERTIFICATE: {(path / 'certificate.json').resolve()}")
            print(f"Recheck independently: python \"{path.resolve() / 'check_certificate.py'}\"")
        else:
            certificate_checker.check_certificate(certificate_checker.read_json(args.certificate))
            print("PASS: exact identities, fingerprints, and cycle declarations rechecked")
    except (ValueError, TypeError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
