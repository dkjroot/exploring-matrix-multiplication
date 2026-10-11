"""Import the published rank-48 data without executing the upstream notebook.

Only data are extracted using ast.literal_eval; NumPy is never imported.
The source and derived coefficients both remain in Background Research (ignored).
"""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from matrix_multiplication.alphaevolve import validate_factors

SOURCE = "https://raw.githubusercontent.com/google-deepmind/alphaevolve_results/main/mathematical_results.ipynb"
NOTEBOOK = ROOT / "Background Research" / "alphaevolve-mathematical-results.ipynb"
OUTPUT = ROOT / "Background Research" / "alphaevolve-48.json"


def extract(notebook):
    """Extract literal factors, reject executable expressions, verify the identity."""
    matches = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code" or "decomposition_444 =" not in source:
            continue
        for statement in ast.parse(source).body:
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "decomposition_444"):
                value = statement.value
                if not isinstance(value, ast.Tuple) or len(value.elts) != 3:
                    raise ValueError("expected three literal array factors")
                factors = []
                for call in value.elts:
                    if (not isinstance(call, ast.Call) or len(call.args) != 1
                            or not isinstance(call.func, ast.Attribute)
                            or not isinstance(call.func.value, ast.Name)
                            or call.func.value.id != "np" or call.func.attr != "array"):
                        raise ValueError("expected literal np.array(...) data")
                    array = ast.literal_eval(call.args[0])
                    factor = []
                    for row in array:
                        pairs = []
                        for number in row:
                            number = complex(number)
                            real, imag = 2 * number.real, 2 * number.imag
                            if not (real.is_integer() and imag.is_integer()):
                                raise ValueError("coefficient is not a Gaussian half-integer")
                            pairs.append([int(real), int(imag)])
                        factor.append(pairs)
                    factors.append(factor)
                matches.append(factors)
    if len(matches) != 1:
        raise ValueError("expected exactly one decomposition_444 data assignment")
    validate_factors(matches[0])
    return matches[0]


def main():
    """Import local/downloaded data with attribution, hash, and exact verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=NOTEBOOK)
    parser.add_argument("--download", action="store_true", help="download the public source notebook first")
    args = parser.parse_args()
    if args.download:
        with urlopen(SOURCE, timeout=45) as response:
            content = response.read()
    else:
        content = args.notebook.read_bytes()
    factors = extract(json.loads(content))
    document = {
        "source_url": SOURCE,
        "source_sha256": sha256(content).hexdigest(),
        "attribution": "Google DeepMind, AlphaEvolve results, 2025; see upstream license and citation instructions.",
        "format": "Three 16-by-48 arrays; each coefficient [real, imag] is twice its published value. W indexes column*4+row.",
        "factors_doubled": factors,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if args.download:
        NOTEBOOK.write_bytes(content)
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Imported rank-48 coefficients; all 4096 tensor coefficients verified exactly.")
    print(OUTPUT)


if __name__ == "__main__":
    main()
