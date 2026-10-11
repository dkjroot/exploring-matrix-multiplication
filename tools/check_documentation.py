"""Check that maintained Python modules, classes, and functions have docstrings.

This lightweight AST check supports the project's readable-reference goal. It
does not judge prose quality or import/execute sources. Downloaded research code
and dependency directories are deliberately outside the maintained source set.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_file(path):
    """Return missing documentation locations for one parsed Python source."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures = []
    if ast.get_docstring(tree) is None:
        failures.append(f"{path.relative_to(ROOT)}:1: missing module header docstring")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is None:
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: missing docstring: {node.name}")
    return failures


def main():
    """Inspect application, library, tool, and test sources; return a CLI status."""
    paths = [ROOT / "explore.py", ROOT / "sweep.py"]
    for directory in ("matrix_multiplication", "tools", "tests"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    failures = [failure for path in paths for failure in check_file(path)]
    for failure in failures:
        print(failure)
    if not failures:
        print(f"Documentation coverage passed for {len(paths)} Python files.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
