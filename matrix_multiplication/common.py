"""Reusable matrix/block layer between scalar arithmetic and algorithm formulas.

The flow is validate -> pad/encode -> recursive nodes -> block steps -> scalar
leaves -> join/crop/decode. Structural work is recorded separately from arithmetic.
Copying blocks favors readability; it changes constants and memory, not exponents.
Only this layer owns traversal, padding, and elementary matrix loops.
"""

from math import isfinite
from numbers import Rational

from . import scalars
from .instrumentation import count, scope, session


def validate(a, b, cutoff=1):
    """Return (rows, inner, columns) after checking the shared public contract.

    Lists/tuples must be rectangular and nonempty, dimensions compatible, and
    values finite. Rational validation avoids converting huge integers to floats.
    This control/input work is outside the scalar arithmetic operation budget.
    """
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("cutoff must be a positive integer")
    shapes = []
    # Empty list representations cannot tell us their intended column count.
    for name, matrix in (("a", a), ("b", b)):
        if not isinstance(matrix, (list, tuple)) or not matrix:
            raise ValueError(f"{name} must be a nonempty list or tuple of rows")
        width = None
        for row in matrix:
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(f"{name} must have nonempty list or tuple rows")
            if width is None:
                width = len(row)
            if len(row) != width:
                raise ValueError(f"{name} must be rectangular")
            for value in row:
                # bool is an int subclass, but accepting it would hide accidental
                # predicates in numerical input. Exact Rational values are finite.
                if isinstance(value, bool) or not isinstance(value, (Rational, float, complex)):
                    raise TypeError("entries must be integers, rational numbers, floats, or complex numbers")
                if isinstance(value, float) and not isfinite(value):
                    raise ValueError("entries must be finite")
                if isinstance(value, complex) and not (isfinite(value.real) and isfinite(value.imag)):
                    raise ValueError("entries must be finite")
        shapes.append((len(matrix), width))
    if shapes[0][1] != shapes[1][0]:
        raise ValueError("a's column count must equal b's row count")
    return shapes[0][0], shapes[0][1], shapes[1][1]


def classical(a, b):
    """Evaluate row-by-column dots using the same scalar primitives as recursion.

    A length-k dot performs k products and k additions, including the initial
    addition to zero. Explicit accumulation makes the schedule consistent across
    algorithms, rather than relying on version-dependent built-in sum behavior.
    """
    with scope("matrix.classical", rows=len(a), inner=len(b), columns=len(b[0])):
        result = zeros(len(a), len(b[0]))
        for i, row in enumerate(a):
            for j in range(len(b[0])):
                result[i][j] = dot(row, column(b, j))
        return result


def column(matrix, index):
    """Copy one column into a vector; expose this leaf-kernel data movement."""
    with scope("vector.column", length=len(matrix), index=index):
        count("vector_cells_allocated", len(matrix))
        count("vector_cells_copied", len(matrix))
        return [row[index] for row in matrix]


def dot(left, right):
    """Accumulate a dot product, exposing every product and addition in order."""
    with scope("vector.dot", length=len(left)):
        total = 0
        for x, y in zip(left, right):
            total = scalars.add(total, scalars.multiply(x, y))
        return total


def zeros(rows, columns=None):
    """Allocate independent zero-filled rows; count logical matrix cells, not bytes."""
    columns = rows if columns is None else columns
    with scope("matrix.zeros", rows=rows, columns=columns):
        count("matrix_cells_allocated", rows * columns)
        return [[0 for _ in range(columns)] for _ in range(rows)]


def padded(matrix, size):
    """Copy a rectangular input into a zero square without mutating its rows."""
    with scope("matrix.pad", source_rows=len(matrix), source_columns=len(matrix[0]), size=size):
        result = zeros(size)
        for i, row in enumerate(matrix):
            result[i][:len(row)] = row
            count("matrix_cells_copied", len(row))
        return result


def padded_size(rows, inner, columns, radix):
    """Choose a radix power covering all dimensions, so splitting stays exact."""
    size = 1
    while size < max(rows, inner, columns):
        size *= radix
    return size


def crop(matrix, rows, columns):
    """Copy only the requested rectangle out of a padded product."""
    with scope("matrix.crop", rows=rows, columns=columns):
        count("matrix_cells_allocated", rows * columns)
        count("matrix_cells_copied", rows * columns)
        return [row[:columns] for row in matrix[:rows]]


def map_entries(matrix, function, *, name="matrix.map"):
    """Apply a scalar conversion/normalization while keeping the matrix loop shared."""
    with scope(name, rows=len(matrix), columns=len(matrix[0])):
        count("matrix_cells_allocated", len(matrix) * len(matrix[0]))
        return [[function(value) for value in row] for row in matrix]


def recursive_product(a, b, cutoff, radix, kernel, *, name="recursive",
                      instrumentation=None, encode=None, decode=None):
    """Run the shared public contract around an algorithm's square kernel.

    Encoding/decoding lets a scheme choose an exact scalar domain independently
    of traversal. Observation ends before independent correctness verification.
    """
    with session(instrumentation), scope(f"{name}.multiply", radix=radix, cutoff=cutoff):
        rows, inner, columns = validate(a, b, cutoff)
        size = padded_size(rows, inner, columns, radix)
        count("input_cells", rows * inner + inner * columns)
        count("padded_input_cells", 2 * size * size)
        left, right = padded(a, size), padded(b, size)
        if encode is not None:
            left = map_entries(left, encode, name="scalar.encode_matrix")
            right = map_entries(right, encode, name="scalar.encode_matrix")
        result = crop(kernel(left, right, cutoff), rows, columns)
        return map_entries(result, decode, name="scalar.decode_matrix") if decode else result


def recursive_kernel(a, b, cutoff, radix, step, *, name, depth=0):
    """Traverse recursion once for all schemes; ``step`` supplies only its formula.

    ``step(a_blocks, b_blocks, product)`` forms input combinations, calls the
    supplied child product, and returns output blocks in row-major order. The
    number of calls is chosen by the scheme (7 or 48), never by instrumentation.
    """
    with scope(f"{name}.node", depth=depth, size=len(a), cutoff=cutoff):
        count("recursive_nodes")
        if len(a) <= cutoff:
            count("classical_leaves")
            return classical(a, b)

        def product(left, right):
            """Descend one level with the same scheme and incremented trace depth."""
            return recursive_kernel(left, right, cutoff, radix, step, name=name, depth=depth + 1)

        blocks = step(split(a, radix), split(b, radix), product)
        return join(blocks, radix)


def add(a, b):
    """Return an elementwise block sum without changing either input block."""
    return elementwise(a, b, scalars.add, name="block.add")


def subtract(a, b):
    """Return an elementwise block difference using observable scalar subtraction."""
    return elementwise(a, b, scalars.subtract, name="block.subtract")


def elementwise(a, b, function, *, name):
    """Share the shape-preserving loop for block arithmetic on validated blocks."""
    with scope(name, rows=len(a), columns=len(a[0])):
        count("matrix_cells_allocated", len(a) * len(a[0]))
        return [[function(x, y) for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def accumulate_scaled(target, source, coefficient):
    """Add a scaled block into fresh working storage, exposing scale and add costs.

    The target belongs to a linear combination or output assembly, never to a
    caller's input. This is the shared loop for sparse U/V and W evaluation.
    """
    with scope("block.accumulate_scaled", rows=len(source), columns=len(source[0])):
        for i, row in enumerate(source):
            for j, value in enumerate(row):
                target[i][j] = scalars.add(target[i][j], scalars.scale(value, coefficient))


def linear_combination(blocks, terms):
    """Evaluate sparse (block_index, coefficient) terms in their published order."""
    with scope("block.linear_combination", terms=len(terms), size=len(blocks[0])):
        result = zeros(len(blocks[0]))
        for index, coefficient in terms:
            accumulate_scaled(result, blocks[index], coefficient)
        return result


def split(matrix, radix=2):
    """Copy a square into radix-squared equal blocks in row-major order."""
    with scope("block.split", size=len(matrix), radix=radix):
        size = len(matrix) // radix
        count("matrix_cells_allocated", len(matrix) ** 2)
        count("matrix_cells_copied", len(matrix) ** 2)
        return [[row[j * size:(j + 1) * size]
                 for row in matrix[i * size:(i + 1) * size]]
                for i in range(radix) for j in range(radix)]


def join(blocks, radix=2):
    """Reverse split, copying row-major output blocks into one fresh square."""
    with scope("block.join", block_size=len(blocks[0]), radix=radix):
        size = len(blocks[0])
        count("matrix_cells_allocated", (radix * size) ** 2)
        count("matrix_cells_copied", (radix * size) ** 2)
        return [[value for j in range(radix) for value in blocks[i * radix + j][r]]
                for i in range(radix) for r in range(size)]
