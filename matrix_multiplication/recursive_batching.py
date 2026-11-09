"""Execute equal-size independent matrix products by recursively batching terms.

The input recipe computes a direct sum of square matrix products. A slot in a
child call is one independent matrix pair. Outer bilinear terms fill those
slots in order, and each child's results return to those same terms. See
Solomonik and Bläser, arXiv:2610.26563v2, Appendix A, for the divisible case.
"""

from dataclasses import dataclass
from fractions import Fraction

from .common import accumulate_scaled, join, linear_combination, split, validate, zeros
from .instrumentation import count, move_values, scope, session
from .scalars import multiply
from .tensors import Tensor


def verify_batch_tensor(tensor, size, lanes):
    """Compare the complete coefficient dictionary with independent dot products.

    Coordinates are lane-major, then row-major within a lane. Dictionary
    equality also checks every zero coefficient, including all mixed lanes.
    """
    width = lanes*size*size
    expected = {}
    for lane in range(lanes):
        offset = lane*size*size
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    expected[offset+i*size+k, offset+k*size+j, offset+i*size+j] = 1
    if tensor.dimensions != (width,)*3 or tensor.expand() != expected:
        raise ValueError("recipe fails the complete independent square-product identity")
    return {"dimension": size, "coefficient_checks": width**3, "nonzero_coefficients": len(expected)}


@dataclass(frozen=True)
class RecursiveBatch:
    """A checked rational square-sum recipe with a fixed, divisible schedule."""

    tensor: Tensor
    radix: int
    lanes: int

    def __post_init__(self):
        """Verify the identity once, before any recursive execution."""
        if type(self.radix) is not int or self.radix < 2 or type(self.lanes) is not int or self.lanes < 1:
            raise ValueError("batch requires an integer radix at least two and a positive lane count")
        if self.tensor.rank % self.lanes:
            raise ValueError("recipe product count must divide evenly into child calls")
        verify_batch_tensor(self.tensor, self.radix, self.lanes)

    @property
    def batches(self):
        """Map each child call and slot to its parent term, in evaluation order."""
        return tuple(tuple(range(first, first+self.lanes))
                     for first in range(0, self.tensor.rank, self.lanes))

    def depth(self, size):
        """Return the number of levels for a positive exact power of the radix."""
        if type(size) is not int or size < 1:
            raise ValueError("batch dimension must be a positive integer")
        depth = 0
        while size > 1:
            size, remainder = divmod(size, self.radix)
            if remainder:
                raise ValueError("batch dimension must be a power of the radix")
            depth += 1
        return depth

    def budget(self, size):
        """Count scalar leaves and sparse add/scale work over all recursive nodes."""
        depth = self.depth(size)
        fanout = self.tensor.rank//self.lanes
        entries = sum(len(factor) for term in self.tensor.terms for factor in term)
        combinations = sum(fanout**level * entries * (size//self.radix**(level+1))**2
                           for level in range(depth))
        return {"field_multiplications": self.lanes*fanout**depth,
                "field_additions": combinations, "coefficient_multiplications": combinations}

    def multiply(self, left, right, *, instrumentation=None):
        """Execute equally sized batches of square integer or rational matrices."""
        left, right = tuple(left), tuple(right)
        if len(left) != self.lanes or len(right) != self.lanes:
            raise ValueError("provide one left and right matrix per batch slot")
        size = len(left[0])
        self.depth(size)
        for a, b in zip(left, right):
            if validate(a, b) != (size,)*3:
                raise ValueError("batch matrices must be square and share one dimension")
            if any(type(value) not in (int, Fraction) for matrix in (a, b) for row in matrix for value in row):
                raise ValueError("batch entries must be exact integers or fractions")
        with session(instrumentation):
            return self._node(left, right, self.batches, 0)

    def _node(self, left, right, batches, depth):
        """Run one node and scatter each child result through its parent output factor."""
        size = len(left[0])
        with scope("recursive_batch.node", depth=depth, size=size, lanes=self.lanes):
            count("recursive_nodes")
            if size == 1:
                count("classical_leaves")
                count("matrix_cells_allocated", self.lanes)
                output = tuple([[multiply(a[0][0], b[0][0])]] for a, b in zip(left, right))
                move_values((matrix[0][0] for matrix in output), "write")
                return output

            radix = self.radix
            blocks_a = [block for matrix in left for block in split(matrix, radix)]
            blocks_b = [block for matrix in right for block in split(matrix, radix)]
            output = [zeros(size//radix) for _ in blocks_a]
            for batch in batches:
                with scope("recursive_batch.child", parent_terms=batch):
                    terms = tuple(self.tensor.terms[index] for index in batch)
                    a = tuple(linear_combination(blocks_a, u) for u, _, _ in terms)
                    b = tuple(linear_combination(blocks_b, v) for _, v, _ in terms)
                    child_results = self._node(a, b, batches, depth+1)
                    for result, (_, _, w) in zip(child_results, terms):
                        for coordinate, coefficient in w:
                            accumulate_scaled(output[coordinate], result, coefficient)
            width = radix*radix
            return tuple(join(output[start:start+width], radix)
                         for start in range(0, len(output), width))

    def second_level_tensor(self):
        """Expand the call schedule into sparse scalar factors for a full identity check.

        An inner factor names a child slot and a block entry. Replace the slot
        with the corresponding outer term's factor, then interleave block and
        within-block row and column indices to recover ordinary matrix order.
        Each child term contributes once, regardless of how many slots it uses.
        """
        radix = self.radix
        width = radix*radix

        def lift(inner, batch, leg):
            """Substitute one child's sparse factor into a lane-major outer factor."""
            for index, coefficient in inner:
                slot, entry = divmod(index, width)
                inner_row, inner_col = divmod(entry, radix)
                for outer, value in self.tensor.terms[batch[slot]][leg]:
                    lane, block = divmod(outer, width)
                    block_row, block_col = divmod(block, radix)
                    row = block_row*radix+inner_row
                    col = block_col*radix+inner_col
                    yield lane*width**2 + row*width + col, coefficient*value

        terms = tuple(tuple(tuple(lift(factor, batch, leg)) for leg, factor in enumerate(term))
                      for batch in self.batches for term in self.tensor.terms)
        return Tensor((self.lanes*width**2,)*3, terms)
