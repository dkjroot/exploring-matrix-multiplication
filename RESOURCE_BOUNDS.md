# Bounds beyond executable dimensions

The operand and word-work sliders answer a bounded-input question: how large
could these quantities be for **any real integer inputs whose entries have
magnitude at most 4**, with this fixed verified recipe, cutoff and size policy?
They do not estimate the particular random input that a future execution would
receive. Saved measurements stay separate and are shown alone initially.

`resource_bounds.py` evaluates the maintained formulas on abstract entries.
An entry records a nonnegative exact rational magnitude M and a positive integer
D such that its reduced denominator divides D. Thus its stored numerator has
magnitude at most floor(M D). Addition and subtraction use magnitude M1+M2
and denominator lcm(D1,D2). Multiplication uses M1 M2 and D1 D2. These rules
follow from the triangle inequality and denominator divisibility. They remain
valid when actual fractions cancel or actual values are zero; they do not assume
that favourable cancellation will occur. Coefficients are exact constants.

For example, classical multiplication of all-four inputs has products 16 and
dot-product prefixes 16, 32, ..., 16n. The operand bound includes the prefix
16(n−1), while the result bound includes 16n. For recursive formulas, coefficient
combinations can grow more quickly than the final result. The model follows
those intermediate operations, rather than substituting the final dot-product
bound for every intermediate value.

Each block is represented by one abstract entry and its actual block area.
Local arithmetic counts are multiplied by that area. All rank children are
enclosed by the maximum input magnitude and the least common multiple of their
denominators. One recursive evaluation of that shared enclosure bounds every
child, and its work is multiplied by the rank. This sacrifices tightness to
avoid traversing an exponential tree. The full verified identity guarantees
that a returned block entry is a dot product; this tightens the enclosure used
by its parent. Final normalizations likewise use the denominator guaranteed by
the complete identity. The real-component schedule for exact complex arithmetic
is evaluated explicitly, including literal zero imaginary components.

Classical leaves and residual fringe terms use the maximum possible dot prefix
at the original inner dimension. Padding follows the same padded dimensions as
execution. Consequently these bounds apply to both supported traversal policies.
No requested matrix is allocated; model calculations are excluded from any
active instrumentation session.

Magnitude bounds give numerator and denominator bit-width bounds by exact integer
bit length. The existing 64-bit model is monotone in those widths: multiplication
uses word-pair products, and fractional addition includes cross-products and a
numerator sum. A denominator bound D greater than one permits both integral and
fractional values; counting possible denominator work is conservative. The
universal denominator can be much larger than actual reduced denominators, so
fractional recipes can have especially loose bounds. GCD, cancellation, division,
carry propagation, allocation and Python's choice of integer algorithm remain
outside this declared word model. The graphs do not bound hardware instructions
or execution time.

Each point is a bound at its saved integer dimension. Four subdivisions per
doubling supplement slider boundaries, stage-prefix neighbours and all execution dimensions; connecting
segments are visual aids and do not prove bounds between those dimensions.
Authoritative numbers remain arbitrary-precision integers in JSON and decimal
strings in the browser; only plotted coordinates are approximate.

Every successful sweep sample is checked against its operand, result, denominator
and both word-work bounds. The abstract scalar-call total must also equal the
captured total. A mismatch fails that sample and retains the evidence. Tests
add exhaustive signed 2×2 examples, large integers crossing 64-bit boundaries,
both size policies, mixed recipes and fractional/complex constructions. These
checks support implementation correctness; the inequalities above justify the
unexecuted dimensions.

Measured Python allocation peaks deliberately have no extension. A logical
matrix-entry count does not establish a bound on Python objects, allocator
behaviour, retained instrumentation data, setup or cold caches. None of those
measured bytes are extrapolated from these arithmetic bounds.
