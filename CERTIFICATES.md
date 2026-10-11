# Exact Python verification certificates

This package is a reproducible algebraic verification, not a Lean proof. Run:

```
python check_certificate.py
```

The bundled checker uses only Python's standard library. It does not import the
multiplier, its scalar primitives, its scheme parser, or its existing verifier.
It recomputes the identities rather than trusting a saved success flag. Any
failure exits nonzero. You may also supply a different certificate JSON path.

## What is being checked?

A radix-d recipe describes d-by-d multiplication with r products. Each product
multiplies a fixed linear combination of left entries by a fixed linear
combination of right entries. Fixed linear combinations of those products form
the output; a positive integer divisor removes any common coefficient scale.
All indices are row-major. Coefficients are exact rational real/imaginary pairs.
The declared field Q(i) means numbers a+bi where a and b are rational.

Version-2 recipes replace a square radix with dimensions [m,k,n]: they multiply
m-by-k by k-by-n and produce m-by-n. Both verifiers check (m*k*n) squared
coefficient relationships, using m*k times k*n basis pairs. Smirnov's [3,3,6]
recipe therefore requires 2,916 checks. Its cyclic orientations are checked
separately, including every transposed output-index transformation.

There are d squared possible left entries, d squared right entries, and d squared
output entries. Thus there are d to the sixth coefficient relationships. The
checker tests every one, including those whose expected value is zero. It does
not sample random input matrices or accept a floating-point tolerance.

## Why do finitely many checks establish the whole identity?

Let E_ij be the matrix with a single one at row i, column j. Every matrix A is a
sum of A_ij E_ij. The recipe is bilinear: it is linear in A with B fixed, and
linear in B with A fixed. This follows from its allowed sum/product/sum form.
Expanding both matrices therefore expresses the recipe on A and B as a sum of
A_ij B_kl times the recipe on E_ij and E_kl.

Ordinary multiplication is also bilinear. E_ij E_kl equals E_il if j=k, and the
zero matrix otherwise. If the recipe agrees on every basis pair, each term in
the two expansions agrees, hence the result agrees for every A and B over Q(i).
The same polynomial identity extends to complex mathematical inputs through the
inclusion of Q(i) in the complex numbers. Real inputs can be embedded as complex
numbers with zero imaginary component; their product is real even when the recipe
uses complex intermediate values. This does not promise that
finite-precision floating evaluation is exact or numerically stable.

The original library checker expands sparse tensor contributions. This checker
instead constructs dense coefficient columns, evaluates every basis-input pair,
and checks every output against the matrix-unit product. These are independently
written implementations of equivalent finite criteria. Both use exact Fraction
arithmetic; their correctness still depends on Python and the checker code.

## Why can these recipes be composed recursively?

The polynomial expansion preserves the order left-block times right-block.
Coefficients act as central scalars, so the identity also applies to square
matrix blocks, even though block products themselves need not commute.

For an induction on recursion depth, classical multiplication is the correct
leaf case. At a non-leaf node, assume every child call correctly multiplies its
two blocks. Substituting those correct block products into the verified base
identity gives the correct parent product. Repeating this argument proves the
mathematical recursive construction correct at every finite depth. Zero-padding
and cropping preserve the requested rectangular product.

A cycle with stage radices d1,...,dk and ranks r1,...,rk has radix D equal to
their product and rank R equal to their product. The checker recomputes these
integers from the ordered verified recipes. Repeating the fixed cycle gives
T(n)=R T(n/D)+O(n squared), for fixed coefficients and unit-cost field arithmetic.
The exponent expression is log_D(R); if R=D squared, combination overhead can
give an additional log(n) factor. Fixed cutoffs affect constants. Actual padded
counts, interpreter correctness, integer bit complexity, and numerical stability
need separate checks; this certificate does not certify those properties.

For rectangular stages, multiply the row, inner, and column dimensions separately
over the whole cycle. All three products must equal the same D before this
checker accepts a square recurrence. Smirnov's orientations [3,3,6], [3,6,3],
[6,3,3] give [54,54,54] and 40 cubed = 64,000 products, yielding log_54(64000).
No exponent is claimed for an unbalanced rectangular cycle.

## Fingerprints and reproducibility

Recipe fingerprints use SHA-256 of JSON with sorted keys, compact separators,
ASCII escapes, and no nonfinite numbers. They include provenance. Formatting
changes alone do not affect these semantic fingerprints. The checker compares
stored fingerprints and evidence with independent recomputation. It also checks
the cycle declaration, including order-bound recipe records and exact integers.

file-sha256.json records byte fingerprints of the recipe package, this explanation,
and the standalone checker. They aid external comparison; they are not signatures
or protection against someone replacing both files and their checksums. The
checker is readable source: review it, and compare fingerprints against a trusted
copy when establishing provenance. Its reported certificate fingerprint permits
exact comparison between independent runs.

Built-in Strassen/Winograd coefficients are symbolically extracted from this
repository's actual step formulas. Their source-file fingerprints are recorded;
these associate the recipe with source, but do not formally verify that program.
Imported recipe provenance is retained, not independently authenticated.

## Licensing

The checker and this explanation are supplied under this repository's GPL-3.0
license. Third-party coefficient notices remain in each recipe's provenance;
those materials may have different conditions. Generated packages live under
ignored outputs/ by default. Before publishing one, retain the repository LICENSE
and required third-party licenses/notices. A correctness check grants no license.
