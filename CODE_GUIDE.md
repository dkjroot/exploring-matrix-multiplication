# Reading the reference implementation

The project has two related jobs. One is to execute a known matrix multiplication
algorithm and observe every arithmetic primitive. The other is to construct new
finite identities and check whether they can become matrix multiplication
algorithms. Construction work and multiplier work have different meanings, so
they deliberately have separate code and evidence.

## Start with an actual multiplication

Read `matrix_multiplication/naive.py`, then `strassen.py` and
`strassen_winograd.py`. Each public function has the same interface. A `step`
function describes the formula at one level; its `product` argument supplies a
child multiplication. The formula chooses the child-product count. Traversal
chooses the shapes and stopping condition.

Continue with `common.py`, which owns input validation, padding, copying,
recursion, splitting, joining, and cropping. Its elementwise block operations call
`scalars.py`, where field operations become observable component operations.
`instrumentation.py` observes that execution using a context-local observer.

```mermaid
flowchart TD
    A[Public multiply call] --> B[Validate and pad inputs]
    B --> C[Recursive node]
    C --> D{All dimensions at most cutoff?}
    D -->|Yes| E[Classical dot products]
    D -->|No| F[Split into blocks]
    F --> G[Algorithm formula or sparse scheme]
    G -->|Child product calls| C
    G --> H[Join output blocks]
    H --> I[Crop and decode result]
    E --> I
    G --> J[Shared block and scalar primitives]
    E --> J
    J --> K[Counters, profiles, bounded traces]
```

For example, Strassen multiplying two 4×4 matrices with cutoff 1 makes seven
2×2 child calls. Each makes seven 1×1 leaf calls, so 49 data multiplications occur.
With cutoff 2, those seven children instead use the classical loop, which needs
eight multiplications per child: 56 total. Additions are counted separately,
including each classical product's accumulation into the running sum.

Padding and cutoffs explain why a finite graph has steps and plateaus rather than
following one smooth power law. They do not change the family's asymptotic
exponent when the cutoff remains fixed as inputs grow.

## Follow coefficients through execution and checking

`schemes.py` describes a bilinear recipe using three sparse tables:

- **U / left:** coefficients combining left input blocks for each child product.
- **V / right:** coefficients combining right input blocks for that product.
- **W / output:** coefficients distributing its result to output blocks.

Each table has one column per product. A term `(index, coefficient)` selects a
row-major block and scales it. A common divisor, if present, normalizes the output
once per node. The generic evaluator performs the literal scale/add schedule,
including unit coefficients, so its operation count can be predicted exactly.

`scheme_io.py` reads data without executing imported code. It checks the complete
identity with exact rational complex pairs. Sparse expansion adds every
`U_i,t * V_j,t * W_k,t` contribution, including cancellations, then compares the
result with the matrix multiplication tensor. Both unexpected nonzeros and
missing required entries are errors.

`composition.py` selects square stages by depth. `rectangular.py` handles balanced
rectangular cycles and the trace-symmetry rotation used by `smirnov.py`. Each
stage has a fixed product count. Multiplying the cycle's product counts and
dimension contractions gives its repeating recurrence without storing a gigantic
combined coefficient tensor.

`certificates.py` extracts the maintained Strassen formulas symbolically and
exports portable evidence. `certificate_checker.py` intentionally stays separate:
it imports only the standard library and evaluates every matrix-unit input pair.
The two checks share the mathematical specification, but use different evaluation
procedures. Neither claims to prove the Python interpreter or universal
floating-point accuracy.

## Separate observations from predictions

`verification.py` computes an independent classical answer with `Fraction`
arithmetic. Rational inputs require exact equality. Floating inputs are compared
with an explicitly recorded tolerance against their exact binary-input answer.
Verification does not use the observed scalar primitives.

`work_sweep.py` executes requested sizes and cutoffs with counters enabled and
primitive clocks disabled. Its predictions independently count the declared
recursion schedule. Its fitted slopes describe the finite observed range; they
do not prove an asymptotic exponent. Equivalent cutoffs can reuse a measurement
only when the known traversal has identical padding, ordered stages, and leaves.
Every reused row records its source cutoff.

`experiments.py` supplies optional elapsed-time and Python-allocation diagnostics.
Those are secondary to the arithmetic objective. `reporting.py` renders original
JSON evidence as offline HTML and SVG, labels both axes, and keeps predictions,
observations, and earlier diagnostic pairs distinct.

## Read the construction machinery next

`tensors.py` represents a sum of rational rank-one terms. Its `rank` property is
the number of terms in that particular decomposition, an upper bound on minimum
rank. A generic tensor is not necessarily matrix multiplication. `to_scheme`
must check that identity before it enters a multiplier.

The module supplies tensor products, powers, direct sums, and linear restrictions.
`matrix_product` additionally permutes paired coordinates into matrix row-major
order. That permutation is essential: pairing two flattened matrices does not
automatically group the new rows before the new columns.

`decomposition_search.py` searches exact algebraic transformations. Flips preserve
a tensor by cancelling two cross terms. Reductions use a verified dependence
between two-factor columns to remove a product. Optional splits temporarily
increase rank. Each accepted move can be replayed without rerunning randomness;
the final recipe is checked completely. Bounded unsuccessful walks establish no
impossibility result.

`degenerations.py` builds the published CW polynomial family independently and
uses exact interpolation to recover its leading tensor coefficient. Recovery
explicitly pays its finite product cost. A smaller border-rank bound is a
different mathematical statement and is not labelled an executable multiplier.
See `CONSTRUCTIONS.md` for examples and remaining extraction/packing work.

`constructions.py` exposes these operations through one shared typed request API
and a CLI. `research_suite.py` orchestrates that API, certificates, diagnostics,
and the size sweep. Candidate integration is a separate checked stage: an exact
identity must also have a supported repeating execution plan. Stage failures
retain diagnostics, successful artifacts, and a nonzero final status.

## Add an algorithm without duplicating the machinery

For an explicit formula, supply a block `step` using shared arithmetic and the
provided child-product callback. For a coefficient recipe, supply exact U/V/W
data and verify it before execution. Define its independent operation budget and
test complete base identities, recursive products, numeric domains, padding, and
agreement between predicted and observed work. Preserve source attribution and
license notices in data and references.

Keep new construction algorithms separate from execution primitives. Explain
index conventions and cancellation arguments near the code that uses them.
Extend exact checking before introducing new coefficient fields. Use small
regressions for development; larger research runs belong in dated ignored output
directories, with their settings and provenance recorded.
