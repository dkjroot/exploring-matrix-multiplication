# Comparing implementation candidates

Run `python sweep.py` for the combined research report. Existing construction,
search, reusable-part, certificate, arithmetic, operand-growth and diagnostic
families remain included. Its comparison now has three separate graphs:

1. **Mathematical work efficiency**: classical work divided by candidate work on
   identical inputs. Higher is better under the declared arithmetic schedule.
2. **Measured Python memory growth**: traced allocation peaks, including observer
   overhead and cold preparation inside the call. This is an implementation
   diagnostic, not intrinsic peak algorithm storage. Lower is better within that
   scope. `--no-sweep-memory` leaves this graph explicitly unavailable.
3. **Logical data movement**: mathematical payload reads and writes in the
   reference traversal. Lower is better within that schedule. This is neither
   measured cache/DRAM traffic nor a bound for every implementation of a recipe.

The overview uses the smallest requested cutoff, matching the existing arithmetic
overview; expandable evidence keeps every cutoff. Integer work totals and the
reduced numerator/denominator of each efficiency ratio are saved in JSON.
Floating coordinates in the HTML are display evidence only. A score of two means
half the modelled arithmetic work, not a prediction of twice the execution speed.
No score measures distance from a mathematically optimal algorithm.
The score ranks explicit executable schedules. A generic coefficient recipe can
perform scales by one that a hand-written equivalent formula omits, and a recipe
can select a more expensive exact representation. These differences are visible
costs of the current reference schedules, not proofs that every implementation of
the underlying identities must differ by the same factor.

## What one work unit means

`exact-schoolbook-euclid-64-v1` gives each of these categories weight one:
64-bit magnitude-word product, word addition, word subtraction, carry/borrow
step, word comparison, word shift, and sign step. Equal weights are a portable
convention rather than measured instruction prices. A single score needs this
explicit choice; a future hardware model can assign different weights to the
saved category totals without overwriting this model.

An integer of magnitude bit width b has ceil(b/64) words; zero has one bit and
one word. Words are accounting units and never narrow the exact values.

- Multiplication of u-word and v-word integers uses u*v word products, 2*u*v
  word additions, 2*u*v carry steps, and one sign step. This describes schoolbook
  accumulation of two-word partial products with explicit carry handling.
  Fixed coefficients are charged by the same rule, separately attributed to
  coefficient work. There are no zero/one data-product shortcuts.
- Addition/subtraction uses a full max(u,v)-word magnitude comparison, the same
  number of sum/difference steps and carry/borrow steps, and one sign step.
  Operand signs determine whether the magnitude operation is addition or
  subtraction. Semantic field additions and subtractions remain separate in the
  existing counters, irrespective of this magnitude-level classification.
- Division uses a binary restoring schedule. With b dividend bits and w words
  sufficient for divisor_bits+1 bits of remainder storage, it charges b*(w+1)
  word shifts, b*w comparisons, popcount(abs(quotient))*w subtractions and borrow
  steps, and one sign step. The extra shift per dividend bit accounts for input
  bit/quotient access. This simple schedule charges division by one too when
  requested; an identity reduction by one is omitted by the rational schedule.
  Integer normalization uses checked quotient/remainder arithmetic. Non-exact
  divisions encountered during Euclid are also charged.
- GCD uses our own Euclidean loop, charging each remainder division and full-word
  zero test, plus its terminating zero test. Iterations are saved separately.

These quantities describe a declared reference schedule. They are not Python
instructions, measured CPU time, or the bit complexity of the best big-integer
algorithms. Control flow outside the arithmetic schedule, allocation, movement,
setup and verification are separate. GCD calls and iterations are explanatory
statistics: they are **not** added on top of the word work they expand into.

## Exact fractions and complex components

For a/b +/- c/d, the schedule finds g=gcd(b,d), forms s=b/g and t=d/g, computes
a*t +/- c*s over denominator s*d, then reduces numerator and denominator using
their GCD and checked exact divisions. Multiplication by a unit denominator is
a copy. Each replayed result is checked against the original scalar result.

For (a/b)*(c/d), it finds gcd(a,d) and gcd(c,b), divides out those cross factors,
and only then multiplies the reduced numerators and denominators. Division uses
the reciprocal and handles its sign. This exposes both reduction cost and the
word products saved by cross-cancellation. The schedule records rational
operations, normalizations, normalization reductions, GCD calls and iterations,
exact divisions and cross-cancellations. These are model statistics, not claims
about Fraction's internal implementation.

Exact Gaussian pairs already expand into real component primitives. Each
component is charged once, so field and component layers must not be summed.
Floats and opaque native complex operations get partial coverage and no complete
work score. Failed calls or a replay mismatch also prevent scoring. Observation
does not change the original scalar operators or their evaluation order.

## Classical denominator and scope

Every successful candidate is compared with classical multiplication of the same
inputs, identified by their fingerprint. A verified classical sweep sample is
reused when available. Otherwise a separate classical reference is executed and
verified outside every candidate's timer, memory probe and arithmetic counters.
`classical_reference_executions` records these additional calls. Selecting only
a generated recipe still produces a denominator.

Inputs, size policy, cutoff and arithmetic representation can change rankings.
The current growth sweep uses signed integer inputs with magnitude at most four;
some recipes require rational or exact complex intermediate values. Single-pair
API reports also support exact fractions. The graph does not establish the same
ranking for other inputs or domains, and its observed work is not extrapolated
beyond captured sizes. Existing recurrence projections remain separate, exact
field-category predictions.

## Logical movement and memory

`logical-value-traffic-64-v1` counts scalar primitive operand reads and result
writes, plus explicit matrix/vector copies, zero initialization, conversions
and stores in the shared traversal. A copy has one read and one write. Repeated
uses count repeatedly. An integer uses magnitude-word payload; a Fraction stores
both numerator and denominator, even when its denominator is one; an exact
Gaussian pair stores both components. Integer divmod writes quotient and
remainder. A 4,096-bit integer copy transfers 128 logical words.

Python often copies an object reference rather than all these words. This model
describes mathematical payload flow. It excludes internal scalar temporaries,
cache reuse, pointers/object headers, setup, validation, observer replay and
verification. It cannot predict DRAM traffic. Unknown payloads are recorded
explicitly and do not get a complete transfer total. Raw JSON separates
arithmetic ports from structural traffic and reads from writes.

No new algorithm peak-memory bound is claimed. Python allocation observations now
also include arithmetic replay machinery; compare their named scope, not an
earlier run's different observer as if only the algorithm had changed.
Lifetime-aware peak payload storage and cache-capacity-aware transfer schedules
remain further modelling work. Keep several candidates when arithmetic, storage
and communication tradeoffs differ, rather than assigning memory an arbitrary
penalty inside the arithmetic score.

## Earlier evidence remains valid

`exact-operand-word-cost-v1` remains unchanged. It counts unreduced rational
expressions and excludes GCD, division and carry work. Older graphs and proved
resource bounds continue to refer to that definition. The new score is a
separate model; do not add its total to older estimates or project it using
their bounds. Older JSON gets no invented score or movement observation.

Implementation: `mathematical_work.py`, `data_movement.py`, `work_comparison.py`,
and hooks in the existing scalar observer and shared matrix primitives. New
statistics are present in `explore.py`, automatic pair diagnostics, construction
execution snapshots, and all ordinary/generated series of the combined sweep.
