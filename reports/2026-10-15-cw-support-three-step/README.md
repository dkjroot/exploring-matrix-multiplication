# Three sequential CW support experiments, 15 October 2026

The three requested research steps are complete. They yield an exact balanced
matrix restriction absent from our earlier controls, a replayed reduction from
eight products to seven, and a certified 4-by-4 recombination with 49 products.
This reproduces the known rank-seven exponent, about 2.8074; it does not improve
the stronger Smirnov baseline of about 2.7743 or claim a new mathematical record.

## Step 1: change shape within the old seed pools

We searched the three squared interior CW₂ components for shape (2,2,2), using
only coordinates already retained by earlier seed witnesses. Each source used
a 4,096-unit cap. An input-pair screen and an output-layout proposal each consume
one unit, with separate counts. Across the three sources there were 12,285 A/B
screens and three layout proposals; none yielded a matrix product. Exhaustion
is unknown at a cap. This is a bounded miss, not an exclusion.

## Step 2: broaden the spaces and coordinated target shapes

The next step allowed the complete squared component spaces, trying (2,2,2),
(2,2,3), (2,3,2), and (3,2,2), with a separate 4,096-unit cap for every target and
orientation. It tested 48,925 A/B screens and 227 layout proposals. Eight full
2-by-2 layouts were verified in the squared (1,1,2) component; two were retained.
The other prefixes missed. Those orientation misses do not exclude equivalent
maps obtainable by exact symmetry transport.

The first retained map has local source spaces (16,16,36). In matrix row-major
order its A coordinates are (0,1,4,5), B coordinates (8,12,9,13), and C coordinates
(11,23,17,29). All four output coordinates are outside the old seed output pool.
Thus the broader scope resolved a concrete limitation of the pooled search.

The support conditions are necessary but insufficient: for (m,k,n), each output
needs k coefficient-one products, each A entry occurs n times and each B entry
m times. Complete recognition and a separate original-source coefficient check
still establish every admitted identity. Subset streams are diagonally
interleaved to avoid spending the whole prefix on one A subset. Exact expansion
incorporates cancellations before pruning; no floating-point proposal or hidden
matrix implementation is used.

## Step 3: pay recovery, reduce exactly and reuse the result

Both retained layouts recover from a 432-product interpolation recipe to eight
executable products by exact coefficient expansion. The first recovered recipe
enters a bounded rational flip/reduction campaign: 1,000 attempted flips per
restart, ten restarts per seed, coefficient height at most one, seeds 0 then 1.

Seed 0 misses rank seven after 10,000 attempted flips. Seed 1 succeeds after
1,779. Its retained 468-event transcript replays exactly from the original
eight-product recipe. Complete coefficient verification and actual signed
rational-input execution pass for the seven-product result. Search and
coefficient compilation are setup work; the finished recipe pays seven data
products, with its additions and coefficient arithmetic observed separately.

The refined part is tensored with a separate copy of itself, with explicit
matrix index permutations, giving a 4-by-4, 49-product recipe. Its independent
certificate verifies the exact base and repeating recurrence. The arithmetic
exponent is log₄(49) = log₂(7), not omega itself. Actual recursive execution and
independent arithmetic budgets agree. In the small growth sweep the exported
radix-four recipe uses eight products at size two (a classical fallback) and 49
at size four; no smaller-size seven-product execution is invented for that plan.
The separate seven-product part remains reusable in the library.

## Evidence preserved outside ignored output

- `parts.json`: the four executable recipes and their provenance: the first
  eight-product part, its seven-product reduction, the second eight-product
  layout, and the 49-product recombination. Source-coordinate maps, finite
  recovery costs and actual arithmetic observations are retained.
- `scheme.json`: the exported 4-by-4 recipe, usable directly by explore/sweep.
- `certificate.json`: the original independently checkable base/recurrence
  certificate. The maintained checker is in the source tree, not only output.
- `evidence.json`: compact complete step summaries, admitted layouts, original
  source fingerprints, original setup observations, and both failed/successful
  rank seeds. The successful replay is retained in full. Reproducible raw support
  attempt rows are represented by their source hash and representative cases.
- `regression.log` and `focused.log`: the passed 79-case relevant regression run
  and the nine new focused cases.
- `file-sha256.json`: fingerprints binding these retained artifacts.

The selected combined suite is `outputs/2026-10-15_171135_216977/`; all 26 stages
passed. It includes the earlier report families, new three-step family,
certificates, exact and floating diagnostics, and generated-recipe measurements.
Its original timing, Python allocation peak and source fingerprints are kept as
observations of that run, not relabelled as new measurements. Setup arithmetic
operation counts remain unmeasured; elapsed time is secondary evidence. Source
documentation coverage and diff checks passed. The much slower unrelated full
suite was not rerun for this bounded change.

Archived JSON/log bytes are exempt from Git newline conversion so original file
hashes survive commit and checkout. Source files continue to use LF. No useful
implementation or selected result depends on retaining `outputs/`.

## Reproduce or reuse

From the repository root, with Python 3.10 or newer and the standard library:

```
python -m matrix_multiplication.constructions cw-support --certificate
python sweep.py --sizes 2 4 --cutoffs 1 --algorithms naive strassen --cw-parameters 2 --cw-powers 2 --search-steps 50 --search-restarts 1 --pair-trace-limit 10 --projection-cycles 2
python -m matrix_multiplication.certificates verify reports/2026-10-15-cw-support-three-step/certificate.json
python explore.py --composition reports/2026-10-15-cw-support-three-step/scheme.json --counts-only
```

The CW experiment needs no external coefficient downloads. Regeneration creates
fresh dated reports with the same exact identities and search outcomes under
the same implementation and parameters; original timing, memory, timestamps and
hashes of a different revision are not recreated. On Windows, use `-X utf8` when
running older tests that read UTF-8 HTML with the platform default encoding.

```python
from matrix_multiplication.recipe_parts import load_parts
from matrix_multiplication.decomposition_search import replay

parts, provenance = load_parts("reports/2026-10-15-cw-support-three-step/parts.json")
# Each executable part is completely checked when loaded. The replay events
# and starting-part link are retained in evidence.json for derivation replay.
```

Mathematical sources and implementation provenance are credited in
CONSTRUCTIONS.md and in the artifact metadata: the CW polynomial/extraction
sources and the flip-graph framework. The implementation and constructed data
are our exact reference work under the repository GPL-3.0 license. The seven-
product outcome is a reproduction of a known case, not a claim of novelty.

## Next research angle

Transport the verified maps through explicit, fully checked cyclic symmetries,
then test joint independence of the eight known layouts and recover accepted
unions as a whole. RESEARCH_PLAN.md bounds the next pair search at 4,096 and
requires same-target finite recovery/packing comparisons. Space overlap,
symmetry intuition and border-rank labels cannot supply independence or unpaid
overhead. A stronger construction remains open.
