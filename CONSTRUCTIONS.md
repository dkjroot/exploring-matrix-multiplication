# Constructing and searching for executable schemes

The construction layer is independently written Python using exact rational
arithmetic. It builds coefficient recipes; it does not call somebody else's
matrix multiplier, solver, or search implementation. Generated matrix recipes
use our existing recursive execution primitives and operation counters.

This is the next research tool, **not a new exponent result**. Smirnov's
2.7742999804 remains our lowest fully verified executable square exponent.
Neither increasing a known tensor power nor discovering another seven-product
2-by-2 scheme beats that exponent.

## A working discovery experiment

To run all of these report families together, simply use `python sweep.py`.
The combined suite generates discovery/replay and tensor-product recipes, CW
recovery/extraction/packing reports, certificates, diagnostic pairs/traces, and the original size
sweep in one dated directory. Generated matrix recipes are measured automatically.
Its manifest records budgets and outcomes; a bounded search miss remains a miss,
while an invalid identity or failed execution marks the suite as failed.
With `--search-start`, a generated real rectangular base is joined to its two
verified cyclic orientations automatically, producing a balanced square-target
plan for the measurement sweep and portable certificate.
`python sweep.py --measurements-only` retains the original count-only workflow.

```sh
python -m matrix_multiplication.constructions search --target-rank 7 --certificate
```

The input is the eight elementary products of classical 2-by-2 multiplication.
No seven-product formula or coefficients are supplied. With the default seed,
this bounded rational walk finds a seven-product identity. Each move preserves
the tensor algebraically; the final scheme passes a complete exact matrix
identity check. `--certificate` additionally runs the independent basis checker
and bundles its portable checker. The resulting recipe has exponent log_2(7),
the same exponent as Strassen, rather than a new record.

Every command writes a fresh `outputs/construction_DATE_TIME/` directory and
automatically generates `index.html`. A search saves:

- `scheme.json`: the verified finite matrix multiplication recipe.
- `report.json`: the starting recipe, settings, attempt counts, best path, source
  attribution, rank, and exact comparison against exponent 11/4.
- `index.html`: a readable summary and graph of product count along the best
  replay path, with both axes labelled.
- `certificate/`: an optional portable exact verification package.

Unsuccessful bounded searches also save their best verified recipe and explicitly
report `target_reached: false`. They do not establish that the target is impossible.
The best-path graph excludes unsuccessful branches; the attempt totals include
them. A flip that removes a zero factor can decrease rank, so the graph replays
the transformations rather than estimating the decrease from event names.

Replay a search without re-running its random walk:

```sh
python -m matrix_multiplication.constructions replay outputs/construction_DATE_TIME/report.json --certificate
```

The starting recipe is reverified, every move is replayed, reduction dependencies
are checked, and the final matrix identity is verified again. Saved success labels
are insufficient to accept an incorrect recipe. Replay does not certify search
attempt statistics or assert that the random walk explored all possibilities.

## Measure a generated algorithm

Use the actual path printed by the construction command:

```sh
python explore.py --composition outputs/construction_DATE_TIME/scheme.json --instrument --counts-only --report outputs/generated-pair.json
python sweep.py --measurements-only --sizes 2 4 8 --cutoffs 1 2 --algorithms naive strassen --composition outputs/construction_DATE_TIME/scheme.json
```

These execute the generated coefficients and produce the existing automatic
HTML arithmetic reports, including observed products, all arithmetic steps,
padding/cutoff effects, and predicted-versus-measured growth. The seven-product
discovery has been exercised at sizes 2, 4, and 8 with 7, 49, and 343 measured
leaf products, as well as rectangular input and cutoff checks. Its coefficient
evaluation may require more additions/scalings than our handwritten Strassen
step. That changes the finite operation count, not the square exponent.

Construction and search are setup work. Their attempt counters and transcript
are separate from the arithmetic measurements of executing a matrix product.
We do not currently count every rational operation inside the generator itself.

The combined sweep's small integer and floating-point pair reports include
primitive profiles, memory diagnostics, and bounded traces automatically. Their
original `measurements.json` files retain the `reports`/`events` envelope. The
size sweep itself keeps primitive clocks and allocation probes disabled; it
embeds those separate diagnostic snapshots without assigning their timings or
memory values to larger matrix sizes. Use `--pair-trace-limit` to change the
retained prefix; it does not disable any arithmetic counting or identity check.

## Search controls and strict exponent targets

```sh
python -m matrix_multiplication.constructions search --size 3 --target-rank 23 --steps 10000 --restarts 20 --seed 42
python -m matrix_multiplication.constructions search --size 4 --target-exponent 11/4 --steps 10000 --restarts 20 --max-height 4 --extra-rank 2
python -m matrix_multiplication.constructions search --start path/to/real-rational-scheme.json --target-rank 100 --max-height 8
```

These larger commands are research searches, not promises of success. Choose a
size, rank target, and walk budget deliberately; only the small default discovery
is a regression fixture. `--target-rank` defaults to one below the starting rank.
`--target-exponent` instead chooses the largest integer rank strictly below the
requested square bound. For example, below 2.75 at dimension 4 requires rank at
most 45; the 48-product algorithm is insufficient. Rank comparisons use integer
powers `rank^denominator < dimension^numerator`, without float rounding.

The search offers:

- Proportional common-factor flips in all three tensor legs, both shear
  directions, and signs +1/-1. Two cross terms cancel exactly.
- Sparse rational Gaussian elimination to find dependencies between two-factor
  products. A dependency lets the remaining factor absorb one removed product.
- Optional exact factor splits, allowing temporary rank increases to leave
  components unreachable by monotone walks. `--extra-rank` caps that increase;
  `--split-every` controls how often it is attempted.
- Seeded restarts and a coefficient numerator/denominator ceiling. The default
  `--max-height 1` keeps coefficients in {-1,0,1} over Q; this is **not** arithmetic
  modulo two. Raise it to explore larger rational coefficients. The starting
  recipe must fit that ceiling too, and a flip is rejected if its reduced result
  exceeds it. Exact reductions can increase coefficient sizes even when they
  lower the product count, so identity preservation alone does not satisfy the
  resource bound.

The search is deliberately heuristic and incomplete. Shared factors can become
rare; coefficient bounds and a rank ceiling exclude possible paths. A bounded
walk does not search all rational decompositions. Larger resources can enlarge
experiments but cannot guarantee a suitable decomposition exists at a chosen
dimension or is reachable by these moves.

## Tensor construction primitives

`matrix_multiplication/tensors.py` provides sparse rational vectors, exact tensor
expansion, direct sums, tensor products/powers, general linear maps, coordinate
restriction, and checked conversion to/from our matrix recipe format. Generic
tensors are kept distinct from matrix multiplication recipes. Conversion checks
every matrix tensor coefficient, rejecting an unrelated tensor of matching size.

Tensor products of matrix recipes need a coordinate permutation: pairing two
flattened matrices does not by itself yield the desired paired-row/paired-column
layout. `matrix_product` supplies and checks that permutation. For example:

```sh
python -m matrix_multiplication.constructions product --left strassen --right strassen --certificate
```

This yields an executable 4-by-4 rank-49 recipe. It has the same exponent as the
input scheme. Mixed existing schemes cannot beat their best constituent exponent
just by tensoring them. Improvements require a rank reduction or a better
restriction/extraction construction. Attribution and license metadata from input
documents are carried into generated artifacts; transformations do not erase
upstream reuse conditions. Keep externally sourced data in ignored directories
and include its notices before redistribution.

## Starting the older asymptotic route

```sh
python -m matrix_multiplication.constructions cw --q 2 --power 2
```

This constructs the published Coppersmith–Winograd polynomial family, takes a
tensor power with a **shared formal parameter**, recovers its leading tensor
coefficient by exact rational interpolation, and compares every coefficient with
an independently constructed literal CW tensor power. It saves a generic
`tensor.json`, a research report, and its HTML summary. It deliberately does not
save `scheme.json`: CW is not itself an ordinary square matrix product.

For parameter q and power p, the polynomial family has `(q+2)^p` rank-one terms.
Multiplying the published Laurent formula by epsilon^3 removes negative powers;
the coefficient of epsilon^(3p) is the desired CW tensor power. Its degree bound
is 10p. Interpolation at the exact integers 0 through 10p uses at most
`(10p+1)*(q+2)^p` products; zero factors can reduce that finite count. The report
lists the polynomial term count, degree, interpolation points, actual recovered
recipe products, and literal recipe products separately. We never substitute
the border rank for the finite executable count or choose a small floating
epsilon and call the result exact.

## Verified extraction of independent products

The first finite extraction experiment is now available, and runs automatically
for every q/power pair selected by the combined suite:

```sh
python -m matrix_multiplication.constructions cw-extract --q 2 --power 2
```

CW variables fall into three groups: group 0 contains coordinate 0, group 1
contains coordinates 1 through q, and group 2 contains coordinate q+1. A sector
chooses a group for each of the three tensor legs, X/Y/Z. For example, `011`
means keep group 0 in X and group 1 in both Y and Z. The six supported sector
triples sum to two: `002`, `011`, `020`, `101`, `110`, and `200`.

Each sector describes a matrix product. `101` supplies a row index of width q,
`110` supplies an inner index of width q, and `011` supplies a column index of
width q. The other three are scalar products. A power combines these choices:
`101,110,011` has shape (q,q,q). The generator records the exact source-coordinate
ordering that turns a sector word into row-major A, B, and C matrices. The tests
verify every word in several small powers, including nontrivial permutations.

Selecting several sector words is harder than selecting one. Different products
must use disjoint variable groups in **all three** legs. Even that is insufficient:
the union of retained groups can leave an extra sector joining X from one
product to Y/Z from other products. The generator checks all supported group
combinations and rejects these unwanted cross-products. Its default selection
uses a deterministic greedy procedure, preferring larger and more balanced
products. This is an induced matching, not a proof of the largest possible
selection. Explicit choices can be supplied by repeating `--sector-word`.

For the q=2, power=2 default extraction, the selected shapes are (2,1,2),
(2,2,1), and two (1,1,1) products. They are four independent products, rather
than one larger dense product. The verification checks every tensor coefficient
twice: first the literal CW restriction against an independently constructed
classical direct sum, then exact polynomial recovery against that same target.
Restriction happens before interpolation, deleting unused variables and zero
terms before they are evaluated. Small actual signed rational inputs then pass
the separate matrix oracle, with every arithmetic operation observed.

The report saves `tensor.json`, explicit coordinate selections, complete identity
results, and execution counters. The HTML presents the product shapes and cost
comparison before the raw JSON. Those four products need **10** classical data
multiplications, but our finite recovered recipe needs **176**. Restriction also
lowers the factor-based polynomial degree bound from 20 to 11, so 12 interpolation
points suffice; recovering at the original degree bound would cost 320 products.
The degree bound is checked from exact retained coefficients, without assuming
extra cancellations. Polynomial term
count, interpolation upper bound, and actual executed product count are separate
fields. Coefficient scalings and additions appear in the execution counters.
The reusable `tensor_execution.py` kernel counts even scaling by one; its
documented schedule differs from the matrix-scheme execution engine's shortcuts.

A single component with all dimensions at least two also exports `scheme.json`,
which enters the existing composition, measurement, and certificate machinery:

```sh
python -m matrix_multiplication.constructions cw-extract --q 2 --power 3 --sector-word 101,110,011 --certificate
```

This particular exact 2-by-2 recovered recipe has **324** products, versus eight
classically. It verifies the executable bridge, but has a worse exponent. It is
not automatically added to the default growth grid, where repeated recursion
would be needlessly expensive. A multi-component sum is never exported as a
single dense matrix recipe. Unit-size axes work in the generic tensor kernel;
they do not require widening the existing recursive scheme format.

## Improved recovery and executable dense packing

```sh
python -m matrix_multiplication.constructions cw-pack --q 2 --power 2 --size 2 --certificate
python sweep.py --cw-parameters 2 --cw-powers 2 --cw-packing-size 4
```

`cw-pack` first compares two exact recovery routes. Interpolation evaluates the
polynomial at several points. Direct coefficient expansion instead writes each
factor as a sum of coefficient vectors and forms only triples of degrees whose
sum is the required degree. This is ordinary polynomial convolution expressed
as rank-one tensor terms. Sparse degrees can make it cheaper; dense degrees can
make it more expensive. Neither route assumes that border rank is finite rank.

For CW₂ power two, coefficient expansion gives 42 products instead of the
interpolated 176. Our existing exact dependency reductions reduce those 42 to
10. The implementation then regenerates/replays every reduction and compares
all tensor coefficients. It has not replaced the result with a prewritten
classical algorithm: the recipe follows from the polynomial coefficient terms
and their recorded exact dependencies. Both candidate costs and the selected
recipe's measured arithmetic remain in the report.

Packing uses those independent products as slots for tiles of a dense product.
Think of the dense multiplication as triples (row, inner index, column). A tile
computes one rectangular set of triples. Different inner tiles can contribute
to the same output entry, so their outputs must be added. The planner assigns
tiles to slots, uses successive batches when needed, and pads partial slots
with zeros. It requires every dense-product triple to be covered exactly once.
An independent multiplicity check catches holes and double counting.

Explicit linear maps connect each slot's input entries to the dense A/B entries
and its outputs to dense C entries. Applying the maps to the recovered bilinear
recipe produces a new dense recipe. The maps can create additional dependencies,
which are reduced and replayed again. The complete resulting tensor is then
checked against matrix multiplication, followed by an actual rational-input
call checked with the separate oracle. Construction and replay remain setup
work; all arithmetic executing the final recipe uses observable primitives.

For the 2-by-2 example, four mixed slots cover the volume in one batch. The
mapped recipe has 10 products, then two dependency reductions give eight.
Exporting it through `scheme.json` gives a genuine recursive multiplier with
8, 64, and 512 measured data products at sizes 2, 4, and 8 with cutoff one.
Its exponent is **3**, so this is a verified pipeline milestone, not an
improvement over the existing Smirnov baseline. Size three similarly produces
a 27-product dense base. Size two is a small correctness case, not a promising
base for beating 2.75: a 2-by-2 rank at most six would be required, which is
excluded by the known seven-product lower bound.

Every `python sweep.py` run performs recovery, extraction, coarsening, and packing for its
selected q/power grid. `--cw-packing-size` controls the square construction base
independently of `--sizes`. The suite admits one best packing with rank at most
the classical base volume, breaking equal-rank ties by declared arithmetic
steps and then name. It joins the growth plots as `generated-cw-packed`, plus
pair diagnostics and certificates. Worse-than-cubic bases retain their finite
cost/identity reports and can be measured explicitly with `--composition`; they
are excluded from automatic recursion to avoid unexpectedly enormous sweeps.

The current planner is finite volume tiling, **not** the large-power conversion
behind Schönhage's asymptotic sum inequality. Sum-index coarsening is implemented
below. Stronger extraction from its components, many copies of larger components, and a constructive
large-power recovery/packing schedule remain mathematical research. No complete
laser-method construction or its published limiting exponent is implemented.
Materializing the catalog (6^power), literal tensor, and polynomial family still
grows exponentially; succinct index rules and streaming remain future work.

## Coarse CW components and exact matrix classification

```sh
python -m matrix_multiplication.constructions cw-coarse --q 2 --power 2 --size 2 --certificate
```

This command runs automatically for every q/power pair in `python sweep.py`;
it uses the same `--cw-parameters`, `--cw-powers`, and `--cw-packing-size` settings.
Its report directory is `cw-coarse-qQ-powerP/` inside the combined dated output.
There is no second reporting command. Its dense scheme competes with fine
packing for `generated-cw-packed`; all catalogs remain available even when
another candidate wins or a different stage fails.

A fine group is a word such as (0,1) or (1,0). Coarsening puts both words into
the group with summed weight one. At power p there are weights 0 through 2p,
and sector triples whose weights sum to 2p. The catalog restricts the literal
CW power to every such triple, records every exact coefficient and original
coordinate selection, and independently recombines those coefficients to check
the complete original tensor. These components share spaces. They are not all
independent products just because they appear as separate catalog rows.

`tensor_classification.py` distinguishes vector-space dimensions from matrix
dimensions. A product of shape (m,k,n) needs spaces (m*k,k*n,m*n). Integer
arithmetic first checks whether those dimensions are possible. Equal support
neighborhoods then identify rows, inner indices and columns. Explicit bijective
permutations must pass a complete coefficient comparison with the classical
matrix tensor before the component receives a matrix label. Unit axes work
throughout this classifier; no recursive scheme-format extension is needed.
The test accepts coefficient-one basis permutations. It does not search general
linear changes of basis. Non-recognition is limited to that test, whereas a
dimension obstruction rules out an ordinary matrix tensor on the full spaces.
Neither rules out useful smaller restrictions or products extracted from powers.

For CW₂ squared, 36 fine sector words merge into 15 components. Space dimensions
by weight are (1,4,6,4,1). Twelve components have verified matrix layouts. The
interior sectors (1,1,2), (1,2,1) and (2,1,1) have permutations of space dimensions
(4,4,6), which cannot equal (m*k,k*n,m*n) for positive integers. For example,
the first would require m² = 4*6/4 = 6. Their complete sparse coefficients remain
in `report.json` for subsequent extraction research, rather than being silently
discarded or called matrix multiplications.

The greedy coarse selector chooses shapes (1,1,6) and (4,1,1). These are larger
individual pieces than the fine selection, with the same total classical volume
of ten. For q=1 at power two, coarse selection increases that volume from four
to five. This is an extraction observation, not a lower exponent. The selector
rejects both shared weights and unintended supported cross-components, then
checks the full selected direct-sum identity. Recovery restricts the polynomial
family before interpolation, compares coefficient expansion, replays exact
dependency reductions, and measures a real rational-input direct-sum call.

Packing uses the shared finite tile planner and exact map checker. The CW₂
power-two example still gives an eight-product 2-by-2 dense base, exponent three.
Its exported scheme and optional certificate are genuine executable identities.
The new opportunity is to examine powers and restrictions of the retained
interior components, where simply recognizing a bigger matrix is insufficient.
Recursive laser extraction, better selection, and constructive large-power
conversion remain separate mathematical work; this catalog does not supply
their published exponent or the OpenAI witness.

## Relation to the OpenAI witness problem

For any chosen positive epsilon, the 9/4 argument claims a finite size u and
rank r below `u^(9/4+epsilon)`. Executing such an algorithm requires an actual
verified decomposition. Our generator can search particular finite rational
instances and accept a successful exact witness; it does not turn that existence
argument into a finite recipe. It does not determine u or supply a search bound.
The witness could need algebraic coefficients outside Q, which this initial
construction/search layer does not represent. The existing multiplier/checker
supports Q(i), but that is also less general than an arbitrary algebraic field.

A finite scheme with exponent exactly 9/4 is not required to approach that bound.
For example, a chosen target of 23/10 would be entered as `--target-exponent
23/10`, with a suitable square starting size and a deliberate search budget.
This is a research target, not a guarantee that the selected finite search will
find the paper's witness. We will claim progress below 2.75 only after an actual
verified square construction, its recursive cost argument, and executable
measurements support it.

## References and attribution

- Manuel Kauers and Jakob Moosbauer, [Flip Graphs for Matrix Multiplication](https://arxiv.org/abs/2212.01175): mathematical flip/reduction search framework. Our rational elimination, walk, split operations, and transcript implementation are independently written; no upstream implementation was copied or executed.
- Markus Bläser and Vladimir Lysikov, [On Degeneration of Tensors and Algebras](https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf), MFCS 2016, equation (2): CW degeneration. The publication identifies CC-BY licensing. We credit the mathematical formula; no publication text or implementation code is redistributed here.
- Josh Alman and coauthors, [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349): tensor powers, CW sectors and extraction; sections 2.2 and 3.7 describe the sum-index coarsening motivating this implementation. The paper's exponent is not assigned to our finite experiments.
- [OpenAI 9/4 manuscript](https://raw.githubusercontent.com/openai/math/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/paper.pdf): existence target, distinct from the finite executable witness we still seek.
