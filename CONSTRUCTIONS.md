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
recovery reports, certificates, diagnostic pairs/traces, and the original size
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

The next missing mathematical work is a specified extraction of independent
matrix-product components from these powers, followed by a constructive way to
pack/recover those products as the desired dense multiplication. The restriction,
direct-sum, exact recovery, and verification primitives now exist for that work.
No complete laser-method construction or its published limiting exponent has
been implemented yet. Materializing powers can grow exponentially; eventually
we will need succinct index rules and streamed execution instead of full lists.

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
- Josh Alman and coauthors, [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349): tensor powers, CW sectors, and extraction background. The paper's exponent is not assigned to our finite experiments.
- [OpenAI 9/4 manuscript](https://raw.githubusercontent.com/openai/math/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/paper.pdf): existence target, distinct from the finite executable witness we still seek.
