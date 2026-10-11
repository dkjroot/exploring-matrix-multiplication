# Exploring matrix multiplication

Five implementations, including the Smirnov baseline, use Python arithmetic
and lists, without NumPy, SciPy,
BLAS, or another matrix multiplication library. Python 3.10 or newer is required;
there are no third-party runtime or test dependencies.

The research goal is to own the algorithm implementations and expose their steps
for composition, instrumentation, and profiling. The current Python reference
implementations use primitive arithmetic, but future execution backends may use
compiled code, custom CUDA kernels, packed arrays, and runtime/compiler libraries.
Published formulas and coefficient data are specifications; another library's
implementation of a studied multiplication algorithm must not replace our own.

The current research target is the lowest verified arithmetic exponent of a
reproducible, openly licensed implementation of exact dense square matrix
multiplication. Resource use and elapsed time will be measured separately. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the public-code audit, candidate baseline,
verification requirements, and next experiments. A record is a research objective,
not a result already established by this repository.

The priority is to turn stronger mathematical constructions into explicit,
testable algorithms, even when they require enormous constants, memory, or time.
Historical implementation difficulties are research problems to tackle, rather
than ceilings on the project. GPU throughput and memory traffic are secondary
diagnostics; success is a verified arithmetic recurrence with a lower exponent.
The 2.25 result remains a target for constructive research. Reproducing an older
bound in executable form would also be meaningful progress.

This repository retains its existing [GPL version 3 license](LICENSE). Comparisons
include implementations under any open-source license. All referenced
implementations are attributed below; imported data and reference snapshots remain
in the ignored `Background Research/` directory. When redistributing third-party
material, retain its license and required notices. Material with unresolved reuse
terms is reference material rather than a release dependency.

| Algorithm | Module | Recursive scheme | Square arithmetic exponent |
| --- | --- | --- | --- |
| Naive | `matrix_multiplication/naive.py` | Row-by-column dot products | 3 |
| Strassen | `matrix_multiplication/strassen.py` | Seven products; 18 additions/subtractions per 2-by-2 block level | log₂(7), about 2.8074 |
| Strassen-Winograd | `matrix_multiplication/strassen_winograd.py` | Seven products; 15 additions/subtractions per level | log₂(7), about 2.8074 |
| AlphaEvolve | `matrix_multiplication/alphaevolve.py` | 48 complex-field products per 4-by-4 block level | log₄(48), about 2.7925 |
| Smirnov | `matrix_multiplication/smirnov.py` | Three 40-product rectangular orientations form a 54-by-54 square base | log₅₄(64000), about 2.7743 |

## Run the explorer

The AlphaEvolve notebook and the AlphaEvolve/Smirnov coefficients are local
research data, excluded from Git. In a fresh checkout, import them explicitly
with these standard-library-only tools:

```
python tools/import_alphaevolve.py --download
python tools/import_smirnov.py --download
```

If you downloaded the notebook manually, use
`python tools/import_alphaevolve.py "path/to/mathematical_results.ipynb"` instead.
The importer reads literal data and never runs notebook code. It verifies all
4,096 coefficients of the rank-48 tensor identity before saving the data.
The Smirnov importer verifies its rectangular recipe and retains its provenance
and license. The
default explorer (`--algorithm all`) and sweep include all five algorithms;
missing coefficients produce an explicit error rather than silently skipping one.

```
python explore.py
python explore.py --list
python explore.py --algorithm all --a "[[1,2,3],[4,5,6]]" --b "[[7,8],[9,10],[11,12]]"
python explore.py --algorithm alphaevolve --a "[[1j,2],[3,4j]]" --b "[[5,6],[7,8]]"
python explore.py --self-test
```

The explorer prints the product and verifies it against independent exact
rational arithmetic. It exits with status 1 if an algorithm errors or its result
fails verification. `--atol` and `--rtol` configure floating-point tolerances.
`python -m unittest discover -s tests -v` runs the same regression suite.
The tests also work before downloading research data: unavailable data-dependent
cases report explicit skips, while self-contained algorithms, construction
primitives, and missing-data diagnostics still run. Existing but corrupted data
fails verification; it is never treated as an unavailable optional dependency.

## Reading and extending the code

[CODE_GUIDE.md](CODE_GUIDE.md) gives a suggested reading order, a worked recursion
example, and the boundaries between execution, prediction, construction, and
verification. Every maintained Python module, class, and function has a
docstring. Inline comments explain index conventions, exact cancellations,
observation scopes, and why each layer exists. Check documentation coverage with:

```
python tools/check_documentation.py
```

The check confirms documentation is present; human review is still needed to
decide whether its explanation is useful. Tests include adversarial coefficient,
JSON, replay, and failure-path cases rather than relying only on example products.
Recipe containers and Gaussian coefficients are immutable after construction.
JSON recipe loading rejects duplicate keys and nonfinite numbers as well as
approximate coefficients, so execution and certificates share an unambiguous
specification.

The combined suite records ordinary stage exceptions, including their traceback,
in `suite-manifest.json`, continues independent stages, and exits nonzero if any
stage fails. A missing earlier reference report preserves already collected
arithmetic JSON. If the combined page cannot render, its fallback page still
links completed evidence. Interrupted runs retain their last completed manifest
checkpoint; this does not yet resume a partially completed search.

## Shared Python interface

```python
from fractions import Fraction
from matrix_multiplication import multiply, strassen

a = [[Fraction(1, 3), 2], [3, 4]]
b = [[5, 6], [7, 8]]
product = multiply(a, b, algorithm="alphaevolve", cutoff=1)
same_product = strassen.multiply(a, b, cutoff=1)
```

Each module provides `multiply(a, b, *, cutoff=1, instrumentation=None)`. Inputs are nonempty rectangular
lists or tuples of rows, containing integers, standard-library rational numbers,
finite floats, or finite complex values. The left column count must equal the
right row count. Empty matrices and zero-length rows are rejected because these
representations cannot express their dimensions unambiguously. Inputs are never
modified; results are new lists of rows.

Recursive algorithms pad to a square power of two (Strassen variants) or four
(AlphaEvolve). Smirnov pads along its repeating rectangular stage dimensions;
after three stages the accumulated dimension is 54 on every axis. Each algorithm
crops the product to the requested shape. This supports odd and rectangular shapes,
but padding can be expensive for skinny matrices. `cutoff` selects the largest
classical leaf size; the default of 1 exercises full recursion. Naive validates
this option for interface consistency but always performs classical multiplication.
These are readable research implementations; smaller arithmetic exponents do
not guarantee smaller Python runtimes. Optional counters, primitive profiles, and
bounded scalar traces are now available; compiled/GPU execution remains future work.

## Instrumented reference experiments

Readable formulas use a hierarchy of shared primitives: scalar arithmetic,
dot products and block operations, sparse bilinear schemes, recursive traversal,
and public experiment adapters. Every Python module, class, and function has
documentation. [INSTRUMENTATION.md](INSTRUMENTATION.md) explains the hierarchy,
counting conventions, trace format, and profiling limits.

```python
from matrix_multiplication import Instrumentation, multiply

measurement = Instrumentation()
product = multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]],
                   algorithm="strassen", instrumentation=measurement)
print(measurement.snapshot())
```

```
python explore.py --algorithm all --instrument
python explore.py --algorithm strassen --report experiment.json
python explore.py --algorithm all --counts-only --measure-memory --report experiment.json
python explore.py --algorithm strassen --trace trace.json --trace-limit 1000 --trace-values
python tools/check_documentation.py
```

Reports count data field products, coefficient scaling, additions/subtractions,
normalizations, exposed real-domain operations, opaque native complex operations,
and structural cell allocations/copies. They include primitive call counts,
inclusive/exclusive timings, and products by recursion depth. Verification and
coefficient identity validation are separate from measured product arithmetic.

JSON reports also record the formula-derived mathematical exponent, base rank,
and recurrence separately from execution diagnostics: algorithm-call elapsed
time, verification error, and conventional equivalent field-operation rate.
`--counts-only` disables per-primitive clocks while retaining arithmetic counters;
the independent whole-call timer remains. `--measure-memory` optionally records
peak traced Python allocations, including setup and observation. This is neither
process/GPU peak memory nor hardware memory traffic. Those hardware metrics are
explicitly marked unmeasured rather than estimated from cell counters.
Counters do not change formulas or exponents. Counts exclude interpreter/control
work; profiles measure instrumented reference code, not optimized kernel speed.
Bounded traces store a prefix and report dropped events; an API sink can stream
complete traces without retaining them. Scalar values are captured only on request.

## Automatic experiment reports

Run the experiment and produce JSON plus its readable HTML report in one command:

```
python explore.py --algorithm all --counts-only --measure-memory --report experiment.json
```

`explore.py --report experiment.json` writes that JSON and automatically prints the
path to `outputs/YYYY-MM-DD_HHMMSS_microseconds/index.html`. `sweep.py` always
generates both JSON and HTML automatically. There is no second rendering command;
the former standalone rendering script has been replaced by the shared
`matrix_multiplication.reporting` library. Plain `explore.py` without `--report`
remains a terminal-only calculation.
Open that file in a browser. Each run gets its own dated directory and an unchanged
`measurements.json` snapshot alongside the HTML; previous reports are not replaced.
The HTML contains its own CSS and SVG charts and needs no internet connection,
JavaScript, plotting dependency, or image files. Browser printing can produce a PDF.

Charts use Masataka Okabe and Kei Ito's [Color Universal Design palette](https://jfly.uni-koeln.de/color/#pallet),
with distinct marker shapes and solid/dashed styles in addition to color. Hover
a line chart's legend entry to isolate its series and associated model curves;
moving away restores the full chart. Keyboard users can focus the legend buttons
with Tab for the same effect. Isolation is local to that graph, and printing
always shows every series. This uses CSS only and works in current browsers from
the offline HTML file. Both axes name their quantities; log-log views explicitly
label logarithmic scales. A note identifies identical overlapping series.

Graphs show formula-derived square exponents, executed data products, and products
by recursion depth, with separate runtime and Python memory charts. Tables expose
arithmetic breakdowns, accuracy, exact metadata/counters, and primitive profiles.
Missing measurements remain unavailable; unlike experiments are flagged, and
field work is distinguished from component/atomic work. The renderer presents
the source report's evidence rather than independently certifying it.

`--output-root path` on either command overrides the parent output directory.
The reporting library also accepts trace JSON and older reports;
older metadata is left unavailable rather than inferred. `outputs/` is ignored
by Git. To publish a result, manually copy the chosen output directory into
`reports/`; that folder is not ignored and retains its adjacent source snapshot.

### One command for every research report family

```sh
python sweep.py
```

This is now the complete research suite. One command creates one dated directory
and a combined HTML entry page. It retains every existing arithmetic graph and
table, then adds links to these automatically generated artifacts:

- Seeded exact decomposition discovery, its replay log, and a replay experiment.
- A tensor-product matrix recipe; generated matrix recipes automatically join
  the measured size/cutoff comparison.
- CW polynomial recovery experiments for q = 1, 2 and powers = 1, 2, with finite
  interpolation cost distinguished from border rank.
- Portable exact certificates for selected algorithms, custom compositions,
  and generated matrix recipes.
- Separate integer and floating-point diagnostic pairs, with primitive profiles,
  accuracy, Python allocation peaks, and bounded scalar/primitive traces.
- Exact finite rank targets, plus a manifest recording settings, outcomes, and
  artifact fingerprints. Completed stages survive a failure elsewhere.

The default arithmetic sizes remain 1, 2, 4, 8, 16, with cutoffs 1, 2, 4.
Discovery defaults to the bounded 2-by-2 experiment; increasing `--sizes` does
not automatically increase search sizes or CW powers. Profiles and memory are
secondary measurements of the separate small pairs, not of the count-only size
sweep. A search that misses its target is reported honestly as a completed
experiment; invalid identities, wrong products, and arithmetic mismatches cause
a failed suite and nonzero exit status. All outputs remain ignored by Git.

```sh
python sweep.py --sizes 2 4 8 --cutoffs 1 2
python sweep.py --search-seeds 0 1 2 --search-steps 5000 --search-restarts 20
python sweep.py --search-size 3 --search-target-rank 23 --search-max-height 4 --search-extra-rank 2
python sweep.py --search-start path/to/real-rational-scheme.json --search-target-rank 38 --search-max-height 8
python sweep.py --cw-parameters 2 --cw-powers 1 2 --pair-trace-limit 1000
```

These search settings are finite budgets, not promises of new discoveries.
`--search-start` accepts a verified real rational square or rectangular recipe.
For a rectangular result, the suite verifies its two cyclic orientations and
automatically measures/certifies the balanced square cycle. This connects searches
of small rectangular bases to our square-exponent goal.
Use `--measurements-only` to request the original arithmetic report without
running the other report families. Individual explorer/construction/certificate
commands remain available for focused work. See [CONSTRUCTIONS.md](CONSTRUCTIONS.md)
for the generator and [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the expanded paths.

### Measured arithmetic growth across sizes and leaf blocks

```
python sweep.py
python sweep.py --sizes 1 2 3 4 5 8 9 16 --cutoffs 1 2 4
python sweep.py --sizes 4 16 64 --cutoffs 1 4 --algorithms alphaevolve
python sweep.py --sizes 1 2 3 4 5 8 9 16 --cutoffs 1 2 4 --reference-report experiment.json
```

The arithmetic-growth stage executes each configuration, counts arithmetic with primitive clocks
disabled, verifies every output against the independent exact oracle, and writes
JSON plus HTML to one dated `outputs/` directory. These graphs measure work rather
than elapsed time. Larger sizes are supported and can require substantial resources.
The default includes Smirnov. Use `--sizes` and `--cutoffs` for longer experiments,
or `--algorithms naive smirnov` to select those built-in algorithms, without
editing the script. The full suite also measures its generated recipes;
`--measurements-only --algorithms naive smirnov` compares only those two.
Every algorithm and cutoff uses the same reproducible signed-integer inputs at a
given size. Repeating identical inputs is unnecessary for deterministic counts.

The sweep also avoids repeating equivalent cutoffs for the same algorithm and
input. Naive's cutoff always has no effect; recursive algorithms can share a
stopping point too. For example, AlphaEvolve cutoffs 1, 2, and 3 reach the same
leaves, and the default Smirnov sizes use the same stages at cutoffs 1, 2, and 4.
We compare padded dimensions, ordered recursion stages, and actual leaf dimensions
before reusing one successful, verified execution. Equal operation totals alone
are insufficient. Different sizes and algorithms always run independently.

The JSON and HTML retain every requested cutoff, its own predictions, and its
curves. The report distinguishes **actual executions** from **reused measurements**;
the **Measurement source** column identifies the cutoff that supplied the captured
counters and accuracy check. Errors and arithmetic-budget mismatches are never
reused. Unfamiliar or replaced execution callbacks are run independently.
To deliberately execute every cutoff, use `python sweep.py --repeat-equivalent-cutoffs`.
Python callers can pass `reuse_equivalent_cutoffs=False` to `collect_work`.

The report overlays captured **data multiplication** counts with a smooth ideal
model and the discrete recurrence's prediction after padding. Both models use
declared algorithm parameters; neither substitutes for executed counters. Separate
plots show total field arithmetic and component/atomic steps, which describe
overlapping layers and must not be added together. Coefficient multiplications are
fixed constants times values; data multiplications multiply two data-dependent values.
Expand **All arithmetic steps for these block sizes** to compare those total-work
curves with their models too. They include the lower-order additions, coefficient
scaling, and normalization at each recursive level. The model constants come from
our primitive formulas, not a fit to the observations; the captured totals are
checked independently against the padded recurrence. Both linear and log-log
views are available. The leaf cutoff controls when recursion stops and classical
multiplication begins; it is distinct from the scheme's fixed base block size.
Smirnov and composed schedules use exact predictions for every executed size,
including their actual padded dimensions and stage-by-stage arithmetic budgets.
Their smooth ideal curves are omitted because the constants depend on which
stage reaches the cutoff; the full-cycle theoretical exponent is still shown.

The theoretical exponent is calculated from the declared base dimension and
product count, such as `log_2(7)`. The measured log-log slope comes from actual
counts over the sampled size range and is labelled as finite-range evidence.
A 2-by-2 execution alone cannot establish an asymptotic exponent. Padding,
cutoffs, and lower-order operations can make a measured slope differ from the
family exponent. Fixed classical leaf cutoffs preserve the family exponent;
allowing the cutoff to grow with the input changes that assumption.

The sweep report retains the exponent overview, full arithmetic breakdown,
accuracy checks, recursion-depth charts, all operation/structural counters, and
primitive call/profile evidence. Expand **Per-sample primitive and recursion
evidence**, then choose an algorithm/cutoff and size. Primitive clock values are
zero because those clocks are disabled, while call counts remain measured.

Use `--reference-report path/to/earlier.json` to include an earlier single-pair
experiment's timing, memory, accuracy, and primitive evidence in the same HTML;
repeat the option to include several. Those measurements stay labelled as a
separate experiment and never become sweep measurements. The combined output JSON
embeds each original reference JSON verbatim with its source hash, so a copied
report remains self-contained. No earlier file is modified, no previous dated
report is overwritten, and no earlier report is included implicitly.

The configured cutoff is the maximum classical leaf size. For radix-four
recursion, cutoff 2 or 3 still gives size-1 leaves; cutoff 4 permits size-4 leaves.
Both an 8-by-8 and a 16-by-16 AlphaEvolve input pad to size 16, producing a measured
plateau. The report shows actual padded sizes and leaf sizes to explain this.

## Import and compose schemes

### Verified lower-exponent Smirnov baseline

Our independent implementation now executes Smirnov's explicit rank-40 recipe.
It composes (3,3,6), (3,6,3), and (6,3,3) stages into a square dimension-54,
rank-64,000 base, with exponent **2.7742999804**, below AlphaEvolve's 2.7924812504.
This reproduces a known construction, not a new mathematical bound or a public
implementation record. Required rectangular blocks are handled by the same
observable split/join, linear-combination, scalar, and recursive primitives.
The extension is tied to this construction, rather than a general rectangular
algorithm project.

```
python tools/import_smirnov.py --download
python explore.py --algorithm smirnov --counts-only --report outputs/smirnov.json
python tools/check_smirnov.py
python -m matrix_multiplication.certificates create --composition smirnov
```

The bounded test tool uses only 2x2 and 4x4 matrices; it never invokes sweep.py or
reads its defaults. `python tools/check_smirnov.py --full-cycle` explicitly tests
one 54x54 base and its 64,000 executed products against classical multiplication.
Both modes automatically produce dated HTML and JSON. Smirnov is part of the
default suite: `--algorithm all` and `python sweep.py` include it. Python callers
can use `multiply(a, b, algorithm="smirnov", instrumentation=observer)`.
`Composition.from_spec("smirnov")` returns the dedicated balanced cycle; it can
also be passed to `collect_work` with explicitly bounded sizes for comparison.
The bounded checks are independent of sweep.py defaults.

Coefficients and the original BSD license remain in ignored Background Research.
The downloader pins fast-matmul revision `132358be7a6ab57c83dd5ce70ece460bd1ec5dfa`
and checks the coefficient file's SHA-256 before parsing integer/fraction tokens.
It never executes upstream programs. Provenance, transformations, and the full
BSD notice are included in derived recipe data and certificates. Credits go to
Alexey V. Smirnov for the construction and Austin R. Benson/Grey Ballard for the
published coefficient encoding. Our interpreter, cyclic transformations, and
importer are independently written.

The importer and both certificate checkers verify all 2,916 coefficient
relationships in each orientation. A balanced-cycle check verifies that all
three accumulated dimensions are 54 before claiming the square recurrence.
Padding follows successive stage dimensions and can yield rectangular working
matrices even for square inputs; reports retain all three padded dimensions.
For example, a 16x16 input pads to 54x54. Its successive block dimensions are
18x18x9, 6x3x3, and 1x1x1, so cutoffs 1, 2, and 4 all execute the same 64,000
data products. Their curves overlap because each stops at the same stage.
At small sizes the substantial scale/add work can exceed classical arithmetic,
despite the better asymptotic exponent. Counts include that overhead explicitly.
Rectangular base data uses schema version 2 with `dimensions: [m,k,n]` in place
of `radix`. This backend supports exact real coefficients, including rational
ones; approximate constructions and complex rectangular coefficients are outside
its current contract.

The first research checklist item is implemented in
`matrix_multiplication/scheme_io.py` and `matrix_multiplication/composition.py`.
The engine accepts our built-in formulas and imported **square** bilinear schemes.
It runs our existing primitives, rather than calling someone else's multiplier.
Rectangular input matrices are supported through padding. Version-2 rectangular
base data is used by the Smirnov cycle described above.

```
python explore.py --composition strassen,alphaevolve --counts-only --report outputs/composition.json
python explore.py --composition schemes/classical-2.json,strassen --counts-only --report outputs/imported.json
python sweep.py --sizes 1 2 3 4 5 8 9 16 --cutoffs 1 2 4 --algorithms naive strassen alphaevolve --composition strassen,alphaevolve
```

`explore.py --composition` replaces the usual algorithm choice. `sweep.py
--composition` adds a plan alongside the selected algorithms; repeat that option
to compare multiple plans. Both commands automatically generate dated HTML reports
in `outputs/`. Create `outputs/` first if requesting a JSON path in a fresh checkout.

The comma-separated stages repeat by recursion depth. For example, Strassen then
AlphaEvolve splits 16 → 8 → 2 → 1 at cutoff 1. Padding follows the smallest prefix
product of that same radix schedule covering every input dimension. The cutoff
still means the largest square block handled by the classical leaf algorithm.
All recursive child products at a given depth use the same stage.

One complete Strassen/AlphaEvolve cycle has radix 2×4 = 8 and rank 7×48 = 336.
Its exponent is log₈(336), about 2.7974, derived from the **repeating** schedule,
not fitted from executions. This creates no new exponent record: mixing existing
schemes cannot beat the best constituent exponent. It lets us explore constructions
and their actual arithmetic costs without duplicating the execution engine.

Reports retain each imported recipe and its provenance, the ordered schedule,
per-level sizes and product counts, and independent predicted versus captured
data products, field steps, and atomic steps. The sweep includes exact discrete
predictions. Mixed schedules omit a smooth ideal curve because padding and cutoff
phases change the finite constants; observed curves and finite-range fits remain.
Any stage with genuinely complex coefficients makes the whole plan use exact
Gaussian pairs, including real-coefficient stages, with that extra arithmetic counted.

Python callers can use the same matrix/cutoff/instrumentation contract:

```python
from matrix_multiplication.composition import Composition
from matrix_multiplication import Instrumentation

plan = Composition.from_spec("strassen,alphaevolve")
observer = Instrumentation(profile=False)
result = plan.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]], instrumentation=observer)
expected_work = plan.budget(2, 2, 2, cutoff=1)
cycle_model = plan.model()
```

`schemes/classical-2.json` is our own small eight-product example, supplied under
this repository's GPL-3.0 license. The interchange format is JSON with
`schema_version: 1`, `name`, `radix`, positive integer `divisor` (default 1), and
equal-length `left`, `right`, and `output` arrays. Each array contains one sparse
column per product; each column contains `[block_index, coefficient]` terms.
All indices use **row-major** order, including the output. Each coefficient is
an integer, or `{"real": [numerator, denominator], "imag": [numerator, denominator]}`
with exact integer components and positive denominators. Floating coefficients,
duplicate indices, malformed data, and incorrect tensor identities are rejected.
The result assembled from the output columns is divided by `divisor` once per node.

`load_scheme(path)` checks the complete tensor identity before returning a
`BilinearScheme`; `save_scheme(scheme, path, provenance=...)` verifies and exports
one. Metadata in `provenance` can carry the source URL, author, license, and notices;
the importer preserves it in report JSON but does not resolve licensing terms.
Keep third-party coefficient files in `Background Research/` and retain required
notices when publishing. No new external implementation was used for this feature.
These checks establish a finite identity. Portable reproducible certificates are
available as described below.

## Produce and independently check a certificate

```
python -m matrix_multiplication.certificates create --composition strassen,alphaevolve
python -m matrix_multiplication.certificates create --composition schemes/classical-2.json,strassen
```

Creation runs both the existing sparse identity verifier and a separate dense
basis-matrix checker. It writes a dated `outputs/certificate_.../` directory with
the exact ordered recipes, saved evidence, a standalone `check_certificate.py`,
a readable mathematical argument, the repository license, and file fingerprints.
The first command certifies the repeating Strassen/AlphaEvolve mathematical cycle.
With no `--composition`, creation includes Strassen, Strassen-Winograd, and
AlphaEvolve as a repeating three-stage cycle. `--output path` chooses a new package
directory explicitly; existing directories are never overwritten.

To check a package from this repository:

```
python -m matrix_multiplication.certificates verify outputs/certificate_DATE/certificate.json
```

Or copy the package elsewhere and run its standalone checker, without installing
this repository or any third-party library:

```
python path/to/package/check_certificate.py
```

The checker re-evaluates every basis-input pair and every output using exact
rational complex arithmetic. It checks 64 coefficient relationships for each
radix-2 scheme and 4,096 for AlphaEvolve. Saved pass records and fingerprints alone
are not accepted: incorrect coefficients fail even if their hashes are updated.
It also recomputes the ordered cycle's radix, rank, and recurrence. Exit status
is nonzero on any mismatch. Certificates omit timestamps, so unchanged recipes
and provenance reproduce the same semantic fingerprint.

Built-in Strassen and Winograd recipes are symbolically extracted from their
actual maintained step functions; their source-file hashes are retained.
Imported provenance and AlphaEvolve source notices are retained too. Packages
stay ignored by default. Before publishing one, review third-party reuse terms
and include their required notices/licenses in addition to the bundled GPL license.

These are **exact Python verification packages, not Lean proofs**. Their validity
relies on the documented algebraic argument and the correctness of the readable
checker and Python arithmetic. The second checker does not import the original
verifier, scheme parser, multiplier, or scalar primitives. Recursive composition
is justified by induction on block depth; the package does not formally verify
the Python interpreter, execution code, operation counters, or numerical stability.
See [CERTIFICATES.md](CERTIFICATES.md) for the complete argument and trust boundary.

## Construct and search for new recipes

The generator now builds exact rational tensor constructions and searches for
product-count reductions. It independently discovers a seven-product 2-by-2
scheme from the eight classical products; no Strassen coefficients are supplied.
It also constructs CW polynomial tensor powers and performs exact coefficient
recovery, with the finite interpolation overhead explicitly reported.

```sh
python -m matrix_multiplication.constructions search --target-rank 7 --certificate
python -m matrix_multiplication.constructions product --left strassen --right strassen --certificate
python -m matrix_multiplication.constructions cw --q 2 --power 2
```

Each command automatically writes its data and an HTML research summary to a
new dated `outputs/construction_.../` directory. Search/product matrix recipes
can be measured with the existing sweep, using the generated file's actual path:

```sh
python sweep.py --sizes 2 4 8 --cutoffs 1 2 --algorithms naive strassen --composition outputs/construction_DATE_TIME/scheme.json
```

Searches support explicit rank or rational exponent targets, bounded restarts,
coefficient limits, and optional temporary rank increases. All exported matrix
recipes pass complete exact identity verification. A CW research tensor is kept
distinct from a matrix recipe; its border rank is not substituted for its finite
execution cost. **No construction below 2.75 has been achieved by this machinery
yet.** Our verified baseline remains Smirnov's 2.7742999804.

See [CONSTRUCTIONS.md](CONSTRUCTIONS.md) for commands, replay/certificates,
the algebra of each reusable primitive, search limitations, remaining extraction
work, and the distinction from generating the OpenAI witness.

## Research work checklist

- [x] Build our own sparse scheme evaluator, recursive composition engine, and operation accounting. Version-1 square scheme import verifies exact identities; configurable repeating schedules share the existing traversal and include independent work budgets.
- [x] Establish complete exact identity verification with reproducible certificates. Exact square recipes and repeating cycles have portable Python packages, an independent basis checker, fingerprint binding, and a documented composition argument; these are not Lean proofs.
- [ ] Reproduce a licensed baseline, then investigate the stronger aggregation constructions. The verified Smirnov baseline now achieves exponent 2.7742999804; stronger aggregation constructions remain research work.
- [ ] Add custom CUDA execution and memory scheduling when needed to run larger mathematical constructions; hardware tuning is secondary.
- [ ] Search for product-count reductions or better constructions, independently verifying every candidate. The exact rational generator, flip/reduction/split search, replay logs, and CW polynomial recovery are implemented; a stronger exponent result remains open.

The new construction machinery opens these more specific tasks, in research
priority order. Only verified finite matrix identities and their accounted
recurrences can establish a better exponent:

- [x] Run every supported report family from one `python sweep.py` command, with
  a combined report, generated recipes measured automatically, and stage outcomes.
- [ ] Implement a CW sector/extraction generator, verify surviving independent
  matrix products, then implement their packing/recovery into dense multiplication.
  Keep the finite extraction and interpolation overhead in the recurrence.
- [ ] Search reductions of existing explicit real schemes, especially the smaller
  Smirnov rectangular base. A cyclic rank-39 base would give about 2.7553; rank
  38 would give about 2.7357. These are targets, not discovered coefficients.
- [ ] Run reproducible search campaigns across starting recipes, seeds, coefficient
  limits, temporary rank increases, and symmetry transformations. Store checkpoints
  and resume walks; distinguish exact duplicates from unproved equivalences.
- [ ] Add numerical candidate proposals followed by exact rational/algebraic
  reconstruction and complete verification. Numerical fits alone are insufficient.
- [ ] Extend construction/search to Q(i), then explicit algebraic coefficient
  fields where needed, with documented exact arithmetic and cost models.
- [ ] Represent large tensor powers and restrictions with compact index rules and
  streaming execution, so construction size is not dictated by full materialization.
- [ ] Certify the complete construction chain: polynomial identities, exact
  recovery, restrictions, direct-sum independence, and packing, not just final bases.
- [ ] Turn exponent goals into finite size/rank search schedules, using proved
  lower bounds to exclude impossible targets and finite cost budgets to compare
  candidates. The automatic rank-target table is the first implemented piece.
- [ ] Audit the OpenAI proof for an effective finite-size bound or constructive
  witness route. Keep this separate from heuristic finite searches and support
  whatever exact coefficient field a verified witness actually needs.

The immediate research sequence is general coefficient import and exact
certificates, a lower-exponent executable baseline, then explicit construction
and extraction machinery for stronger bounds. The full route, including the
older asymptotic constructions and the 2.25 witness problem, is in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Correctness and accuracy

The tests check every base matrix-unit pair: 16 pairs for each 2-by-2 scheme and
256 pairs for AlphaEvolve's 4-by-4 scheme. Since these formulas are bilinear,
those checks establish their complete base identities. Block recursion preserves
these identities inductively; zero padding and cropping preserve the requested
product. The tests additionally exercise recursion, shapes, identity and zero
matrices, seeded random inputs, exact fractions, large integers, complex values,
input immutability, malformed inputs, and coefficient corruption.

Integer and rational inputs are checked by exact equality. AlphaEvolve represents
complex intermediates as real/imaginary pairs and stores doubled Gaussian-integer
coefficients, preserving exact arithmetic for real integer and Fraction inputs.
Each recursive level removes a common factor of eight exactly. Built-in complex
inputs retain Python's floating-point semantics. The 48-product count describes
complex-field products; the implementation expands a complex product into four
real products, and also performs coefficient scaling and additions.

For float and complex inputs, an independent Fraction reference represents the
actual binary input values exactly. Every result entry must satisfy
`abs(error) <= atol + rtol * abs(reference)` (defaults: 1e-9 for both tolerances).
No finite test suite proves all floating-point inputs accurate: cancellation,
overflow, and operation ordering matter even for the naive algorithm. A regression
case deliberately demonstrates cancellation and verifies that the harness reports
the error instead of accepting naive output as ground truth.

## Why there is no 2.25 implementation

The October 2026 OpenAI manuscript proves an asymptotic arithmetic bound of
`O_epsilon(n^(9/4 + epsilon))` over the complex numbers. Its final step chooses a
base size `u` and a rank decomposition with fewer than `u^(9/4 + delta)` products
by existence; it does not supply that size or the decomposition's coefficients
(Section 5, page 9). This does not give us an explicit finite multiplication scheme
to implement and test. The explorer reports this limit rather than labelling a
different algorithm as 2.25. The earlier Coppersmith-Winograd/laser-method exponent
bounds likewise are not represented by Strassen-Winograd, which is a different
algorithm. Deriving usable finite schemes from those results is further research.

## Sources

https://en.wikipedia.org/wiki/AlphaTensor
https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication
https://pubmed.ncbi.nlm.nih.gov/36198780/
https://arxiv.org/pdf/1612.05766
https://arxiv.org/html/2608.16884v1
https://ieee-focs.org/FOCS-2018-Papers/pdfs/59f580.pdf
https://arxiv.org/pdf/2410.21050
http://par.nsf.gov/servlets/purl/10524471
https://www.quantamagazine.org/new-breakthrough-brings-matrix-multiplication-closer-to-ideal-20240307/
https://binaryverseai.com/openai-matrix-multiplication/
https://siliconangle.com/2026/10/07/openai-publishes-722-ai-generated-math-discoveries-in-major-scientific-milestone/
https://en.wikipedia.org/wiki/Strassen_algorithm
https://www-auth.cs.wisc.edu/lists/theory-reading/2009-December/pdfmN6UVeUiJ3.pdf
https://people.math.harvard.edu/~knill/history/matrix/

## Additional background references

Local copies are stored in `Background Research/`, which is excluded from Git. The directory includes `SOURCES.md` with download details and notes on why these references were selected. The existing literature review is preserved there.

### Recent primary results

- [An Upper Bound of 9/4 for the Matrix Multiplication Exponent](https://raw.githubusercontent.com/openai/math/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/paper.pdf)
- [Complex Matrix Multiplication Below 2.258 and Rectangular Bounds](https://raw.githubusercontent.com/openai/math/main/preprints/Complex-Matrix-Multiplication-Below-2.258-and-Rectangular-Bounds-September-24-2026/Complex-Matrix-Multiplication-Below-2.258-and-Rectangular-Bounds-September-24-2026.pdf)
- [Staggered extraction for exact matrix multiplication over every field](https://raw.githubusercontent.com/openai/math/main/preprints/Staggered-extraction-for-exact-matrix-multiplication-over-every-field-September-24-2026/Staggered-extraction-for-exact-matrix-multiplication-over-every-field-September-24-2026.pdf)
- [OpenAI family 107 formalization notes](https://raw.githubusercontent.com/openai/math/main/lean/docs/107.md)

### AI-assisted discovery and earlier exponent improvements

- [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/pdf/2506.13131)
- [DeepMind's published AlphaEvolve coefficient data and verification notebook](https://github.com/google-deepmind/alphaevolve_results)
- [AlphaEvolve notebook download](https://raw.githubusercontent.com/google-deepmind/alphaevolve_results/main/mathematical_results.ipynb)
- [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/pdf/2404.16349)

### Numerical accuracy and practical performance

- [Fast matrix multiplication is stable](https://arxiv.org/pdf/math/0603207)
- [Improving the numerical stability of fast matrix multiplication](https://arxiv.org/pdf/1507.00687)
- [Strassen's Algorithm Reloaded](https://jianyuhuang.com/papers/sc16.pdf)

### Alternative approaches and barriers

- [A group-theoretic approach to fast matrix multiplication](https://arxiv.org/pdf/math/0307321)
- [Barriers for fast matrix multiplication from irreversibility](https://arxiv.org/pdf/1812.06952)
- [Limits on the Universal Method for Matrix Multiplication](https://arxiv.org/pdf/1812.08731)

### Release context and catalogue

- [OpenAI mathematical manuscript catalogue](https://raw.githubusercontent.com/openai/math/main/CONTENTS.md)
- [OpenAI mathematics release README](https://raw.githubusercontent.com/openai/math/main/README.md)

### Additional download endpoints for the original sources

- [AlphaTensor full paper (Nature)](https://www.nature.com/articles/s41586-022-05172-4.pdf)
- [PubMed record export (NCBI XML)](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=36198780&retmode=xml)

### Further implementation research

Research findings dated 11 October 2026 are saved locally in
`Background Research/IMPLEMENTATION_RESEARCH.md`.

- [Benson and Ballard's fast-matmul implementation](https://github.com/arbenson/fast-matmul)
- [Generated algorithms, including Smirnov's 40-product schemes](https://github.com/arbenson/fast-matmul/tree/master/algorithms)
- [Pinned Smirnov coefficient tables](https://github.com/arbenson/fast-matmul/blob/132358be7a6ab57c83dd5ce70ece460bd1ec5dfa/codegen/algorithms/smirnov336-40-960) and [pinned BSD license](https://github.com/arbenson/fast-matmul/blob/132358be7a6ab57c83dd5ce70ece460bd1ec5dfa/LICENSE): importer inputs retained locally with their notices.
- [Smirnov 3-by-3 times 3-by-6 explicit algorithm encodings](https://fmm.univ-lille.fr/3x3x6.html)
- [DeepMind's AlphaTensor algorithms and benchmarking code](https://github.com/google-deepmind/alphatensor)
- [Practical fast multiplication code generation](https://github.com/flame/fmm-gen)
- [Matrix multiplication catalogue and exploration toolchain](https://github.com/solven-eu/matmulcatalog)
- [Lean proof extension of the 9/4 bound to arbitrary fields](https://github.com/selanavot/matrix-multiplication-all-fields)
- [OpenAI's arithmetic-program model](https://github.com/openai/math/blob/main/lean/OAI/LinearAlgebra/MatrixMultiplication/Model.lean)
- [Practical recursive 48-product algorithms (2026)](https://arxiv.org/abs/2609.12027v2)
- [Williams's matrix multiplication paper, corrected 2014 version of the 2012 work](https://cs.stanford.edu/~virgi/matrixmult-f.pdf)
- [Schönhage's Partial and Total Matrix Multiplication (1981)](https://epubs.siam.org/doi/abs/10.1137/0210032)
- [Benson and Ballard's practical parallel implementation study](https://www.cs.cornell.edu/~arb/papers/fast-matmul-ppopp2015.pdf)
- [Benson and Ballard's study, arXiv download mirror](https://arxiv.org/pdf/1409.2908)
- [Bläser's Fast Matrix Multiplication survey](https://theoryofcomputing.org/articles/gs005/gs005.pdf)
- [Flip Graphs for Matrix Multiplication](https://arxiv.org/abs/2212.01175), Manuel Kauers and Jakob Moosbauer: mathematical background for our independently written exact search engine; no external search implementation is used.
- [On Degeneration of Tensors and Algebras](https://drops.dagstuhl.de/opus/volltexte/2016/6434/pdf/LIPIcs-MFCS-2016-19.pdf), Markus Bläser and Vladimir Lysikov, MFCS 2016, equation (2): credited source of the CW polynomial formula; publication licensed CC-BY. Our code implements the algebra independently.
- [More Asymmetry Yields Faster Matrix Multiplication](https://arxiv.org/abs/2404.16349), Josh Alman and coauthors: primary background for CW tensor powers and extraction; its limiting exponent is not claimed for our finite experiments.

### Record audit and implementation attribution

- [Local Improvements to Trilinear Aggregation (LITA)](https://github.com/khoruzhii/lita), Kirill Khoruzhii, Luzian Serafin, Patrick Gelß, and Sebastian Pokutta. [Pinned revision](https://github.com/khoruzhii/lita/tree/472a79fbc8248423a97bebb3aacbc89e769b9b3e), dated 7 October 2026: public rational schemes and an evaluator; the 32-by-32 scheme has 14,196 products, implying exponent 2.7586393718. No clear license was found in the inspected repository; downloaded coefficients are retained only as local research references and are not distributed here. Independent exact modular sample checks are recorded in `Background Research/lita-audit-results.json`; these are not a complete identity proof.
- [Towards Faster Feasible Matrix Multiplication by Trilinear Aggregation](https://arxiv.org/abs/2508.01748), Oded Schwartz and Eyal Zwecher. Primary background for the recent aggregation constructions.
- [fast-matmul](https://github.com/arbenson/fast-matmul), Austin R. Benson and Grey Ballard: [BSD 2-Clause license](https://github.com/arbenson/fast-matmul/blob/master/LICENSE), copyright Sandia Corporation. The source snapshot and its license are stored locally for reference; upstream execution code is not incorporated into our libraries.
- [AlphaTensor](https://github.com/google-deepmind/alphatensor), Google DeepMind: [Apache 2.0 license](https://github.com/google-deepmind/alphatensor/blob/main/LICENSE). Referenced implementations and data have not been incorporated into our current libraries.
- [AlphaEvolve results](https://github.com/google-deepmind/alphaevolve_results), Google DeepMind: [Apache 2.0 license](https://github.com/google-deepmind/alphaevolve_results/blob/main/LICENSE). Our importer and evaluator are independent code; the local notebook and derived coefficient data retain source attribution and stay excluded from Git. The upstream license is saved alongside these references. Redistribution must include the license, retain applicable notices, and identify modifications, as required by Apache 2.0.
- [Licensing a public repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository): GitHub's explanation of the distinction between public code and licensed open-source code.

### Production GPU and graphics implementations

Research notes dated 11 October 2026 are saved locally in
`Background Research/PRODUCTION_IMPLEMENTATIONS.md`. Documentation, selected
reference sources, and license notices are retained in the ignored research
directory; no vendor implementation has been incorporated into our libraries.

- [NVIDIA's matrix multiplication performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html): tiling, arithmetic intensity, and the effect of matrix shape on throughput. Its hardware-specific examples describe older GPUs, rather than measurements of our RTX 3090.
- [CUTLASS 3.x GEMM model](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) and [efficient GEMM implementation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html): the classical multiplication loops organized around GPU memory and execution hierarchies.
- [NVIDIA CUTLASS C++ implementation](https://github.com/NVIDIA/cutlass), NVIDIA Corporation and affiliates: [BSD 3-Clause license](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt). The license explicitly lists separate NVIDIA EULA terms for `python/CuTeDSL`; the C++ license must not be assumed to cover that directory. Referenced for explanation, not used as our multiplication implementation.
- [NVIDIA cuBLAS API and tuning documentation](https://docs.nvidia.com/cuda/cublas/index.html): kernel selection, numerical modes, and the complex `gemm3m` Gauss reduction. This documents library behavior without publishing every kernel's source.
- [DirectXMath](https://github.com/microsoft/DirectXMath), Microsoft: [MIT license](https://github.com/microsoft/DirectXMath/blob/main/LICENSE). Its [matrix routines](https://github.com/microsoft/DirectXMath/blob/main/Inc/DirectXMathMatrix.inl) provide concrete examples of fixed-size classical multiplication with SSE, AVX2, and ARM NEON variants. A reference snapshot retains the upstream license and copyright header; no code was copied into our implementation.
- [Unity's Matrix4x4 reference source](https://github.com/Unity-Technologies/UnityCsReference/blob/master/Runtime/Export/Math/Matrix4x4.cs) and [MultiplyPoint3x4 documentation](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/matrix4x4/multiplypoint3x4): specialized affine transforms. Unity's [reference-only license](https://github.com/Unity-Technologies/UnityCsReference/blob/master/LICENSE.md) applies; this is a local reading reference, not an open-source implementation dependency or redistributed source.
- [Implementing Strassen's Algorithm with CUTLASS on NVIDIA Volta GPUs](https://arxiv.org/abs/1808.07984), Jianyu Huang, Chenhan D. Yu, and Robert A. van de Geijn (2018): measured research combining Strassen with GPU tiling and fused operations. Its reported V100 speedups are not evidence of performance on current cuBLAS or our RTX 3090.
