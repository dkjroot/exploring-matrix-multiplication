# Exploring matrix multiplication

Four implementations use Python arithmetic and lists, without NumPy, SciPy,
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

## Run the explorer

The AlphaEvolve notebook and extracted coefficients are local research data,
excluded from Git. They have already been imported in this checkout. In a fresh
checkout, import them explicitly with the standard-library-only tool:

```
python tools/import_alphaevolve.py --download
```

If you downloaded the notebook manually, use
`python tools/import_alphaevolve.py "path/to/mathematical_results.ipynb"` instead.
The importer reads literal data and never runs notebook code. It verifies all
4,096 coefficients of the rank-48 tensor identity before saving the data.

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
(AlphaEvolve), then crop the product. This supports odd and rectangular shapes,
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

### Measured arithmetic growth across sizes and leaf blocks

```
python sweep.py
python sweep.py --sizes 1 2 3 4 5 8 9 16 --cutoffs 1 2 4
python sweep.py --sizes 4 16 64 --cutoffs 1 4 --algorithms alphaevolve
python sweep.py --sizes 1 2 3 4 5 8 9 16 --cutoffs 1 2 4 --reference-report experiment.json
```

The sweep executes each configuration, counts arithmetic with primitive clocks
disabled, verifies every output against the independent exact oracle, and writes
JSON plus HTML to one dated `outputs/` directory. These graphs measure work rather
than elapsed time. Larger sizes are supported and can require substantial resources.
Every algorithm and cutoff uses the same reproducible signed-integer inputs at a
given size. Repeating identical inputs is unnecessary for deterministic counts.

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

## Research work checklist

- [ ] Build our own sparse scheme evaluator, recursive composition engine, and operation accounting. The shared evaluator, traversal, and counters are implemented; general scheme import and configurable composition remain.
- [ ] Establish complete exact identity verification with reproducible certificates. AlphaEvolve's base identity is checked; a general certificate system remains.
- [ ] Reproduce a licensed baseline, then investigate the stronger aggregation constructions.
- [ ] Add custom CUDA execution and memory scheduling when needed to run larger mathematical constructions; hardware tuning is secondary.
- [ ] Search for product-count reductions or better constructions, independently verifying every candidate.

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
