# Research target and public implementation audit

Reviewed 11 October 2026. The objective is an independently implemented,
instrumentable exact dense square matrix multiplier with the lowest verified
arithmetic exponent we can achieve and release under an open-source license.
The comparison is not limited to GPL projects. This repository currently uses
GPL version 3; no license change is made by this research plan.

## Research priorities

The README mission and AGENTS.md define the reference guardrails: exact
arithmetic, our own readable primitive hierarchy, and comparable instrumentation
take priority over execution speed. Research searches may use broader GPU or
numerical proposal tooling, but accepted coefficients and identities require
independent exact reconstruction/verification and our own reference execution.
The exponent of one construction is an upper bound on omega, not omega itself.

Secondary measurements now include observed operand/denominator growth and a
declared 64-bit schoolbook word-cost model, separate growth-call Python allocation
peaks and setup stage resources. Exact category recurrences project counts beyond
executed sizes. Input-dependent bit-growth, GCD/division and algorithm peak-storage
recurrences remain tasks to derive; no resource projection is inferred from rank.

Finite-size reports now distinguish a recipe's equal-block divisions from our
portable zero-padding and cutoff policies. Per-sample shapes and stage paths
explain padded plateaus and geometry changes. Smooth curves and sparse connecting
lines are presentation models, not proof that every integer size has that cost.
Add exact optional fringe handling while preserving the current padded baseline;
derive its budgets and recurrence before admitting its costs to comparisons.
GCD, cancellation, division/normalization, and multiword carries/accumulation each
remain explicit cost-model tasks in the README checklist.

The primary result is an executable, verified construction approaching the best
mathematical arithmetic bounds. Large constants, slow execution, high memory use,
and historical difficulty are acceptable research costs. Readability, reusable
primitives, and independent verification make those constructions inspectable.
Hardware throughput is not the objective, and implementing only easy finite
schemes is not the intended endpoint.

Keep exponent and mathematical operation budgets primary. Retain elapsed time,
numerical error, optional Python allocation peaks, and future hardware resource
measurements as secondary diagnostics. None of them substitutes for proving the
construction's recurrence. Larger execution resources may help materialize a
construction, but cannot substitute for deriving or verifying it.

## Record definition

A claim must specify its arithmetic domain, exact base identity, recursive
construction, cost recurrence, and source revision. Operation counts and exponent
must be distinguished from measured runtime, GPU throughput, storage, precision,
and integer bit complexity. Results over Q/real/complex arithmetic are compared
separately from results restricted to fields such as F2 or approximations at a
nonzero approximation parameter.

Public coefficient data plus an evaluator is a different milestone from a tested
recursive implementation. Public code without a clear license is also different
from a redistributable open-source implementation. Search findings below are not
a proof of a global record: there is no exhaustive GitHub implementation registry.

## Strongest candidates identified in this audit

| Candidate | Square recursive exponent | Available material | Audit limit |
| --- | --- | --- | --- |
| [LITA](https://github.com/khoruzhii/lita/tree/472a79fbc8248423a97bebb3aacbc89e769b9b3e) | log_32(14196) = 2.758639371812189 | Rational coefficients, generator and a sample evaluator | Strongest explicit candidate found; complete rational identity not independently proved here, full recursive implementation not tested, no clear license located |
| [Smirnov in fast-matmul](https://github.com/arbenson/fast-matmul) | log_54(64000) = 2.7742999804 | Exact coefficient tables; our independent cyclic reference implementation | All three base identities certified; full 54x54 base executed with 64000 products and exact output; not a new record |
| [AlphaTensor](https://github.com/google-deepmind/alphatensor) | log_4(47) = 2.7772944258 | Published factorization and evaluation tools, Apache 2.0 | The 47-product result is over F2, not an ordinary rational/real/complex scheme |
| [AlphaEvolve](https://github.com/google-deepmind/alphaevolve_results) | log_4(48) = 2.7924812504 | Complex coefficients, Apache 2.0; our own evaluator is already implemented | Our base identity is completely checked; measured CPU/GPU performance work remains |

The LITA search snippet initially described an older 42-by-42, 31,598-product
scheme with exponent about 2.77201. The pinned repository revision now contains
the stronger 32-by-32 scheme. Search snippets must not substitute for reading
the candidate's current files.

## Independent LITA sample audit

Pinned commit: `472a79fbc8248423a97bebb3aacbc89e769b9b3e`, 7 October 2026.
Downloaded coefficient file SHA-256:
`758d06d61c4f52c3ee32f3811a9188c2ed86cc2b3b26ef078ccd2e96a0170094`.

Our own local evaluator loaded sparse rational coefficients, reduced them modulo
three primes, evaluated each input linear form, multiplied the two forms, and
accumulated output contributions. NumPy was used for safe data loading and
primitive array operations; no upstream program, NumPy matrix multiplication,
BLAS routine, or tensor contraction was executed. Integer overflow bounds were
checked before evaluation; the reference used Python integer classical products.

Four sample pairs (one identity case and three seeded random pairs) passed exactly
for each of 1,000,003; 1,000,033; and 1,000,037: twelve passes in total.
The three coefficient axes contain 8,742,427 nonzero entries. This large constant
matters for execution cost even though the advertised product count is 14,196.

These sample checks are screening evidence. They do not prove all rational tensor
coefficients correct, establish numerical stability, demonstrate recursion, or
establish a record. The report and independently written audit script are in the
ignored `Background Research/` directory. The unlicensed upstream data is not
bundled into this repository's release.

## Concrete improvement target

For a verified exact 32-by-32 scheme, reducing 14,196 products to 14,195 would give
log_32(14195) = 2.7586190457, strictly below this candidate's exponent. Another
base dimension or rectangular construction might offer a better target.
This is an objective, not a claim that such a reduction exists.

CUDA tuning, fewer additions, or better buffer schedules improve practical cost,
but do not lower log_u(r) when the base size u and product count r are unchanged.
Repeating a known scheme implements its exponent; publishing our own evaluator
of the same scheme does not create a new mathematical bound. A strict improvement
needs a better decomposition, a more effective extraction construction, or another
mathematically justified algorithmic change.

## Next work

The combined runner is now `python sweep.py`: all supported report families
share one dated output tree, generated schemes join the arithmetic comparison,
and an outcome/fingerprint manifest preserves completed work. Finite search/CW
budgets are separate from arithmetic sweep sizes. This orchestrates existing
capabilities; it does not itself lower the exponent. The original arithmetic-only
mode remains available with `--measurements-only`.

The existing five reference algorithms now share documented scalar/block
primitives, a sparse bilinear evaluator, and recursive traversal. Opt-in
instrumentation counts arithmetic and structural work, aggregates primitive
profiles, and retains or streams scalar traces. See INSTRUMENTATION.md and the
five-step checklist in README.md. No new lower-exponent scheme is claimed by
the initial infrastructure refactor. Coefficient ingestion and portable exact
certificates are now implemented, and the Smirnov extension below supplies a
lower-exponent executable baseline. CPU/GPU backends remain research work.

Prioritize the mathematical construction work below over kernel tuning. A backend
becomes useful when it allows us to execute and inspect a construction that our
reference interpreter cannot handle conveniently, even if it is slower than a
production classical multiplier.

1. Implement an independent sparse bilinear scheme format, exact identity
   verification with reproducible certificates, and scalar/compiled execution.
2. Reproduce a clearly licensed exact baseline such as Smirnov; investigate the
   stronger aggregation formulas and resolve reuse terms before distributing data.
3. Build recursive composition and operation accounting. Test multiple levels,
   rectangular inputs, and exact arithmetic before numerical/GPU performance.
4. Add our own CUDA leaf and linear-combination kernels, with bounded temporary
   storage, batching, transfers, and scalar operation counts observed separately.
5. Search for exact rank reductions or improved constructions, and independently
   certify any candidate. A larger rented GPU is an optional execution resource,
   not a prerequisite for declaring or proving the arithmetic exponent.
6. Refresh the public-code audit before making any priority or record claim;
   release code, certificates, attribution, license notices, and reproducible runs.

## Constructive route toward stronger exponents

### Implemented construction/search milestone

The initial construction layer is now executable. `tensors.py` provides exact
rational products/powers, direct sums, general linear restrictions, coordinate
selection, and matrix-specific row-major reindexing. `decomposition_search.py`
provides proportional-factor flips, exact sparse dependency reductions, optional
temporary rank-increasing splits, seeded bounded walks, and replay transcripts.
Generated matrix recipes pass complete identity verification and can enter the
existing instrumented composition/sweep and independent certificate paths.

The default discovery starts with eight classical 2-by-2 products and finds
seven without being supplied a seven-product formula. This demonstrates a
working generator, not a new exponent result. `degenerations.py` constructs the
published CW polynomial family, powers it with a shared parameter, and exactly
recovers its tensor coefficient with the full finite interpolation cost. Complete
small tensor checks pass. `cw_extraction.py` now selects sector words, rejects
shared groups and unwanted cross-sectors, and completely verifies extracted
independent matrix products. It restricts polynomial factors before recovery and
executes the recovered recipe with measured arithmetic. A single component with
supported dimensions can be exported as a matrix recipe. More effective
selection is still greedy. Direct coefficient expansion and replayed dependency
reductions now improve finite recovery. `dense_packing.py` verifies finite volume
tilings, compiles explicit input/output maps, replays new mapped dependencies,
and exports completely checked dense recipes. These small packings are cubic;
the stronger large-power laser/ASI conversion remains open.

Smirnov remains the verified 2.7742999804 baseline. Crossing 11/4 (2.75) needs
new verified finite rank/extraction evidence. The rational search is incomplete
and may miss paths or decompositions requiring a broader algebraic coefficient
field; it neither supplies the OpenAI finite size nor makes its existence proof
constructive. See [CONSTRUCTIONS.md](CONSTRUCTIONS.md) for executable commands,
source attribution, finite work accounting, and the remaining milestones.

### Remaining mathematical construction work

The next implementation priorities are now more concrete:

1. **Stronger CW extraction and large-power constructive conversion.** The first exact
   extraction milestone is implemented: explicit row-major restrictions, a
   deterministic greedy induced matching, complete literal/recovered direct-sum
   checks, and actual instrumented execution. For q=2, power=2, its four products
   cost 10 classical multiplications but 176 in the recovered recipe. Restriction
   lowers the verified polynomial degree bound from 20 to 11, reducing recovery
   from 320 products at the original bound to 176. Direct coefficient expansion
   now gives 42, reduced to 10 by replayed dependencies. Finite tile packing into
   a 2-by-2 dense base gives 10 mapped products, reduced to eight. The resulting
   cubic recipe is exported, measured, and certified by the combined suite.
   Sum-index coarsening now builds a complete merged-component catalog, verifies
   reconstruction, and recognizes matrix products with explicit basis permutations
   and full coefficient identities. The same suite recovers and packs a selected
   induced coarse sum. For CW₂ squared, twelve of fifteen components are matrix
   products; the three interior components have dimension obstructions. Their
   exact coefficients are retained. The coarse packing is still cubic.
   Next, extract products from powers/restrictions of those interior tensors
   rather than assigning them unsupported matrix shapes. Investigate more
   effective sector selections,
   larger powers, and recovery schedules that exploit their low rank relative
   to extracted volume. Implement a constructive large-power conversion rather
   than assign an asymptotic-sum bound to the current finite tiling. All finite
   overhead must remain in the executable recipe and recurrence. The 2-by-2
   packing is a correctness fixture, not a viable base for crossing 2.75.
2. **Targeted reductions of existing recipes.** Start from verified explicit
   schemes rather than always from classical products. Smirnov's small real
   (3,3,6) base is a useful target: cyclic rank 40 gives 2.7742999804, rank 39
   would give 2.7552591863, and rank 38 would give 2.7357237712. These figures
   are derived from log_54(r^3), not from discovered reductions. This links a
   finite coefficient-search task directly to crossing 2.75.
3. **Campaigns, symmetry, and resumable evidence.** Explore seeds, starting
   recipes, coefficient heights, and temporary rank ceilings; checkpoint RNG
   state and the current tensor so a stopped walk can resume exactly. Add
   basis/permutation transformations with explicit inverses and verified tensor
   preservation. Canonicalization may improve search coverage, but must not
   introduce unjustified reuse of floating execution observations.
4. **Candidate proposals plus exact reconstruction.** Numerical optimization
   can suggest coefficients; rational/algebraic reconstruction followed by a
   complete identity check is the acceptance boundary. Write inspectable proposal
   primitives and retain failed candidates. A small residual or modular samples
   cannot certify a real/complex matrix identity.
5. **Broader coefficient domains.** Extend search/construction from Q to Q(i),
   then number fields represented by exact polynomial relations if necessary.
   Verify reduction and interpolation arithmetic in each field. The OpenAI
   existence argument need not supply a rational or Gaussian-rational witness.
6. **Compact construction and streamed execution.** Replace materialized powers
   with lazy coordinate rules, sparse restrictions, and streamed combinations.
   This makes larger finite constructions inspectable without changing their
   product count or interpreting better memory behavior as a lower exponent.
7. **Certificates for the whole derivation.** Record independently checkable
   polynomial coefficients, lower-coefficient cancellations, interpolation
   weights, restriction maps, component independence, and packing witnesses.
   Final-base verification exists; complete construction-path certificates do not.
8. **Finite goal schedules.** Generate exact rank thresholds for chosen base
   sizes and exponent goals. The automatic suite already computes thresholds
   below 11/4 and 23/10 using integer inequalities; these are not existence
   claims. Add proved lower-bound exclusions and compare each verified candidate's
   finite padded arithmetic as well as its limiting recurrence.
9. **Effective OpenAI witness audit.** Map where the proof introduces a finite
   size or decomposition by existence, investigate whether bounds can be made
   explicit, and connect any constructive result to the generator/certificate
   interface. This is a separate research path from increasing a heuristic walk
   budget. Approaching 9/4 permits a chosen positive epsilon.

CW extraction/packing and explicit-base reductions have the most direct relation
to lowering the exponent; campaign infrastructure and broader domains support
those searches. GPU work becomes useful when these constructions demand it.
The mathematical background remains the credited CW/laser and flip-graph papers
listed in README.md; these priorities are our proposed implementation work, not
claims that those papers provide an immediately usable finite witness.

The square sparse interchange and repeating composition interpreter now exist in
`scheme_io.py` and `composition.py`. Both explorer and size sweeps execute plans
through shared primitives, retain coefficient provenance, and compare captured
arithmetic with per-level recurrence budgets. General finite identity checking is
implemented. Portable exact Python certificates now include a standalone,
independently written basis checker and the composition argument. General
rectangular constructions, formal proof-assistant verification, and compiled execution remain
separate work. The required rectangular block extension and Smirnov cyclic
baseline are now implemented: shape cycles (3,3,6), (3,6,3), (6,3,3) give a
dimension-54 rank-64000 square base. A full base execution was verified against
classical multiplication, including predicted/captured arithmetic budgets.
Coefficients are pinned local research data with source/license notices. This
lowers our implemented exponent to 2.7742999804; it reproduces a known bound.

1. **Executable lower-exponent baseline.** Extend the sparse scheme representation
   to rectangular dimensions, import clearly licensed Smirnov data, and compose
   cyclic shapes into a square recurrence. Certify the base identities and the
   composition, then exercise growing recursion depths. This advances our current
   exponent before attempting much larger asymptotic machinery.
2. **Constructive tensor machinery.** Represent tensor products/powers, index
   restrictions, coefficient transformations, and direct sums as observable
   building blocks. Preserve provenance and derive work bounds symbolically;
   materialize small instances for exact checks. A succinct recipe should expand
   into actual numerical computation, rather than just restate an exponent bound.
3. **Older asymptotic constructions.** Choose a specific published construction,
   make its finite parameters and extraction steps explicit, and implement them.
   Track the overhead of combining or recovering independent products. If a
   construction uses approximate identities, implement and account for exact
   coefficient recovery or another justified conversion; choosing a tiny numeric
   approximation parameter is not an exact algorithm. A finite tensor-power
   experiment must not be labelled with a limiting exponent it has not attained.
4. **The 2.25 witness.** Work on obtaining the finite base size, product count, and
   coefficient decomposition (or an executable generator) justified by the 2026
   existence argument. For a chosen positive epsilon, the target is a certified
   construction below 9/4 + epsilon, not an unsupported claim of a finite rank
   scheme with exponent exactly 9/4. Implement any required coefficient domain
   and count its costs consistently. The absence of an explicit witness in the
   paper defines an open constructive task; it is not a prediction that future
   research cannot solve it.
5. **Verified improvement.** Search over alternative finite schemes or extraction
   constructions, retain failed candidates and certificates as research evidence,
   and claim a stronger exponent only after exact verification and cost analysis.

These are staged research tasks, not promises that converting every theoretical
proof into a useful finite witness is routine. Progress can consist of reusable
construction primitives and verified instances even before a new bound is reached.

The RTX 3090 and reported 128 GB host RAM support useful initial experiments.
No cloud resources were rented or provisioned during this audit.

All upstream references and license links are also recorded in README.md.
