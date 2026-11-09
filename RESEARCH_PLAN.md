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
Exact fringe handling is now the default; the historical padded baseline remains
explicitly selectable. At each recursive node, apply the recipe to the divisible
core and finish each remaining dot term exactly once. The complete count budget
includes fringe work at every depth. Fixed stage counts bound the remainder on
each axis, so square-node overhead is O(n²), preserving the justified recurrence
exponent and any existing borderline logarithmic factor. Core transitions may
still produce finite jumps; this does not promise a smooth cost at every size.
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

Maintain this plan as promising new angles arise, updating the README checklist
where relevant. For each angle, record its motivation, source or exact local
evidence, limitations, and a bounded next experiment with a success criterion.
Preserve existing paths and failed candidates; explain priority changes and keep
proposals distinct from verified constructions.

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

The first bounded interior experiment is implemented in `cw_interior.py`, using
the reusable `tensor_restrictions.py` and `recipe_parts.py` primitives. The command
`python -m matrix_multiplication.constructions cw-interior --certificate` tries
2,048 coordinate candidates for each of the three interior sectors and its
square, retaining every attempted map and outcome. For CW₂ squared, the base
sectors recognize 23, 23 and 22 candidates in this prefix. Two parts per base
sector are retained; their shapes are (2,1,2), (1,2,2) and (2,2,1), with four
products each after full exact polynomial recovery. Squaring a known witness
gives a 16-product part in each orientation. The fresh square-search prefixes
recognize no additional parts. These misses are bounded coordinate-search
evidence, not exclusions. Combining the three orientations from separate source
copies gives a completely verified 4-by-4 recipe with 64 products and exponent
three. This establishes reusable finite parts, not an improved bound.

The experiment is extraction from the retained interior components
of CW₂ squared. The current `coarse_catalog(2, 2)` reconstructs the full tensor
and retains the three sectors (1,1,2), (1,2,1) and (2,1,1), with permutations of
space dimensions (4,4,6). These dimensions obstruct treating a whole component
as a matrix product; they do not exclude useful restrictions or products from
its powers. See CONSTRUCTIONS.md for the dimension argument and exact evidence.

The experiment tries coordinate restrictions of each component, then its square
under explicit candidate and size budgets, checking all three orientations.
It saves source-coordinate maps, budgets, failures and complete coefficient
checks for each retained extraction. Recovery pays the restricted polynomial
recipe's finite cost before comparison with the existing fine/coarse extraction
and dense packing. Nontrivial exactly verified matrix restrictions now establish
the first milestone; an improvement still requires a lower verified
recovered cost for the same product, or a better verified dense recurrence after
packing. A restricted search miss excludes only the enumerated candidates.
This experiment does not yet implement general linear restrictions, recursive
laser extraction or the asymptotic sum conversion. It is prioritized because it
investigates exact components that the current matrix-only selector cannot use.

The bounded seeded follow-up is now implemented in `restriction_neighborhoods.py`
and `cw_neighborhood.py`, with the command
`python -m matrix_multiplication.constructions cw-neighborhood`. Each sector and
component power has a 4,096-attempt cap, split into 2,048 single-coordinate edits
and 2,048 pair tests; original prefix controls use their separate 2,048-attempt
budget. Unused budgets are not counted as work. The default experiment tests 32
edits and one pair per base sector, then 1,664 edits and six pairs per squared
sector: 5,088 edits and 21 pairs in total. It finds no new accepted edit, dense
union, independent sum, shape or product-count saving. Eighteen verified seed
recipes are recovered and executed at their full classical costs. Every source,
candidate map and rejection is retained in the generated report; a compact
selected record is in reports/2026-10-15-cw-neighborhood/.

Joint checking accepts a larger matrix only after complete recognition and an
independent source coefficient check. An independent direct sum additionally
requires disjoint spaces and a complete identity excluding all cross terms.
Accepted sums recover the joint polynomial restriction as a whole and compare
it with paid separate recovery of the same target. An overlapping dense union
uses its own classical target as the comparator: summing separate costs can
compare different computations. Positive and contaminated-source fixtures test
these paths even though the CW experiment has no accepted joint target.

The support angle follows from these exact misses and the move definition. A
single-coordinate replacement preserves all three space dimensions, so it
cannot change the inferred matrix shape. A verified witness may also require
coordinated input/output changes; the accepted-neighbor graph can be isolated
without excluding useful restrictions elsewhere. Try support-guided,
shape-changing restrictions on the pooled seed coordinates and the original
component spaces, starting with target (2,2,2) on squared sources under 4,096
candidate attempts. For chosen four-coordinate A and B subsets, a retained C
coordinate must contain exactly two supported products; use this necessary
condition to prune candidates, then verify every coefficient of the proposed
matrix layout. Enumerate mixed seed footprints as well as whole unions, retain
all attempted maps and reasons for pruning, and recover every admitted recipe
before claiming a saving. A new verified shape relative to the present controls
or a strictly smaller fully paid product count is success. General linear maps
and broader fields remain separate paths; this local search supplies no
exclusion beyond its explicitly tested candidates.

That follow-up has now been run as three sequential experiments, implemented in
`support_search.py` and `cw_support.py`. `cw-support` is a shared CLI/suite path.
Each input-pair screen and each output-layout proposal consumes one of 4,096
attempt units per target and sector; separate counters make degree pruning and
full recognition visible. Diagonal interleaving of lazy A/B subset streams avoids
spending a whole prefix on one A subset. All retained maps still pass complete
recognition and independent checks against the exact source coefficients.

1. **Seed-coordinate pool.** Search the squared interior sectors for (2,2,2)
   using only the old seed coordinate pools. All three 4,096-unit prefixes miss.
2. **Whole spaces and coordinated shape changes.** Broaden to the full squared
   components and targets (2,2,2), (2,2,3), (2,3,2), (3,2,2), each under its own
   4,096-unit cap. Eight completely verified (2,2,2) layouts appear in the
   squared (1,1,2) sector; two are retained. The nearby larger targets and other
   orientation prefixes miss. All four selected outputs of the first retained
   map lie outside its old seed output pool, explaining that pooled miss.
3. **Paid recovery, exact rank walks and recombination.** Both retained layouts
   recover from 432 interpolation products to eight executable products by exact
   coefficient expansion. One recipe enters the declared rank walk: seed 0
   misses after 10,000 flips; seed 1 reaches rank seven after 1,779 flips. Its
   transcript replays exactly and the complete identity passes independently.
   Tensoring that refined part with itself yields a certified 4-by-4 recipe
   with 49 products. This gives log₄(49) = log₂(7), about 2.8074, reproducing the
   known seven-product case; it does not improve the 2.7742999804 Smirnov baseline.

These are construction results relative to our earlier finite controls, not
claims of new mathematical bounds or a global public-code record. The selected
maps, executable parts, unsuccessful seed, successful replay and original run
evidence are retained under reports/2026-10-15-cw-support-three-step/. Full attempt
logs remain reproducible from maintained code in ignored dated output trees.

The cyclic follow-up in `cw_symmetry.py` checks every coefficient of the three
squared interior sectors under a leg rotation. The eight known layouts give
24 verified maps, including 16 transported ones. The earlier rank-seven
recipe also survives both rotations. The experiment checks 28 pairs within
the original sector and 64 between that sector and its next rotation.
These cross-sector pairs represent the other sector combinations up to
simultaneous rotation. Only the 92 tested pairs enter the counts. Their common
source is the restriction of CW₂ to the fourth power that keeps the union of
all 24 maps, so coefficients mixing the sectors remain visible to the checker.

Of the 92 pairs, 64 are independent sums, eight form a single (2,2,4) matrix
product, and 20 share coordinates without forming a recognized dense product.
Exact recovery yields 16 executable products for every accepted pair. The
independent sums tie separate polynomial recovery. When one or both maps have
a retained source-derived rank-seven recipe, executing those separate recipes
takes 15 or 14 products. Applying Strassen separately to any verified pair
also takes 14. The dense unions tie their own classical count. The run establishes
no exponent improvement. See
[the selected cyclic follow-up evidence](reports/2026-11-09-cw-symmetry/README.md)
for the exact maps, rejected pairs, and reloadable recipes.

The rectangular unions are useful controls. The exact rank of (2,2,4) is
already known to be 14, so a walk targeting 13 or 12 would search below a
proved minimum. The direct sum of two 2-by-2 products also has rank 14.
The experiment contains a pair whose two retained rank-seven recipes meet
that count. [Alekseev and Smirnov](https://www.mathnet.ru/eng/spm47),
[Shitov, Theorem 2](https://www.researchgate.net/publication/413758894_Determining_the_tensor_rank_of_2_2_m_matrix_multiplication_with_GPT-56_Sol),
and [Rupniewski](https://arxiv.org/abs/2209.11040).

The 14-product pair now runs recursively in `recursive_batching.py`. Each
node groups its 14 block products into seven calls and routes each returned
block back to its original term. Expanding two levels yields a 98-product
recipe for two 4-by-4 matrix products. Its full identity passes all 32,768
coefficient checks, including the zeros between the two outputs. Actual
signed rational runs at dimensions 2, 4 and 8 pass independent output
checks with 14, 98 and 686 products, matching both the recurrence and two
separate Strassen calls. The batched interpreter uses more additions and
coefficient scalings than the hand-written Strassen calls. The two outputs
in this control start from separate rank-seven recipes, so the count verifies
the recursion without giving a lower exponent.
[Run evidence](reports/2026-11-09-cw-batching/README.md).

Pan's exact two-product aggregation now runs through that scheduler. At
dimension 4, 64 aggregate terms mix entries from the two input pairs and 48
correction terms cancel 384 unwanted coefficients. The full base identity
and the expanded dimension-16 identity both pass, covering 32,768 and
134,217,728 coefficients by sparse comparison. Signed rational runs give
both exact outputs at each size. The observed data-product counts are 112
and 6,272, compared with 128 and 8,192 for separate classical calls and 98
and 4,802 for separate Strassen calls. The extra additions and scalings
also exceed both comparators. The experiment verifies a genuinely mixed
recipe and its finite recurrence. Its product exponent is log₄(56), above
the known Strassen and Smirnov exponents.
[Run evidence](reports/2026-11-09-pan-batching/README.md).
[Solomonik and Bläser, equations 1–3 and Appendix A](https://arxiv.org/html/2610.26563v2).

The next bounded experiment is to handle an incomplete final child group.
Split one term of the checked 14-product CW pair into two half-weight
terms to make a 15-term control with the same exact identity. With two
slots, the straightforward schedule uses one zero-filled slot at each
non-leaf call. Execute it at dimensions 2, 4 and 8, checking both outputs
and accounting for the scheduled 16, 128 and 1,024 scalar products.
Then build one tensor-square base with explicit lane and matrix coordinate
maps. It should have 225 terms for four independent 4-by-4 products.
Check the complete identity, then measure the four-slot schedule with
228 scalar products, including three dummy slots. Compare with 256
products from two separate runs of the padded two-slot schedule on the
same four products. This is a test of the finite rounding overhead in the
equal-size sum construction. The comparison still uses an artificial
control and separate Strassen needs only 196 products for the four outputs.
Larger-power extraction and the effective OpenAI witness audit remain open.

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
   Initial interior restrictions now yield reusable classical-cost parts and
   checked products of their witnesses. Next, seek cheaper or larger extractions
   using the finite batching work above, and investigate more effective sector selections,
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

## Comparing candidates for later implementation

The 15 October instrumentation adds the portable mathematical work schedule
described in MATHEMATICAL_WORK.md, including GCD, cross-cancellation, exact
division and carry/borrow accounting. Same-input classical-relative scores,
measured Python allocation peaks and logical value traffic have separate graphs
in the single automatic sweep. Original field counts and proved bounds retain
their meanings. No better exponent follows from a cheaper finite work score.

The next promising modelling angle is lifetime-aware payload storage and
cache-capacity-aware movement. Local evidence is that `common.py` explicitly
copies blocks and columns, while scalar-port traffic depends on each operand's
word width. Counts of cumulative copies and Python allocation peaks cannot tell
us which intermediates must coexist, or which reads can reuse fast memory.
These models should help shortlist candidates without optimizing the Python
executor or assigning memory an arbitrary weight inside arithmetic efficiency.

The next bounded experiment is to model classical, Strassen and Winograd
execution graphs at dimensions 2, 4 and 8, with leaf cutoffs 1 and 2. Track exact
payload lifetimes under a stated sequential schedule, then simulate two declared
fast-memory capacities. Success requires hand-checked one-cell and 2-by-2
storage/transfer budgets, unchanged exact products and arithmetic counters, and
separate total transfers versus peak live storage. Do not infer a hardware
bandwidth measurement, parallel storage bound, or infinite-size memory recurrence
from this finite exercise. This path augments the existing construction research;
it does not replace the lower-exponent witness and extraction searches above.
