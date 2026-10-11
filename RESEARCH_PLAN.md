# Research target and public implementation audit

Reviewed 11 October 2026. The objective is an independently implemented,
instrumentable exact dense square matrix multiplier with the lowest verified
arithmetic exponent we can achieve and release under an open-source license.
The comparison is not limited to GPL projects. This repository currently uses
GPL version 3; no license change is made by this research plan.

## Research priorities

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
| [Smirnov in fast-matmul](https://github.com/arbenson/fast-matmul) | log_54(64000) = 2.7742999804 | Generated C++ rectangular and square-composition implementations | Clearly licensed BSD 2-Clause candidate; inspected, not built or benchmarked here |
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

The existing four reference algorithms now share documented scalar/block
primitives, a sparse bilinear evaluator, and recursive traversal. Opt-in
instrumentation counts arithmetic and structural work, aggregates primitive
profiles, and retains or streams scalar traces. See INSTRUMENTATION.md and the
five-step checklist in README.md. No new lower-exponent scheme is claimed by
this infrastructure refactor; general coefficient ingestion/certificates and
CPU/GPU backends remain research work.

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
