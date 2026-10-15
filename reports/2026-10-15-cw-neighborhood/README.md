# Seeded CW neighborhoods: a bounded negative result

On 15 October 2026 the seeded-coordinate follow-up found no new verified matrix
shape and no recovered-product saving. This is a finite search result, not an
impossibility proof or a stronger exponent. The Smirnov baseline is unchanged.

## What was tested

The three unrecognized interior sectors of CW₂ squared were tried individually
and squared. Two verified base witnesses came from the original 2,048-candidate
prefix. Their four ordered tensor products seeded each squared source. The
prefix search on the squared source remained a separate control. All exact
source tensors and polynomial families are constructed by maintained code;
external downloaded coefficient data is not needed for this experiment.

Each sector/power was allocated 4,096 follow-up attempts: 2,048 single-coordinate
edits and 2,048 pairs. The accepted-map pool was capped at 64. Only accepted maps
would expand the breadth-first frontier. The actual work was:

| Source scope | Number of scopes | Edits per scope | Pairs per scope | New accepted targets |
| --- | ---: | ---: | ---: | ---: |
| Original interior sector | 3 | 32 | 1 | 0 |
| Squared interior sector | 3 | 1,664 | 6 | 0 |
| Total | 6 | 5,088 | 21 | 0 |

All 5,088 edited restrictions failed the necessary coefficient-support count
for their proposed matrix shape. All 21 seed pairs shared source coordinates
and had no recognized dense matrix union. They were not treated as independent
sums. No new edit was accepted, so each retained frontier and pair pool was
exhausted before its attempt cap. That covers only the stated moves and seeds.
It does not cover paths through rejected maps, coordinated changes, other matrix
shapes, general linear transformations or larger powers.

The 18 seed/control recipes were independently recovered from their restricted
CW polynomial families. Base shapes (2,1,2), (1,2,2) and (2,2,1) each cost four
data products. Their squared shapes cost 16. These are full finite recovered
costs, equal to classical cost, not border-rank annotations. Actual signed
rational execution verifies outputs and retains arithmetic/operand counters.

## Reusable additions and mathematical boundaries

`restriction_neighborhoods.py` provides coordinate-neighbor generation,
subset deduplication, bounded graph traversal, complete joint checks and the
`recover_joint` same-target comparator. `recover_tensor_restriction` in
`tensor_restrictions.py` recovers a whole checked target, including direct sums.

A dense union receives a matrix label only after a full identity check. An
independent sum needs disjoint spaces plus a full coefficient comparison that
excludes unwanted cross terms. Its joint polynomial recovery is compared with
paid separate recovery of that same sum. A dense union uses its own classical
target; overlapping separate parts may compute a different bilinear map, so
their summed costs are not credited as a saving. Positive fixtures exercise
both successful joint paths. An adversarial fixture preserves each separate
part but inserts an exact 1/3 cross coefficient; joint checking rejects it.

The library format also supports independently checked direct sums through
`component_shapes`. This particular experiment retains only ordinary matrix
seed parts because it accepts no joint targets.

## Retained evidence and reproduction

- `evidence.json` retains settings, counters, original fingerprints, source-map
  pools, representative exact rejections and the original combined-suite setup
  observation. The raw full search log is reproducible and is not duplicated
  here; its original SHA-256 is retained.
- `parts.json` is the original reloadable 18-part seed/control library, copied
  byte for byte from the final focused construction run.
- `regression.log` retains the passed 70-case targeted construction/reporting
  run. `focused.log` retains the final eight neighborhood cases, including
  positive same-target comparison and forged-parent rejection checks.
- `file-sha256.json` binds these selected files.

The separate small combined sweep passed all 23 stages, including the new
family, prior construction families, certificates, diagnostic pairs and growth
comparison. Its original stage timings, Python allocation peak and source
fingerprints are preserved in `evidence.json`; those historical observations
are not claimed as measurements of later source files. Documentation and diff
checks also passed. A full rerun of the much slower unrelated regression cases
was not required for this bounded change.

From the repository root:

```
python -m matrix_multiplication.constructions cw-neighborhood --seed-budget 2048 --candidate-budget 4096 --max-witnesses 64
python sweep.py --sizes 2 4 --cutoffs 1 --algorithms naive strassen --cw-parameters 2 --cw-powers 2 --search-steps 50 --search-restarts 1 --pair-trace-limit 10 --projection-cycles 2
python -m unittest tests.test_cw_neighborhood tests.test_cw_interior tests.test_cw_coarsening tests.test_cw_extraction tests.test_constructions tests.test_dense_packing tests.test_research_suite -v
```

On Windows, UTF-8 mode (`python -X utf8`) avoids platform-default decoding of
UTF-8 HTML in older tests. If the sandbox disallows the system temporary folder,
point TEMP and TMP at a writable ignored workspace directory first. No test
requires a generated source file from `outputs/`.

Load the retained parts without the output directory:

```python
from matrix_multiplication.recipe_parts import load_parts
parts, provenance = load_parts("reports/2026-10-15-cw-neighborhood/parts.json")
```

The raw construction was
`outputs/construction_2026-10-15_165006_355089/`; the distinct combined suite was
`outputs/2026-10-15_164517_421537/`. Regeneration reproduces exact recipes and
search outcomes with the same implementation and parameters, not timestamps,
original resource observations or hashes of a different source revision.

## Next bounded angle

Single-coordinate replacements preserve the three retained space dimensions
and therefore cannot change the inferred matrix shape. The all-rejected
frontiers motivate support-guided coordinated changes, rather than merely a
larger budget for this same isolated graph. RESEARCH_PLAN.md now proposes a
4,096-candidate search for (2,2,2) restrictions on squared sources, mixing seed
footprints and pruning by necessary exact support degrees before full identity
checking and paid recovery. This is a proposed next experiment, not evidence
that a better restriction exists.
