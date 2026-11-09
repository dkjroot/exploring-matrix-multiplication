# Pan's exact aggregation through two recursive levels

Pan's formula computes two independent products, AB and UV, through shared
intermediate products. In the aggregate, each left linear form mixes an entry
from A with one from U, and each right form mixes entries from B and V. The
product contributes to both outputs. Expanding it gives two wanted monomials
and six mixed monomials. Three correction families cancel the mixed terms,
with one family for each index that can be summed inside a linear form.

At dimension 4, the recipe spends 64 products on the aggregate and 16 on
each correction family. The correction tensor cancels all 384 excess
coefficients exactly. It uses negative output coefficients, so the
instrumented interpreter records these as scalings and additions. We checked
all 32,768 coefficients of the base against the two independent dot-product
identities.

To go up a level, each of 56 child calls computes two of the 112 requested
block products. The child results return to the original terms, including the
correction terms. The expanded size-16 recipe has 6,272 terms. Its coefficient
check covers all 134,217,728 entries, of which 8,192 are nonzero. We compared
exact sparse coefficient dictionaries, which also checks every zero without
allocating a dense cube.

| Dimension of each product | Pan data products | Separate classical | Separate Strassen | Pan additions | Pan coefficient scalings |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 112 | 128 | 98 | 864 | 864 |
| 16 | 6,272 | 8,192 | 4,802 | 62,208 | 62,208 |

Every count is from an actual instrumented call. Both outputs passed
independent exact rational dot-product checks, and the three methods used
the same signed inputs. Pan's schedule saves scalar data products against
classical multiplication and spends more than Strassen at both sizes.
Its additions and coefficient scalings also exceed the separate calls.
The product recurrence for this fixed base is 56 child calls of one quarter
the dimension, with two scalar products at size one. It gives the exponent
log₄(56) for this schedule.

## Reproduce the run

Use Python 3.10 or later from the repository root. The generator needs no
downloaded inputs or external implementation.

```sh
python -m matrix_multiplication.constructions pan-batch
```

Use `--power 1` to run the base case alone. The automatic sweep includes the
base case and runs the second level when its size list contains 16.

- [HTML report](index.html) explains the cancellation and shows the comparison.
- [report.json](report.json) records the exact checks, term-to-slot maps, and
  original measurements. The elapsed times cover the instrumented public
  calls with input validation. Construction and independent verification
  happen outside those calls.
- [parts.json](parts.json) keeps the base and second-level recipes in compact
  form. The repository's `load_parts` reader rechecks both identities.
- [run.log](run.log) and [file-sha256.json](file-sha256.json) identify the run,
  source files and retained artifacts.

The generator follows the published trilinear equations with the repository's
own row-major indexing, tensor primitives and scheduler.
[Pan](https://doi.org/10.1137/0209027) ·
[Solomonik and Bläser, equations 1–3](https://arxiv.org/html/2610.26563v2#S1).
