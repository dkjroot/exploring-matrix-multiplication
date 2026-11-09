# Cycling the balanced CW layouts

The earlier [support search](../2026-10-15-cw-support-three-step/README.md)
found eight ways to pick out a 2-by-2 matrix product from the square of one
CW₂ interior sector. The other two sectors are rotations of the first. We
checked that symmetry on every coefficient and used it to carry the eight maps
into both sectors. The replayed seven-product recipe also passed the complete
matrix identity after each rotation.

A pair can share coordinates and still form a larger matrix product. It can
also use disjoint coordinates and have unwanted cross terms. To tell these cases
apart, we tested the pairs in one common restriction of CW₂ to the fourth
power. That restriction keeps all coordinates used by the 24 maps together, so
it preserves every coefficient a pair could see. We tested the 28 pairs within
the original sector and the 64 pairs from that sector to its next rotation.
Rotating both members would reach the other sector combinations. The table
counts the pairs actually tested.

| Pair scope | Independent sums | Single matrix products | Rejected |
| --- | ---: | ---: | ---: |
| Within the original sector | 16 | 0 | 12 |
| Between the original and next sector | 48 | 8 | 8 |

Every accepted pair was recovered from the polynomial as one recipe and
executed on signed rational inputs. Every recipe used 16 data products.
Separate polynomial recovery uses eight products per member of an independent
sum. Among the retained source-derived recipes for each particular map, the
rank-seven reductions give separate costs of 15 products on 14 pairs and 14
products on one pair. Any verified pair can also be computed with 14 products
by applying Strassen separately to each member. The eight single
matrix products all multiply a 2-by-2 matrix by a 2-by-4 matrix. They tie their
own classical cost of 16 products. That shape has a known exact rank of 14,
so the unions are useful for checking recovery and coordinate maps. The
verified square exponent remains the Smirnov baseline.
[Alekseev and Smirnov](https://www.mathnet.ru/eng/spm47).

## Reproduce the experiment

Run from the repository root with Python 3.10 or later. This experiment uses
the standard library and the repository's own polynomial generator. It needs
no downloaded coefficients.

```sh
python -m matrix_multiplication.constructions cw-support --power 2 --certificate
python -m matrix_multiplication.certificate_checker reports/2026-11-09-cw-symmetry/certificate.json
```

- [evidence.json](evidence.json) keeps the common coordinate pool, all maps, the
  92 pair outcomes and their cost comparisons. It includes the run settings
  and hashes of the original output and changed construction files.
- [parts.json](parts.json) keeps all 102 executable parts and their original
  observations. The repository's `load_parts` reader verifies every matrix
  or independent-sum identity. Each accepted pair's entry retains its full
  finite recovery costs and its coordinates in the CW fourth power.
- [run.log](run.log) records the original command output and process timings.
  Those timings cover construction, verification, execution and report writing
  together. The per-recipe execution counters are separate.
- [certificate.json](certificate.json) and its [checker log](certificate-check.log)
  record the independent check of the earlier 49-product square recombination,
  exported again by the same run. The pair identities are checked separately
  through the parts library and source-restriction replay.

The command writes a dated directory with an HTML summary, the full support
prefix logs, and the replay transcript. The rank-walk transcripts match those
in the earlier support-search archive.
