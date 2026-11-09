# Recursively batching the CW pair

The [cyclic support experiment](../2026-11-09-cw-symmetry/README.md) retained
two seven-product recipes whose source maps form an independent pair. Here we
join their recipes in that same order and use the result to compute two
larger square products together.

Think of each recursive call as taking two pairs of matrices. At size one
it multiplies the two scalar pairs. At larger sizes, the base recipe asks for
14 block products. Each of seven child calls handles two of them. For example,
child call 3 handles parent terms 6 and 7, with one term from each parent
output. After the call, each returned block is used in the output combination
for the term that requested it. The saved schedule numbers terms and child
calls from zero.

Expanding this schedule through two levels gives a 98-product scalar recipe
for two independent 4-by-4 products. The complete check covers 32,768
coefficients, including every coefficient that would mix the two pairs. Both
outputs also match independent exact dot products on signed rational inputs
at each size below.

| Matrix dimension | Predicted products | Observed products | Two separate Strassen calls | Two classical calls | Batched additions | Batched coefficient scalings |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 14 | 14 | 14 | 16 | 80 | 80 |
| 4 | 98 | 98 | 98 | 128 | 880 | 880 |
| 8 | 686 | 686 | 686 | 1,024 | 7,440 | 7,440 |

The schedule counts each sparse scale and addition, including scaling by one
and adding into an initial zero. The original observations include operand
widths, mathematical work and logical value movement. The elapsed times are
for instrumented calls with input validation. Setup, the complete identity
check and the independent output oracle sit outside those calls. The data
product counts match separate Strassen execution and give the same exponent.

## Reproduce the run

Run this command from the repository root with Python 3.10 or later. The
checked-in CW parts library supplies the input recipes.

```sh
python -m matrix_multiplication.constructions cw-batch
```

The combined sweep also runs this fixed control, using the archived input
independently of the current search budget.

- [HTML report](index.html) shows the routing and comparison.
- [report.json](report.json) records the original execution evidence and the
  source coordinate maps.
- [parts.json](parts.json) contains the 14-product base and the 98-product
  expansion. The repository's `load_parts` reader rechecks both identities.
- [run.log](run.log) and [file-sha256.json](file-sha256.json) record the original
  output path and identify the retained artifacts and source files.

The recursive construction follows the divisible case of the equal-size
batching argument. [Solomonik and Bläser, Appendix A](https://arxiv.org/html/2610.26563v2#A1).
