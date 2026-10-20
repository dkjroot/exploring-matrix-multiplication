# Third-party sources and notices

Reviewed 20 October 2026 against the current source tree. The reference algorithms
and tests use Python's standard library; there are no third-party Python runtime
or test packages. Python itself retains its PSF license. No external matrix
multiplication implementation is bundled.

Mathematical credit is given in README.md's source bibliography and in the
implementation explanations. In particular, the Strassen/Winograd, Smirnov,
Coppersmith-Winograd, and later tensor constructions are not claimed as new
mathematics merely because their implementations are original here.

| Optional external material | Credit and terms | Distribution status |
| --- | --- | --- |
| Smirnov coefficient encoding from fast-matmul | Alexey V. Smirnov; encoding by Austin R. Benson and Grey Ballard; copyright Sandia Corporation; BSD-2-Clause | Downloaded separately into ignored research data. The importer preserves the full upstream notice in converted JSON. |
| AlphaEvolve rank-48 coefficient data | Google DeepMind; Apache-2.0 | Downloaded separately; not bundled. Retain the upstream license, applicable notices, attribution, and a description of coefficient conversion if redistributing. |
| AlphaTensor reference implementation/data | Google DeepMind; Apache-2.0 | Reading reference, not incorporated into the maintained implementation. |
| Other papers and reference repositories | Their respective authors and rights holders | See README bibliography for specific sources, licenses and reuse limits. No general permission to redistribute these references is claimed. |

Pinned Smirnov sources:
[coefficient table](https://github.com/arbenson/fast-matmul/blob/132358be7a6ab57c83dd5ce70ece460bd1ec5dfa/codegen/algorithms/smirnov336-40-960),
[full BSD notice](https://github.com/arbenson/fast-matmul/blob/132358be7a6ab57c83dd5ce70ece460bd1ec5dfa/LICENSE).
AlphaEvolve sources:
[results](https://github.com/google-deepmind/alphaevolve_results),
[license](https://github.com/google-deepmind/alphaevolve_results/blob/main/LICENSE).
AlphaTensor: [source and license](https://github.com/google-deepmind/alphatensor).

Keep third-party notices with any redistributed source/data or derived artifacts
that incorporate that material. The repository GPL does not replace BSD/Apache
notice requirements. Unresolved-license LITA downloads and reference-only Unity
source must not be promoted from local research storage into a public release
without appropriate permission. See [LICENSING.md](LICENSING.md).
