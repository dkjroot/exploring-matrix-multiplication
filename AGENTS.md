# Project rules

Read the mission and research rules at the beginning of README.md before making
algorithm, backend, measurement or reporting changes. This is an educational
exact-arithmetic reference and a constructive mathematics research project.

- Preserve mathematical fidelity. Authoritative constructions, coefficients,
  identities and execution evidence use arbitrary-precision integers, exact
  fractions or supported exact algebraic representations. Do not replace them
  with floats, approximate identities or finite-width arithmetic. Floating-point
  accuracy diagnostics, display coordinates and elapsed-time measurements are
  separate, explicitly labelled secondary evidence.
- Write our own multiplication algorithms and their hierarchy of reusable
  primitives. Do not delegate matrix multiplication, tensor contraction, or
  hidden algorithmic optimizations to NumPy, BLAS, cuBLAS or another implementation.
  Ordinary language scalar arithmetic is permitted. Credit mathematical sources
  and comply with source/data licenses; ignored downloads do not waive licenses.
- A GPU backend may batch explicit atomic arithmetic whose mathematical behavior
  and operation schedule we control. It must preserve exactness, operand-size
  accounting, traceability and comparable counters. Do not use a floating CUDA
  kernel as an exact reference or hide matrix operations in a vendor routine.
- Research/proposal searches may use third-party GPU or numerical tooling more
  broadly, provided proposals and approximate diagnostics remain separate from
  accepted results. Reconstruct exact coefficients and verify the complete exact
  identity independently before admission. A numerical residual, modular sample
  or GPU heuristic cannot certify a recipe or an exclusion. Reference execution
  remains our own transparent primitive hierarchy. Document research dependencies,
  reproducibility, accuracy limits, provenance and licensing.
- Favor human understanding and instrumentation over runtime efficiency. Every
  module, class and function needs documentation. Explain index conventions,
  cancellations, mathematical role and the reason for non-obvious steps. Share
  mathematical primitives without adding products to the algorithm's recurrence.
- Distinguish a verified algorithm exponent from omega, the infimum across all
  matrix algorithms over a stated arithmetic domain. Keep symbolic recurrence
  evidence, measured counts, finite-range fits and projections separate.
- Measure additions, subtractions, data/coefficient products, normalizations,
  operand growth and memory with explicit scope. A 64-bit word-cost model is an
  estimate derived from exact operand sizes, not CPU instructions or FLOP/s.
  Declare excluded reduction/GCD/carry work; never infer bit cost from rank alone.
- Keep setup/construction, algorithm execution, observer overhead and independent
  verification distinct. Unknown costs stay unknown. Do not project measured
  memory peaks or operand sizes to infinity without a justified model.
- Keep all report families in the automatic sweep, preserve earlier evidence and
  record failed candidates. Generate dated ignored outputs; manually published
  reports belong in reports/. Run small regressions before larger experiments.
- Claim a stronger bound only from an exact executable identity, justified
  recursive construction, accounted finite overhead and independent verification.
  A heuristic search miss is not an impossibility proof; a paper's limiting bound
  is not the exponent of an incomplete finite construction.
