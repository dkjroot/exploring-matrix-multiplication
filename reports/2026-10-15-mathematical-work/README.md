# Mathematical work and logical movement accounting — 15 October 2026

The combined sweep now compares three separate quantities: classical-relative
mathematical work, Python-traced allocation peaks, and logical value movement.
The complete definitions and limitations are in
[MATHEMATICAL_WORK.md](../../MATHEMATICAL_WORK.md).

The work schedule accounts for operand widths, products, additions,
subtractions, carry/borrow handling, comparisons, shifts, signs, GCD,
normalization, exact division and cross-cancellation. Existing field counts
and the original unreduced word model retain their meanings. A higher work
efficiency score means less estimated arithmetic under the declared common
schedule; it is not a speedup prediction. Memory and movement remain separate.

This selection preserves the original measurements and manifest from
`outputs/2026-10-15_183546_099446/`. That small combined sweep passed all 29 stages,
including all five built-in algorithms, construction/search families, generated
recipes and certificates. All 20 growth configurations have complete exact-work
and logical-movement coverage. Classical multiplication scores exactly one.
The 49-product generated 4-by-4 recipe participates in the same comparison.

These small sizes do not establish which algorithm is most promising at larger
dimensions. A generic sparse recipe can retain coefficient scales that a
hand-written equivalent omits; scores describe those explicit schedules.
Logical movement is scalar-port and matrix/vector payload flow, not cache/DRAM
traffic. Python allocation peaks include observer replay and are not intrinsic
algorithm storage. None of these resource observations is projected to infinity.

`measurements.json` is the original byte-preserved sweep JSON, including separate
exact/floating diagnostic snapshots. `source-suite-manifest.json` records the
original complete run and its artifact fingerprints; this directory is a
measurement selection rather than a copy of that entire artifact tree.
`file-sha256.json` hashes the retained data and validation logs. JSON and logs
retain their bytes across Git checkout under the repository attributes.

The focused validation log records 28 passing cases; the bounded regression log
records 15 more passing resource-bound and fringe cases. A separate final renderer
check records 20 passing cases after the presentation changes. An exploratory
broader run was stopped during its expensive default projection cases; this
selection does not claim a complete repository regression run. Its earlier
classical-reference failure was fixed and passed the retained targeted rerun.

Reproduce the small complete suite from the repository root:

```sh
python sweep.py --sizes 2 4 --cutoffs 1 --cw-parameters 2 --cw-powers 2 --search-steps 50 --search-restarts 1 --pair-trace-limit 10 --projection-cycles 2
```

Use the ordinary `python sweep.py` command for the larger default combined
report. All existing families and the new comparisons remain enabled together.
Exact counters and recipes are reproducible; original elapsed times and
allocation observations cannot be recreated exactly by rerunning.

Re-render the preserved measurement page with today's maintained renderer:

```sh
python -c "from matrix_multiplication.reporting import write_report; print(write_report('reports/2026-10-15-mathematical-work/measurements.json'))"
```

The remaining resource-modelling path is lifetime-aware payload storage and
cache-capacity-aware transfers, with the bounded experiment in RESEARCH_PLAN.md.
