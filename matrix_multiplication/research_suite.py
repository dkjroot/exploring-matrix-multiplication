"""Run every supported report family as one bounded, reproducible research suite.

The suite orchestrates existing construction, measurement, rendering, and proof
primitives. It preserves the full arithmetic report and links its other evidence
from the same page. Search misses are legitimate outcomes; execution/identity
failures remain failures. A checkpoint manifest preserves completed artifacts
if a later stage fails. No exponent claim follows from merely completing a run.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import sleep
from traceback import format_exc

from . import ALGORITHMS, Instrumentation
from .certificates import write_bundle
from .composition import Composition
from .constructions import ConstructionRequest, run_experiment, target_rank, write_json
from .experiments import ExecutionMeasurement, arithmetic_model
from .reporting import STYLE, attach_reference_reports, read_reports, render_html
from .rectangular import RectangularCycle, cyclic_rotation
from .scheme_io import load_scheme_document, scheme_from_document
from .tensors import classical, to_scheme
from .verification import check_product
from .work_sweep import collect_work, positive_integer_grid


@dataclass(frozen=True)
class SuiteSettings:
    """Finite defaults for each research report family, independent of sweep sizes.

    Increasing measurement sizes does not silently enlarge a discovery search
    or tensor power. Every parameter is recorded in the manifest. Trace prefixes
    are bounded; exact identities and measured arithmetic are never sampled away.
    """

    search_size: int = 2
    search_start: str = None
    search_target_rank: int = None
    search_target_exponent: Fraction = None
    search_steps: int = 1000
    search_restarts: int = 10
    search_seeds: tuple = (0,)
    search_max_height: int = 1
    search_extra_rank: int = 0
    search_split_every: int = 50
    cw_parameters: tuple = (1, 2)
    cw_powers: tuple = (1, 2)
    cw_packing_size: int = 2
    trace_limit: int = 200
    measure_sweep_memory: bool = True
    projection_cycles: int = 12

    def validate(self):
        """Reject invalid ranges before creating output or doing expensive work."""
        positive = (self.search_size, self.search_restarts, self.search_max_height, self.search_split_every,
                    self.cw_packing_size, self.projection_cycles, *self.cw_parameters, *self.cw_powers)
        if any(type(n) is not int or n < 1 for n in positive) or self.search_size < 2:
            raise ValueError("suite sizes/parameters/restarts/height must be positive; search size >= 2")
        if self.cw_packing_size < 2:
            raise ValueError("CW packing size must be at least two")
        if any(type(n) is not int or n < 0 for n in (self.search_steps, self.search_extra_rank, self.trace_limit)):
            raise ValueError("suite steps/extra rank/trace limit must be nonnegative integers")
        if not self.search_seeds or any(type(seed) is not int for seed in self.search_seeds):
            raise ValueError("choose at least one integer search seed")
        if self.search_start is not None and (not isinstance(self.search_start, str) or not self.search_start.strip()):
            raise ValueError("search starting recipe must be a nonempty path or built-in formula name")
        if not self.cw_parameters or not self.cw_powers:
            raise ValueError("choose at least one CW parameter and power")
        if self.search_target_rank is not None and (type(self.search_target_rank) is not int or self.search_target_rank < 1):
            raise ValueError("search rank target must be positive")
        if self.search_target_exponent is not None:
            if type(self.search_target_exponent) not in (int, Fraction) or self.search_target_exponent <= 0:
                raise ValueError("search exponent target must be an exact positive rational")
            if self.search_target_rank is not None:
                raise ValueError("choose a search rank or exponent target")

    def document(self):
        """Convert settings to strict JSON while retaining an exact exponent target."""
        document = asdict(self)
        exponent = document["search_target_exponent"]
        document["search_target_exponent"] = str(exponent) if exponent is not None else None
        return document


def write_report(directory, data, title, source_name):
    """Write a report at a fixed suite location using the existing full renderer."""
    directory.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(data, indent=2, allow_nan=False)+"\n").encode("utf-8")
    (directory / "measurements.json").write_bytes(raw)
    rendered = render_html(read_reports(raw), title=title, source_name=source_name,
                           source_hash=sha256(raw).hexdigest(), generated=datetime.now().isoformat(timespec="seconds"))
    (directory / "index.html").write_text(rendered, encoding="utf-8")


def pair_evidence(implementations, plans, *, floating=False, trace_limit=200):
    """Observe one small pair per algorithm, with profiles, memory, and bounded traces.

    Integer and float cases have different arithmetic/accuracy scopes; neither
    supplies wall-clock or memory observations to the separate size-sweep calls.
    Verification happens after the call and outside measurement. Partial errors
    retain their counters. Native complex atomic operations remain opaque.
    """
    a, b = ([[0.1, -0.3], [0.7, 1.1]], [[0.2, 0.9], [-0.4, 0.6]]) if floating else (
        [[1, -2], [3, 4]], [[-5, 6], [7, 8]])
    reports, events = {}, {}
    for name, implementation in implementations.items():
        observer = Instrumentation(profile=True, trace=True, trace_limit=trace_limit)
        execution = ExecutionMeasurement(measure_memory=True)
        verification = None
        status, error = "error", None
        try:
            with execution:
                result = implementation(a, b, cutoff=1, instrumentation=observer)
            verification = check_product(a, b, result)
            status = "passed" if verification.passed else "failed"
        except (ArithmeticError, ValueError, TypeError, OSError) as exc:
            error = str(exc)
        report = {**observer.snapshot(), "status": status,
                  "execution_measurements": execution.snapshot(8, verification),
                  "experiment": {"algorithm": name, "cutoff": 1, "input_shapes": [[2, 2], [2, 2]],
                                 "scalar_domain": "floating point" if floating else "exact signed integers",
                                 "classical_data_product_baseline": 8,
                                 "input_repr_sha256": sha256(repr((a, b)).encode("utf-8")).hexdigest(),
                                 "primitive_timing_enabled": True,
                                 "scope": "separate small diagnostic pair in this suite"}}
        if error is not None:
            report["error"] = error
        plan = plans.get(name)
        report["mathematical_model"] = plan.model() if plan else arithmetic_model(name)
        if plan is not None:
            budget = plan.budget(2, 2, 2, cutoff=1)
            report["composition_budget"] = budget
            matches = {"field_multiplications": observer.counts["field_multiplications"] == budget["field_multiplications"],
                       **{key: report[key] == budget[key] for key in
                          ("field_arithmetic_operations", "atomic_arithmetic_operations")}}
            report["composition_budget_matches"] = matches
            if status == "passed" and not all(matches.values()):
                report.update(status="failed", error="observed pair arithmetic differs from verified plan budget")
        reports[name], events[name] = report, observer.events
    return {"reports": reports, "events": events}


def target_evidence():
    """Produce exact finite rank thresholds; a threshold is not a feasible witness."""
    goals = (Fraction(11, 4), Fraction(23, 10))
    rows = [{"dimension": d, **{str(goal): target_rank(d, goal) for goal in goals}}
            for d in (2, 3, 4, 8, 16, 32, 54)]
    return {"schema": "finite-rank-targets-v1", "strict_exponent_targets": list(map(str, goals)),
            "rows": rows,
            "meaning": "Largest integer rank strictly beating each square exponent; no existence claim.",
            "known_limit": "2x2 requires at least seven products; its threshold six for 11/4 is infeasible.",
            "source": "https://arxiv.org/abs/2212.01175"}


def generated_plan(path):
    """Convert a generated base into an executable verified square-target cycle.

    A real rectangular identity can be rotated through its trace symmetry. Three
    orientations contract all axes by m*k*n and use rank^3 products, giving a
    balanced square recurrence. This makes small rectangular-base searches useful
    to the square-exponent objective without inventing a square rank for one stage.
    """
    document = load_scheme_document(path)
    scheme = scheme_from_document(document)
    if scheme.dimensions is None:
        return Composition.from_spec(str(path))
    provenance = document.get("provenance", {})
    if len(set(scheme.shape)) == 1:
        return RectangularCycle((scheme,), provenance=provenance)
    second = cyclic_rotation(scheme)
    return RectangularCycle((scheme, second, cyclic_rotation(second)), provenance=provenance)


class SuiteRun:
    """Checkpoint completed evidence and failures without discarding earlier stages."""

    def __init__(self, directory, configuration, progress=None):
        """Create a fresh run directory and record its requested report families."""
        self.directory = directory
        self.progress = progress
        self.manifest = {"schema": "matrix-multiplication-research-suite-v1", "status": "running",
                         "configuration": configuration, "stages": [],
                         "scope": "All implemented report families with recorded bounded experiments; not every possible parameter combination.",
                         "exponent_claim": "No new exponent follows from completing the suite."}
        directory.mkdir(parents=True, exist_ok=False)
        self.checkpoint()

    def checkpoint(self):
        """Atomically replace the manifest, briefly retrying temporary access conflicts.

        An indexer, backup reader, or file scanner may momentarily hold the old
        destination open. Keep the completed temporary JSON intact and retry the
        same atomic replacement; never delete the previous checkpoint first.
        Four attempts wait at most 70 ms in total. Persistent permission errors
        still propagate, because pretending to save a checkpoint would be false.
        This is report-storage work, outside every multiplication measurement.
        """
        temporary = self.directory / "suite-manifest.pending.json"
        write_json(temporary, self.manifest)
        for attempt in range(4):
            try:
                temporary.replace(self.directory / "suite-manifest.json")
                return
            except PermissionError:
                if attempt == 3:
                    raise
                sleep(0.01 * 2**attempt)

    def perform(self, name, family, operation):
        """Run an independent stage; retain failures and continue other families.

        This is the suite's fault boundary, not an algebraic validity shortcut.
        Any ordinary exception, including a programming error in a new backend,
        marks the stage as an error and records its traceback for investigation.
        KeyboardInterrupt and SystemExit inherit BaseException, so user-requested
        cancellation still propagates; the last completed checkpoint remains.
        """
        if self.progress:
            self.progress(f"Suite: {name}")
        stage = {"name": name, "family": family}
        # Setup stages have their own resource scope, separate from algorithm
        # calls and independent verification inside pair/growth measurements.
        # Never nest memory probes around those calls: an existing tracer cannot
        # provide a fresh per-call peak. These setup peaks include file/report
        # creation and observer overhead, and do not pretend to count every raw
        # Fraction operation in a construction/search implementation.
        setup = family not in ("profiles, memory, accuracy, traces",
                               "measured arithmetic and theoretical comparison", "report rendering")
        measurement = ExecutionMeasurement(measure_memory=setup)
        try:
            with measurement:
                stage.update(operation())
            stage.setdefault("status", "passed")
        except Exception as exc:
            stage["status"] = "error"
            stage["error"] = str(exc)
            stage["error_type"] = type(exc).__name__
            stage["traceback"] = format_exc()
            if self.progress:
                self.progress(f"Suite stage failed: {name}: {exc}")
        if setup:
            stage["setup_measurements"] = {"elapsed_seconds": measurement.elapsed_seconds,
                "python_traced_peak_bytes": {"status": measurement.memory_status, "value": measurement.python_peak_bytes},
                "scope": "suite stage including generation, exact checking and artifact writing; cached inputs may pre-exist",
                "coefficient_arithmetic_operations": {"status": "not_counted", "value": None}}
        self.manifest["stages"].append(stage)
        self.checkpoint()
        return stage

    def register_candidate(self, name, recipe_path, plans):
        """Check a generated recipe and admit it to later execution only on success.

        Finding an exact base and building a supported recursive plan are distinct
        steps. For example, rational search may succeed while a future coefficient
        field lacks an execution backend. Perform this conversion inside its own
        checkpointed stage; failure must not abort unrelated evidence families.
        Mutation of the shared plan dictionary happens last, after verification.
        """
        plan = generated_plan(recipe_path)
        model = plan.model()
        plans[name] = plan
        return {"scope": f"Executable repeating plan for {name}",
                "base_dimension": model["base_dimension"],
                "base_bilinear_rank": model["base_bilinear_rank"]}

    def construction(self, name, request):
        """Use the same generator API as the individual construction CLI."""
        folder, report = run_experiment(request, self.directory / name)
        result = {"status": "passed", "report": f"{name}/index.html", "data": f"{name}/report.json",
                  "outcome": report["outcome"]}
        if "search" in report:
            result.update(rank=report["rank"], target_reached=report["search"]["target_reached"])
        if "rank" in report:
            result["rank"] = report["rank"]
        if (folder / "scheme.json").exists():
            result["scheme"] = f"{name}/scheme.json"
        return result

    def pair(self, name, implementations, plans, floating, trace_limit):
        """Write full secondary diagnostics and their original bounded trace envelope."""
        data = pair_evidence(implementations, plans, floating=floating, trace_limit=trace_limit)
        write_report(self.directory / name, data, "Floating-point diagnostic pair" if floating else "Exact diagnostic pair", name)
        return {"status": "passed" if all(r["status"] == "passed" for r in data["reports"].values()) else "failed",
                "report": f"{name}/index.html", "data": f"{name}/measurements.json",
                "algorithm_calls": len(data["reports"]), "trace_limit_per_algorithm": trace_limit}

    def certificate(self, name, plan, scope):
        """Run both exact verifiers and save one portable package per measured plan."""
        folder = Path("certificates") / name
        write_bundle(plan, self.directory / folder)
        return {"status": "passed", "data": (folder / "certificate.json").as_posix(),
                "argument": (folder / "README.md").as_posix(), "scope": scope}

    def finalize(self):
        """Bind all completed artifacts to fingerprints and report overall health."""
        self.manifest["status"] = "passed" if all(s["status"] == "passed" for s in self.manifest["stages"]) else "failed"
        # The manifest deliberately excludes its own digest and the root HTML,
        # which embeds the manifest; including either would create a hash cycle.
        self.manifest["artifact_sha256"] = {
            path.relative_to(self.directory).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.directory.rglob("*")) if path.is_file()
            and path.name not in ("suite-manifest.json", "suite-manifest.pending.json")
            and path != self.directory / "index.html"}
        self.checkpoint()


def suite_panel(manifest, *, growth_available=True):
    """Render accessible suite navigation; keep every existing arithmetic section."""
    rows = []
    setup_rows = []
    for stage in manifest["stages"]:
        links = ' · '.join(f'<a href="{escape(stage[key], quote=True)}">{label}</a>'
                           for key, label in (("report", "HTML"), ("data", "JSON"), ("scheme", "Recipe"), ("argument", "Argument"))
                           if key in stage)
        note = stage.get("error", stage.get("outcome", stage.get("scope", "")))
        rows.append(f'<tr><td>{escape(stage["name"])}</td><td>{escape(stage["family"])}</td>'
                    f'<td>{escape(stage["status"])}</td><td>{links}</td><td>{escape(note)}</td></tr>')
        if 'setup_measurements' in stage:
            measurement = stage['setup_measurements']
            peak = measurement['python_traced_peak_bytes']
            setup_rows.append(f'<tr><td>{escape(stage["name"])}</td><td>{measurement["elapsed_seconds"]:.6g}</td>'
                              f'<td>{escape(str(peak["value"]))}</td><td>{escape(peak["status"])}</td></tr>')
    targets = manifest.get("finite_targets", {}).get("rows", [])
    target_rows = ''.join(f'<tr><td>{r["dimension"]}</td><td>{r["11/4"]:,}</td><td>{r["23/10"]:,}</td></tr>' for r in targets)
    growth_note = ('The full arithmetic growth report remains below.' if growth_available else
                   'The combined arithmetic page could not be produced; completed evidence remains linked below.')
    return ('<section class="panel" id="research-suite"><h2>Complete research suite</h2>'
            f'<p>Suite status: <strong>{escape(manifest["status"])}</strong>. '
            f'{growth_note} One suite run performs several kinds of experiment, rather than one large '
            'multiplication: it builds and searches for recipes, replays saved algebraic changes, checks '
            'exact identities, runs small diagnostic input pairs, and measures arithmetic across matrix '
            'sizes. The table is a map to those results. Use an HTML link for the explanation, JSON for '
            'the original data, Recipe for saved matrix coefficients, or Argument for the certificate notes.</p>'
            '<p>The budgets and parameter ranges are recorded so that another person can reproduce the '
            'experiment. We have not tested every possible size or searched every possible recipe. '
            'A “passed” search stage can still miss its requested improvement: it means the experiment '
            'completed its checks, not that its target was found. A search miss is not a proof of '
            'impossibility. CW tensors are mathematical construction evidence that may need more '
            'extraction and conversion before becoming one matrix multiplier. Their border rank is '
            'a bound involving a limiting polynomial construction; it is not a counted executable '
            'matrix exponent. The linked construction pages explain the finite recovery costs.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Stage</th><th>Family</th><th>Status</th><th>Evidence</th><th>Scope/outcome</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<details><summary>Separate setup and construction costs</summary>'
            '<p>Before a recipe can multiply data, we may have to generate coefficients, search for '
            'improvements, check identities and write its files. This table measures those preparation '
            'stages. Its time and Python allocation peak include the checks and file writing, and '
            'previously cached objects may already exist when the probe starts. The values are separate '
            'from the multiplication’s arithmetic counts; we do not yet count every raw coefficient '
            'operation in setup. Increasing matrix sizes does not automatically enlarge a search '
            'budget or tensor power. Preparing one fixed recipe is also not repeated for every '
            'individual scalar product when we apply that recipe.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Setup stage</th><th>Seconds</th><th>Traced peak bytes</th><th>Memory status</th></tr></thead>'
            f'<tbody>{"".join(setup_rows)}</tbody></table></div></details>'
            '<h3>Finite product-count targets</h3><p>These numbers tell us what a future square base '
            'recipe would need to achieve. For a given base size, we calculate the largest integer '
            'product count that would make its repeating exponent strictly smaller than the target '
            'in the column heading. The thresholds are not discovered algorithms or promises that '
            'such coefficients exist. In particular, the six-product target for a 2×2 multiplication '
            'is excluded by the known seven-product lower bound, so that base size cannot reach it.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Square base size</th><th>Below 2.75</th><th>Below 2.30</th></tr></thead>'
            f'<tbody>{target_rows}</tbody></table></div>'
            '<p><a href="suite-manifest.json">Complete settings, stage outcomes, and artifact fingerprints</a> · '
            '<a href="finite-targets.json">Exact target table and source</a>. '
            'The 2.30 column is an example research goal near 9/4=2.25. It does not mean that we '
            'have implemented the OpenAI method or obtained the finite coefficients needed for it.</p></section>')


def run_suite(sizes, cutoffs, *, algorithms=None, compositions=None, settings=None,
              seed=1729, output_root=None, reference_reports=(), progress=None,
              reuse_equivalent_cutoffs=True):
    """Produce one dated report tree with all supported families and a combined index.

    New verified matrix recipes join the measured size sweep automatically.
    Certificates establish finite identities/recurrences, not Python correctness.
    Mathematical search failure to reach its target leaves suite health intact;
    wrong products, work-budget mismatches, and failed proof stages do not.
    """
    settings = settings or SuiteSettings()
    settings.validate()
    # Materialize optional source iterables once: recording a generator in the
    # configuration must not consume it before the attachment stage can read it.
    reference_reports = tuple(reference_reports)
    sizes = positive_integer_grid(sizes, "sizes")
    cutoffs = positive_integer_grid(cutoffs, "cutoffs")
    names = list(ALGORITHMS) if algorithms is None else list(dict.fromkeys(algorithms))
    if not names or any(name not in ALGORITHMS for name in names):
        raise ValueError("choose at least one registered algorithm")
    root = Path(output_root) if output_root is not None else Path(__file__).resolve().parents[1] / "outputs"
    directory = root / datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    plans = dict(compositions or {})
    if any(not isinstance(name, str) or not name or name in ALGORITHMS for name in plans):
        raise ValueError("custom composition names must be nonempty and distinct from registered algorithms")
    reserved = {f"generated-search-{i}" for i in range(1, len(set(settings.search_seeds))+1)} | {
        "generated-tensor-product", "generated-cw-packed"}
    if reserved & plans.keys():
        raise ValueError("custom compositions use a reserved suite name")
    configuration = {"sizes": sizes, "cutoffs": cutoffs, "algorithms": names, "input_seed": seed,
                     "requested_compositions": {name: plan.model() for name, plan in plans.items()},
                     "reuse_equivalent_cutoffs": reuse_equivalent_cutoffs, "research": settings.document(),
                     "reference_reports": [str(path) for path in reference_reports]}
    source_root = Path(__file__).resolve().parents[1]
    configuration["source_sha256"] = {
        path.relative_to(source_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in [source_root / "sweep.py", *sorted((source_root / "matrix_multiplication").glob("*.py"))]}
    run = SuiteRun(directory, configuration, progress)
    # Phase 1: construct candidate identities. These are setup experiments;
    # searching, replaying, and interpolating are never counted as arithmetic
    # performed by the final multiplier. A missed rank target is useful evidence.
    for index, search_seed in enumerate(dict.fromkeys(settings.search_seeds), 1):
        label = f"search-{index}"
        request = ConstructionRequest("search", size=settings.search_size, start=settings.search_start,
                                      target_rank=settings.search_target_rank,
                                      target_exponent=settings.search_target_exponent, steps=settings.search_steps,
                                      restarts=settings.search_restarts, seed=search_seed,
                                      max_height=settings.search_max_height, extra_rank=settings.search_extra_rank,
                                      split_every=settings.search_split_every)
        stage = run.perform(label, "discovery", lambda: run.construction(label, request))
        if stage["status"] == "passed":
            candidate_name = f"generated-search-{index}"
            run.perform(f"register-{candidate_name}", "candidate integration", lambda: run.register_candidate(
                candidate_name, directory / stage["scheme"], plans))
            replay_label = f"replay-{index}"
            replay_request = ConstructionRequest("replay", report=directory / stage["data"])
            run.perform(replay_label, "replay", lambda: run.construction(replay_label, replay_request))
    product = run.perform("tensor-product", "construction", lambda: run.construction(
        "tensor-product", ConstructionRequest("product", left="strassen", right="strassen")))
    if product["status"] == "passed":
        run.perform("register-generated-tensor-product", "candidate integration", lambda: run.register_candidate(
            "generated-tensor-product", directory / product["scheme"], plans))
    packing_candidates = []
    for q in dict.fromkeys(settings.cw_parameters):
        for power in dict.fromkeys(settings.cw_powers):
            label = f"cw-q{q}-power{power}"
            run.perform(label, "polynomial tensor recovery", lambda: run.construction(label, ConstructionRequest("cw", q=q, power=power)))
            # Extraction is independent of the full recovery stage: it restricts
            # polynomial factors before interpolation and checks its own full
            # identity. Preserve this evidence even if full recovery fails.
            extraction_label = f"cw-extraction-q{q}-power{power}"
            run.perform(extraction_label, "verified independent product extraction", lambda: run.construction(
                extraction_label, ConstructionRequest("cw-extract", q=q, power=power)))
            packing_label = f"cw-packing-q{q}-power{power}"
            packed = run.perform(packing_label, "exact recovery optimization and dense packing", lambda: run.construction(
                packing_label, ConstructionRequest("cw-pack", q=q, power=power, size=settings.cw_packing_size)))
            if packed["status"] == "passed":
                packing_candidates.append((packing_label, directory / packed["scheme"], packed["rank"]))
            # Coarsening is a separate experiment, so failure in fine extraction
            # cannot suppress the merged-component catalog or vice versa. Only
            # its completely verified dense scheme joins candidate comparison;
            # unrecognized components remain visible in its own evidence report.
            coarse_label = f"cw-coarse-q{q}-power{power}"
            coarse = run.perform(coarse_label, "coarse CW classification and verified packing", lambda: run.construction(
                coarse_label, ConstructionRequest("cw-coarse", q=q, power=power, size=settings.cw_packing_size)))
            if coarse["status"] == "passed":
                packing_candidates.append((coarse_label, directory / coarse["scheme"], coarse["rank"]))

    def integrate_best_packing():
        """Measure one best finite packing; keep all construction comparisons.

        All candidates use the same recorded square base size, so comparing rank
        is an exact comparison of their recursive data-product exponents. Break
        ties by the declared arithmetic budget, then name, deterministically.
        Admit only cubic-or-better recipes: a worse exponent can make an ordinary
        growth grid unexpectedly enormous. Its base execution/cost report still
        remains available, and users can measure it explicitly as a composition.
        """
        candidates, decisions = [], []
        size = settings.cw_packing_size
        for label, path, rank in packing_candidates:
            decision = {"construction": label, "rank": rank, "base_dimension": size,
                        "admitted_to_comparison": rank <= size**3}
            decisions.append(decision)
            if rank <= size**3:
                plan = generated_plan(path)
                steps = plan.budget(size, size, size)["field_arithmetic_operations"]
                candidates.append((rank, steps, label, plan))
        if not candidates:
            return {"scope": "No cubic-or-better CW packing admitted to growth; finite construction evidence retained.",
                    "candidates": decisions}
        rank, steps, label, plan = min(candidates, key=lambda item: item[:3])
        plans["generated-cw-packed"] = plan
        return {"scope": f"Best finite CW packing from {label}; all candidate reports retained.",
                "selected_construction": label, "base_dimension": size, "base_bilinear_rank": rank,
                "base_field_arithmetic_operations": steps, "candidates": decisions}

    if packing_candidates:
        run.perform("register-generated-cw-packed", "candidate integration", integrate_best_packing)
    # Phase 2: independently check every plan before producing its certificate.
    # Build independent proof plans for registered algorithms. The naive bundle
    # certifies its classical identity, without pretending to certify its program.
    proof_plans = dict(plans)
    for name in names:
        def registered_certificate():
            """Construct and check one registered identity, retaining source notices."""
            plan = (Composition([to_scheme(classical((2, 2, 2)), (2, 2, 2), "classical-identity")])
                    if name == "naive" else Composition.from_spec(name))
            proof_plans[name] = plan
            return run.certificate(name, plan, "Classical base identity; not a proof of the naive Python program." if name == "naive"
                                   else "Exact base identities and repeating recurrence; not a proof of the execution program.")
        run.perform(f"certificate-{name}", "exact certificate", registered_certificate)
    for index, (name, plan) in enumerate(plans.items(), 1):
        run.perform(f"certificate-{name}", "exact certificate", lambda: run.certificate(
            f"generated-{index}", plan, f"Verified finite scheme/cycle for {name}; no claim of a record."))
    implementations = {name: ALGORITHMS[name] for name in names}
    implementations.update({name: plan.multiply for name, plan in plans.items()})
    # Keep registered execution callbacks and their actual models. The synthetic
    # naive identity is used only for proof, not for measuring the naive program.
    execution_plans = {name: plan for name, plan in proof_plans.items() if name != "naive"}
    # Phase 3: obtain separately scoped diagnostics on tiny exact and float pairs.
    # Their timings, memory peaks, and traces are useful for investigating code,
    # but cannot serve as samples in the mathematical size-growth comparison.
    pair_sources = []
    for label, floating in (("pair-exact", False), ("pair-floating", True)):
        stage = run.perform(label, "profiles, memory, accuracy, traces", lambda: run.pair(
            label, implementations, execution_plans, floating, settings.trace_limit))
        if "data" in stage:
            pair_sources.append(directory / stage["data"])
    data = None

    def measure_growth():
        """Count the original and generated algorithms over the requested grid."""
        nonlocal data
        measured = collect_work(sizes, cutoffs, algorithms=names, seed=seed, progress=progress,
                                compositions=plans, reuse_equivalent_cutoffs=reuse_equivalent_cutoffs,
                                measure_memory=settings.measure_sweep_memory,
                                projection_cycles=settings.projection_cycles)
        raw = (json.dumps(measured, indent=2, allow_nan=False)+"\n").encode("utf-8")
        # Save primary evidence before enriching it with secondary snapshots.
        # A missing earlier report must neither discard completed counts nor
        # leave the final renderer trying to read a file that was never created.
        (directory / "measurements.json").write_bytes(raw)
        data = measured
        raw = attach_reference_reports(raw, [*reference_reports, *pair_sources])
        (directory / "measurements.json").write_bytes(raw)
        data = json.loads(raw)
        samples = [sample for line in data["series"] for sample in line["samples"]]
        passed = all(sample["status"] == "passed" and sample["prediction_matches_observed_data_multiplications"] is True
                     and sample["prediction_matches_observed_arithmetic_steps"] is True for sample in samples)
        return {"status": "passed" if passed else "failed", "report": "index.html#arithmetic-growth",
                "data": "measurements.json", **data["execution_summary"]}

    # Phase 4: execute the requested size/cutoff grid with counting only, then
    # retain the small-pair snapshots alongside it with their own input scope.
    run.perform("arithmetic-growth", "measured arithmetic and theoretical comparison", measure_growth)
    run.manifest["finite_targets"] = target_evidence()
    write_json(directory / "finite-targets.json", run.manifest["finite_targets"])
    html = None

    def render_growth():
        """Build the arithmetic page as a fallible stage, retaining every section.

        The navigation is added after finalization so its status includes a
        renderer failure. A failed page renderer still leaves the original JSON,
        completed certificates, and diagnostic pages available from the fallback.
        """
        nonlocal html
        raw = (directory / "measurements.json").read_bytes()
        html = render_html(read_reports(raw), title="Matrix multiplication research suite", source_name="measurements.json",
                           source_hash=sha256(raw).hexdigest(), generated=datetime.now().isoformat(timespec="seconds"))
        return {"report": "index.html", "scope": "All existing arithmetic and diagnostic report sections"}

    if data is not None:
        run.perform("report-render", "combined HTML presentation", render_growth)
    run.finalize()
    panel = suite_panel(run.manifest, growth_available=html is not None)
    if html is not None:
        html = html.replace("<main>", "<main>"+panel+'<div id="arithmetic-growth"></div>', 1)
    else:
        # Reuse the accessible layout even when measurements or their renderer
        # failed. The user can read the error and follow successful stage links.
        html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>Research suite</title><style>{STYLE}</style></head><body><main>'
                '<h1>Matrix multiplication research suite</h1>'+panel+'</main></body></html>')
    destination = directory / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination, int(run.manifest["status"] != "passed")
