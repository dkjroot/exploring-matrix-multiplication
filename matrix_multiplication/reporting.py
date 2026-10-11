"""Render experiment evidence automatically as offline HTML with inline SVG charts.

Only the Python standard library is needed. Charts preserve the distinction
between an algorithm-family exponent, measured field work, and secondary runtime
diagnostics. Missing values are displayed as unavailable, never filled with zero.
Each generation creates a unique dated output directory with the original JSON.
"""

from datetime import datetime
from hashlib import sha256
from html import escape
import json
from math import isfinite, log10
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#6554c0", "#167d8d", "#be5733", "#446fba", "#91603c")
STYLE = """
:root { color-scheme: light; --ink:#202c40; --muted:#59677a; --line:#dce2eb; }
* { box-sizing:border-box; } body { margin:0; background:#f4f5f9; color:var(--ink);
font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1240px; margin:auto; padding:42px 28px 64px; }
header { border-top:5px solid #6554c0; padding:26px 0; }
.eyebrow { color:#6554c0; font-size:12px; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }
h1 { font-size:clamp(28px,4vw,42px); line-height:1.15; margin:12px 0 18px; }
h2 { font-size:23px; margin:0 0 10px; } h3 { font-size:18px; margin:20px 0 8px; }
p { margin:8px 0 18px; } .muted,.note { color:var(--muted); } .note { font-size:14px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }
.panel { background:white; border:1px solid var(--line); border-radius:12px; padding:24px; margin:0 0 22px; }
.grid .panel { margin:0; } .grid { margin-bottom:22px; }
.summary { display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }
.badge { border:1px solid var(--line); border-radius:8px; padding:9px 13px; background:white; }
.passed { color:#176349; } .failed,.error { color:#a33535; } .unknown { color:var(--muted); }
.table-wrap { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { border-bottom:1px solid var(--line); padding:11px 12px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:650; } td.number { font-variant-numeric:tabular-nums; white-space:nowrap; }
code { font-family:ui-monospace,Consolas,monospace; font-size:.9em; overflow-wrap:anywhere; }
svg { width:100%; max-width:900px; margin:auto; height:auto; display:block; } svg text { font-family:system-ui,sans-serif; fill:var(--ink); }
svg .axis { fill:var(--muted); font-size:12px; } details { margin:16px 0; }
summary { cursor:pointer; font-weight:650; } pre { overflow:auto; max-height:500px; padding:18px;
background:#f4f5f9; border-radius:8px; font-size:12px; } a { color:#5745b1; }
.callout { border-left:4px solid #be5733; padding:12px 18px; background:#fff6ee; border-radius:4px; }
footer { color:var(--muted); font-size:13px; } .hash { overflow-wrap:anywhere; }
@media(max-width:800px) { .grid { grid-template-columns:1fr; } main { padding:20px 14px; } .panel { padding:16px; } }
@media print { body { background:white; } main { max-width:none; padding:0; } .grid { display:block; }
.panel { margin:16px 0!important; break-inside:avoid; } .table-wrap { overflow:visible; } a { color:inherit; } }
"""


def text(value):
    """Escape every data-derived label for HTML text and quoted attributes."""
    return escape(str(value), quote=True)


def number(value):
    """Recognize finite nonnegative numeric measurements without treating bool as one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if value >= 0 and isfinite(value) else None
    except OverflowError:
        # Huge exact counters can still be displayed in tables, but cannot be
        # mapped reliably onto floating SVG coordinates.
        return None


def display(value):
    """Format a measurement or report its absence without inventing a zero."""
    if value is None:
        return "Unavailable"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6g}" if isfinite(value) else str(value)
    return str(value)


def reject_constant(value):
    """Reject nonstandard NaN/Infinity tokens in input JSON with a clear diagnostic."""
    raise ValueError(f"nonstandard JSON number: {value}")


def read_reports(raw):
    """Accept explorer reports or trace envelopes, checking rendered container types.

Older reports without mathematical/execution metadata remain readable. Numeric
fields may be absent; unusable chart values are explicitly listed as unavailable.
Unknown fields survive in the raw-data view and the unchanged source snapshot.
"""
    data = json.loads(raw, parse_constant=reject_constant)
    if isinstance(data, dict) and data.get("schema") == "matrix-multiplication-work-sweep-v1":
        if not isinstance(data.get("configuration"), dict) or not isinstance(data.get("series"), list) or not data["series"]:
            raise ValueError("work sweep: expected configuration and nonempty series")
        for line in data["series"]:
            if not isinstance(line, dict) or not isinstance(line.get("samples"), list) or not line["samples"]:
                raise ValueError("work sweep: each series needs samples")
            for sample in line["samples"]:
                if not isinstance(sample, dict) or number(sample.get("size")) in (None, 0):
                    raise ValueError("work sweep: samples need positive sizes")
                read_reports(json.dumps({"observation": sample.get("observation")}))
        if not isinstance(data.get("reference_reports", []), list):
            raise ValueError("work sweep: reference_reports must be a list")
        for reference in data.get("reference_reports", []):
            if not isinstance(reference, dict) or not isinstance(reference.get("source_json"), str):
                raise ValueError("work sweep: each reference needs its original JSON text")
            previous = read_reports(reference["source_json"])
            if previous.get("schema") == "matrix-multiplication-work-sweep-v1":
                raise ValueError("reference reports must be single-pair experiments, not nested sweeps")
        return data
    if isinstance(data, dict) and "reports" in data and "events" in data:
        data = data["reports"]
    if not isinstance(data, dict) or not data:
        raise ValueError("expected a nonempty object of algorithm reports")
    containers = ("counts", "profiles", "products_by_depth", "experiment",
                  "mathematical_model", "execution_measurements")
    for name, report in data.items():
        if not isinstance(report, dict) or not any(key in report for key in ("counts", "status", "profiles")):
            raise ValueError(f"{name}: expected an algorithm report")
        for key in containers:
            if key in report and not isinstance(report[key], dict):
                raise ValueError(f"{name}.{key}: expected an object")
        execution = report.get("execution_measurements", {})
        for key in ("accuracy", "python_traced_peak_bytes", "process_peak_memory_bytes",
                    "gpu_peak_memory_bytes", "hardware_memory_traffic_bytes", "hardware_instruction_count"):
            if key in execution and not isinstance(execution[key], dict):
                raise ValueError(f"{name}.execution_measurements.{key}: expected an object")
        if any(not isinstance(profile, dict) for profile in report.get("profiles", {}).values()):
            raise ValueError(f"{name}.profiles: each profile must be an object")
    return data


def table(headers, rows):
    """Render an escaped, horizontally scrollable data table with semantic headers."""
    head = "".join(f'<th scope="col">{text(label)}</th>' for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{text(display(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def bar_chart(title, entries, unit, *, maximum=None):
    """Draw a zero-based linear SVG bar chart with exact values and missing labels.

Each row includes a text value, so tiny bars and zero measurements remain legible.
No missing value is plotted as a zero. Tooltips retain full labels and values.
Charts are inline vectors: no plotting package, image asset, CDN, or JavaScript.
"""
    values = [value for _, value in entries if number(value) is not None]
    ceiling = max(values, default=0)
    if maximum is not None:
        ceiling = max(ceiling, maximum)
    ceiling = ceiling or 1
    width, left, plot, row_height = 680, 205, 315, 48
    height = 48 + row_height * len(entries)
    title_markup = text(title)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title_markup}">',
             f"<title>{title_markup}</title>",
             f"<desc>Linear bars start at zero. Units: {text(unit)}. Values are labelled in every row.</desc>"]
    for tick in range(5):
        x = left + plot * tick / 4
        parts.append(f'<line x1="{x}" x2="{x}" y1="25" y2="{height - 8}" stroke="#e4e8ef"/>')
        parts.append(f'<text x="{x}" y="16" text-anchor="middle" class="axis">{text(display(ceiling * tick / 4))}</text>')
    for index, (label, value) in enumerate(entries):
        y = 32 + index * row_height
        short_label = label if len(label) <= 28 else label[:25] + "…"
        label_size = 14 if len(short_label) > 22 else 16
        parts.append(f'<text x="8" y="{y + 19}" font-size="{label_size}"><title>{text(label)}</title>{text(short_label)}</text>')
        numeric = number(value)
        if numeric is not None:
            size = numeric / ceiling * plot
            parts.append(f'<rect x="{left}" y="{y}" width="{size:.5f}" height="27" rx="4" fill="{COLORS[index % len(COLORS)]}"><title>{text(label)}: {text(display(value))} {text(unit)}</title></rect>')
        value_text = display(value) if numeric is not None else "Unavailable"
        parts.append(f'<text x="{left + plot + 16}" y="{y + 19}" font-size="16">{text(value_text)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def panel(title, explanation, body):
    """Keep each graph adjacent to the interpretation required to read it correctly."""
    return f'<section class="panel"><h2>{text(title)}</h2><p class="note">{text(explanation)}</p>{body}</section>'


def comparison_warning(reports):
    """Flag unlike or undocumented input experiments without declaring a winner."""
    signatures = []
    incomplete = False
    for report in reports.values():
        experiment = report.get("experiment", {})
        keys = ("input_repr_sha256", "input_shapes", "cutoff", "atol", "rtol",
                "primitive_timing_enabled")
        incomplete |= any(key not in experiment for key in keys)
        signatures.append(json.dumps([experiment.get(key) for key in keys], sort_keys=True))
    if len(set(signatures)) > 1:
        return "These algorithms have different inputs, cutoffs, tolerances, or profiling settings. Side-by-side charts describe the runs; they do not establish a fair comparison."
    if incomplete:
        return "Some experiment metadata is missing. Comparability cannot be established from this report."
    memory_settings = {report.get("execution_measurements", {}).get("memory_tracking_enabled") for report in reports.values()}
    if len(memory_settings) > 1:
        return "Memory tracking settings differ across these runs; timings include different observation overhead."
    return "Runs share recorded inputs, cutoff, tolerances, and observation settings. Arithmetic domains and padding can still differ by algorithm."


def algorithm_details(name, report):
    """Expose exact counters, metadata, and primitive profiles behind a readable summary."""
    model = report.get("mathematical_model", {})
    experiment = report.get("experiment", {})
    execution = report.get("execution_measurements", {})
    accuracy = execution.get("accuracy", {})
    status = report.get("status", "unknown")
    status_class = status if status in ("passed", "failed", "error") else "unknown"
    body = f'<h3 class="{status_class}">{text(name)} · {text(status)}</h3>'
    if report.get("error"):
        body += f'<p class="callout">{text(report["error"])}</p>'
    rows = [("Input shapes", json.dumps(experiment.get("input_shapes"))),
            ("Classical leaf cutoff", experiment.get("cutoff")),
            ("Unpadded classical product baseline", experiment.get("classical_data_product_baseline")),
            ("Padded input cells (both matrices)", report.get("counts", {}).get("padded_input_cells")),
            ("Square family exponent", model.get("square_arithmetic_exponent")),
            ("Base dimension / bilinear rank", f'{display(model.get("base_dimension"))} / {display(model.get("base_bilinear_rank"))}'),
            ("Arithmetic domain", model.get("arithmetic_domain")),
            ("Recurrence", model.get("recurrence")),
            ("Base identity evidence (reported)", model.get("identity_validation")),
            ("Accuracy mode", accuracy.get("mode")),
            ("Maximum absolute error", accuracy.get("maximum_absolute_error")),
            ("Trace retained / dropped", f'{display(report.get("retained_trace_events"))} / {display(report.get("dropped_trace_events"))}')]
    body += table(("Property", "Reported value"), rows)
    if accuracy.get("failures"):
        body += f'<p class="callout">Failed entries: {text(json.dumps(accuracy["failures"]))}</p>'
    assumptions = model.get("assumptions")
    if assumptions:
        body += f'<p class="note">Model assumptions: {text(assumptions)}</p>'
    counts = sorted(report.get("counts", {}).items())
    body += "<details><summary>All recorded operation and structural counters</summary>"
    body += table(("Counter", "Count"), counts) + "</details>"
    if report.get("primitive_calls"):
        body += '<details><summary>Primitive call counts</summary>'
        body += table(("Primitive", "Calls"), sorted(report["primitive_calls"].items())) + '</details>'
    profiles = report.get("profiles", {})
    if profiles:
        timed = experiment.get("primitive_timing_enabled")
        if timed is False:
            body += '<p class="note">Primitive timing was disabled. Zero profile times mean clocks were off.</p>'
        elif timed is None:
            body += '<p class="note">Primitive timing settings are not recorded; zero values may mean clocks were disabled.</p>'
        rows = [(primitive, profile.get("calls"), profile.get("inclusive_seconds"), profile.get("exclusive_seconds"))
                for primitive, profile in sorted(profiles.items(), key=lambda item: number(item[1].get("exclusive_seconds")) or 0, reverse=True)]
        body += "<details><summary>Primitive profiles (inclusive times overlap)</summary>"
        body += table(("Primitive", "Calls", "Inclusive seconds", "Exclusive seconds"), rows) + "</details>"
    return body


def curve_chart(title, series, *, logarithmic=False):
    """Plot measured points and model curves with explicit linear or log-log axes.

Solid lines connect sampled executions, dashed lines show the ideal recurrence
curve, and crosses show padding-aware model predictions at measured dimensions.
No value on a theoretical curve is represented as a captured execution count.
"""
    all_points = [(x, y) for line in series for x, y in line["points"]
                  if number(x) not in (None, 0) and number(y) is not None and (not logarithmic or y > 0)]
    if not all_points:
        return '<p class="note">No successful positive work measurements available.</p>'
    transform = log10 if logarithmic else float
    xmin, xmax = min(transform(x) for x, _ in all_points), max(transform(x) for x, _ in all_points)
    ymin = min(transform(y) for _, y in all_points) if logarithmic else 0
    ymax = max(transform(y) for _, y in all_points)
    xmax = xmax if xmax > xmin else xmin + 1
    ymax = ymax if ymax > ymin else ymin + 1
    left, top, width, height = 100, 35, 720, 260
    parts = [f'<svg viewBox="0 0 900 355" role="img" aria-label="{text(title)}"><title>{text(title)}</title>',
             f'<desc>{"Log-log" if logarithmic else "Linear"} axes. Solid measured points; dashed ideal model; crosses are padding-aware predictions.</desc>']
    # Tick actual sampled dimensions, rather than incidental integer points in
    # the dense ideal curve. This makes the padding boundaries easier to read.
    sampled_xs = {x for line in series if line.get("kind") != "ideal"
                  for x, _ in line["points"] if number(x) not in (None, 0)}
    xs = sorted(sampled_xs or {x for x, _ in all_points})
    if len(xs) > 8:
        xs = [xs[round(index * (len(xs) - 1) / 7)] for index in range(8)]
    for value in xs:
        x = left + (transform(value) - xmin) / (xmax - xmin) * width
        parts.append(f'<line x1="{x}" x2="{x}" y1="{top}" y2="{top + height}" stroke="#e4e8ef"/><text x="{x}" y="{top + height + 23}" text-anchor="middle" font-size="14">{text(display(value))}</text>')
    for index in range(5):
        transformed = ymin + (ymax - ymin) * index / 4
        value = 10 ** transformed if logarithmic else transformed
        y = top + height - height * index / 4
        parts.append(f'<line x1="{left}" x2="{left + width}" y1="{y}" y2="{y}" stroke="#e4e8ef"/><text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="14">{text(display(value))}</text>')
    parts.append('<text x="460" y="345" text-anchor="middle" font-size="16">Requested square dimension n</text>')
    for line in series:
        points = [(x, y) for x, y in line["points"] if number(x) not in (None, 0) and number(y) is not None and (not logarithmic or y > 0)]
        color = COLORS[line["color"] % len(COLORS)]
        mapped = [(left + (transform(x) - xmin) / (xmax - xmin) * width,
                   top + height - (transform(y) - ymin) / (ymax - ymin) * height) for x, y in points]
        kind = line.get("kind", "observed")
        if kind != "prediction" and mapped:
            coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in mapped)
            dash = ' stroke-dasharray="8 6"' if kind == "ideal" else ""
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"{dash}><title>{text(line["label"])}</title></polyline>')
        if kind != "ideal":
            for (size, work), (x, y) in zip(points, mapped):
                tooltip = text(f'{line["label"]}: n={display(size)}, work={display(work)}')
                if kind == "prediction":
                    parts.append(f'<g stroke="{color}" stroke-width="2"><title>{tooltip}</title><line x1="{x - 7}" x2="{x + 7}" y1="{y}" y2="{y}"/><line x1="{x}" x2="{x}" y1="{y - 7}" y2="{y + 7}"/></g>')
                else:
                    parts.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"><title>{tooltip}</title></circle>')
    parts.append("</svg>")
    legend = '<div class="summary">' + "".join(
        f'<span class="note" style="color:{COLORS[line["color"] % len(COLORS)]}">{text(line["label"])}</span>'
        for line in series) + "</div>"
    return "".join(parts) + legend


def measured_series(lines, metric):
    """Extract only successful executed counts, keeping field and atomic layers apart."""
    result = []
    for index, line in enumerate(lines):
        points = []
        for sample in line["samples"]:
            if sample["status"] != "passed":
                continue
            observation = sample["observation"]
            value = observation["counts"].get(metric, 0) if metric == "field_multiplications" else observation.get(metric)
            if number(value) is not None:
                points.append((sample["size"], value))
        result.append({"label": f'{line["algorithm"]}, cutoff {line["cutoff"]}',
                       "color": index, "points": points, "kind": "observed"})
    return result


def comparison_series(lines, metric):
    """Overlay captured work with separately declared smooth and padded models.

    Older sweep JSON can lack total-step models; missing values produce no model
    points. Predicted crosses remain visible even when an execution failed, while
    successful executed counters alone define the measured curves.
    """
    result = []
    key = "data_multiplications" if metric == "field_multiplications" else metric
    for index, line in enumerate(lines):
        observed = measured_series([line], metric)[0]
        observed.update(color=index, label=f'cutoff {line["cutoff"]} · measured')
        result.append(observed)
        for kind, label, points in (
            ("ideal", "ideal model", [(point["size"], point.get(key)) for point in line.get("ideal_curve", [])]),
            ("prediction", "padded prediction", [(sample["size"], sample["prediction"].get(f"padded_{key}")) for sample in line["samples"]]),
        ):
            result.append({"label": f'cutoff {line["cutoff"]} · {label}', "color": index,
                           "kind": kind, "points": [(x, y) for x, y in points if number(y) is not None]})
    return result


def render_work_sweep(data, *, title, source_name, source_hash, generated):
    """Show measured count growth, declared ideal curves, and padding/cutoff effects."""
    lines = data["series"]
    configuration = data["configuration"]
    names = list(dict.fromkeys(line["algorithm"] for line in lines))
    samples = [sample for line in lines for sample in line["samples"]]
    passed = sum(sample["status"] == "passed" for sample in samples)
    matches = sum(sample.get("prediction_matches_observed_data_multiplications") is True for sample in samples)
    step_matches = sum(sample.get("prediction_matches_observed_arithmetic_steps") is True for sample in samples)
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>{text(title)}</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">Measured arithmetic growth</div>',
             f'<h1>{text(title)}</h1><p>Executed multiplication and arithmetic-step counts across matrix sizes and classical leaf blocks. Wall-clock time is not plotted.</p>',
             f'<p class="note">Generated {text(generated)} · Sizes {text(configuration.get("sizes"))} · Cutoffs {text(configuration.get("cutoffs"))} · Seed {text(configuration.get("seed"))}</p>',
             f'<div class="summary"><span class="badge">{len(samples)} executions</span><span class="badge">{passed} verified outputs</span><span class="badge">{matches} multiplication budgets match the recurrence</span><span class="badge">{step_matches} full arithmetic budgets match</span></div></header>',
             '<p class="callout">Theoretical exponents are calculated from declared scheme parameters. Measured slopes below are finite-range fits to captured counters, not proofs of an asymptotic exponent. A power law is straight on log-log axes; padding can produce plateaus.</p>']
    models = {line["algorithm"]: line["mathematical_model"] for line in lines}
    parts.append(panel("Square arithmetic exponents · theoretical",
                       "Calculated from declared base dimensions and ranks, independently of the executed sweep. Fixed leaf cutoffs preserve each family exponent.",
                       bar_chart("Square arithmetic exponents", [(name, model.get("square_arithmetic_exponent")) for name, model in models.items()], "exponent", maximum=3)))
    baseline_cutoff = min(line["cutoff"] for line in lines)
    baseline = [line for line in lines if line["cutoff"] == baseline_cutoff]
    for metric, heading in (("field_arithmetic_operations", "Measured field arithmetic steps"),
                            ("atomic_arithmetic_operations", "Measured component / atomic steps")):
        curves = measured_series(baseline, metric)
        body = curve_chart(heading, curves)
        body += '<details><summary>Log-log view</summary>' + curve_chart(heading, curves, logarithmic=True) + '</details>'
        explanation = (f"Cutoff {baseline_cutoff}. Field steps include data and coefficient multiplications, additions, subtractions, and normalization. Complex-field operations differ from real-field operations."
                       if metric == "field_arithmetic_operations" else
                       f"Cutoff {baseline_cutoff}. Exposed component calls include real arithmetic, exact divmods, and any opaque native complex operations. This overlaps the field layer: do not add the two totals.")
        parts.append(panel(heading, explanation, body))
    for name in names:
        group = [line for line in lines if line["algorithm"] == name]
        curves = comparison_series(group, "field_multiplications")
        model = group[0]["mathematical_model"]
        body = f'<p>Theoretical family exponent: <strong>{text(display(model["square_arithmetic_exponent"]))}</strong> ({text(model["exponent_expression"])}). {text(model.get("parameter_source", "Declared scheme parameters"))}.</p>'
        body += curve_chart(f"{name} data multiplications", curves)
        body += '<details><summary>Log-log view of multiplications</summary>' + curve_chart(f"{name} data multiplications", curves, logarithmic=True) + '</details>'
        body += '<details><summary>All arithmetic steps for these block sizes</summary>'
        body += '<p>Total-work models include quadratic additions, coefficient scaling, and normalization at each recursive level. Their smooth curves retain these lower-order terms; they are not simply a rescaled n<sup>p</sup>. Model constants describe our primitive schedule and are not fitted to the captured counters.</p>'
        for metric, heading in (("field_arithmetic_operations", "Field steps"), ("atomic_arithmetic_operations", "Component / atomic steps")):
            step_curves = comparison_series(group, metric)
            body += f'<h3>{heading}</h3>' + curve_chart(heading, step_curves)
            body += '<details><summary>Log-log view of arithmetic steps</summary>' + curve_chart(heading, step_curves, logarithmic=True) + '</details>'
        body += '</details>'
        rows = []
        for line in group:
            fits = line.get("empirical_fits", {})
            rows.append((line["cutoff"], model["square_arithmetic_exponent"],
                         fits.get("field_multiplications", {}).get("slope"),
                         fits.get("field_arithmetic_operations", {}).get("slope"),
                         fits.get("atomic_arithmetic_operations", {}).get("slope")))
        body += table(("Leaf cutoff", "Theoretical exponent", "Measured multiplication slope", "Measured field-step slope", "Measured component-step slope"), rows)
        body += '<p class="note">Fits use successful executed samples over the requested size range. Multiplication slopes can change with padding; total-step slopes also include coefficient work and lower-order terms. No finite fitted slope establishes a new mathematical bound.</p>'
        parts.append(panel(f"{name} · measured versus theoretical multiplications", "Solid points are captured counts; connecting lines are visual guides. Dashed curves predict ideal unpadded growth with the effective radix-power leaf. Crosses predict the padded recurrence at sampled sizes; matching crosses and points check that execution follows the model. Fixed cutoffs can change constants and finite work without changing the asymptotic family exponent.", body))
    rows = []
    for line in lines:
        for sample in line["samples"]:
            observation, predicted = sample["observation"], sample["prediction"]
            counts = observation["counts"]
            rows.append((line["algorithm"], line["cutoff"], sample["size"], predicted["padded_dimension"],
                         predicted["effective_leaf_dimension"], counts.get("field_multiplications", 0),
                         predicted["padded_data_multiplications"], counts.get("coefficient_multiplications", 0),
                         observation.get("field_arithmetic_operations"), observation.get("atomic_arithmetic_operations"), sample["status"]))
    parts.append(panel("Exact sampled counts and effective blocks", "Data multiplications multiply data-dependent values. Coefficient multiplications multiply by fixed constants. Field and component totals overlap. The configured cutoff is a maximum leaf size; actual recursive leaf sizes must be radix powers.",
                       table(("Algorithm", "Cutoff", "n", "Padded n", "Actual leaf n", "Data multiplications", "Padded prediction", "Coefficient multiplications", "Field steps", "Component steps", "Output check"), rows)))
    parts.append(sweep_evidence(data))
    for line in lines:
        for sample in line["samples"]:
            if sample["status"] != "passed":
                parts.append(f'<p class="callout">{text(line["algorithm"])} n={text(sample["size"])} cutoff={text(line["cutoff"])}: {text(sample.get("error", sample.get("verification")))}</p>')
    for reference in data.get("reference_reports", []):
        previous = read_reports(reference["source_json"])
        parts.append('<section class="panel"><h2>Earlier single-pair experiment</h2>')
        parts.append(f'<p class="note">Source: {text(reference.get("source_name"))}. These are preserved measurements of a separate input pair, not timing or memory measurements of the sweep. SHA-256: {text(reference.get("source_sha256"))}.</p>')
        parts.append('<details><summary>Earlier arithmetic, accuracy, timing, memory, and primitive evidence</summary>')
        parts.append(experiment_sections(previous))
        parts.append('<details><summary>Original earlier JSON snapshot</summary><pre>' + text(reference["source_json"]) + '</pre></details></details></section>')
    parts.append(f'<section class="panel"><h2>Reproducibility</h2><p><a href="measurements.json">Original sweep JSON</a> · Source: {text(source_name)}</p><p class="hash note">SHA-256: {text(source_hash)}</p>')
    parts.append('<details><summary>Complete data and captured counters</summary><pre>' + text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre></details></section>')
    parts.append('<footer>Observed work excludes input generation, the exact oracle, report calculations, and measurement bookkeeping. Counts assume unit-cost scalar operations; integer bit complexity is separate.</footer></main></body></html>')
    return "".join(parts)


def sweep_sample_report(line, sample):
    """Adapt captured sweep evidence to the shared single-experiment view.

    Shapes and classical baseline follow the sweep's square input contract.
    Accuracy and primitive counters are captured evidence. No wall-clock, memory,
    or tolerance measurement is invented; unavailable fields remain unavailable.
    """
    size = sample["size"]
    return {**sample["observation"], "status": sample["status"], "error": sample.get("error"),
            "mathematical_model": line["mathematical_model"],
            "experiment": {"input_shapes": [[size, size], [size, size]], "cutoff": line["cutoff"],
                           "input_repr_sha256": sample.get("input_repr_sha256"),
                           "classical_data_product_baseline": size ** 3, "primitive_timing_enabled": False},
            "execution_measurements": {"accuracy": sample.get("verification", {})}}


def sweep_evidence(data):
    """Restore breakdowns, recursion charts, and per-sample primitive evidence.

    Group expandable samples by algorithm and cutoff so the growing sweep does
    not flood the overview. Counters and profiles use the existing shared view;
    inclusive work still overlaps and must never be summed across primitives.
    """
    breakdown, accuracy = [], []
    for line in data["series"]:
        for sample in line["samples"]:
            report = sweep_sample_report(line, sample)
            key = (line["algorithm"], line["cutoff"], sample["size"])
            breakdown.append((*key, *(arithmetic_count(report, category) for category in (
                "field_multiplications", "coefficient_multiplications", "field_additions",
                "field_subtractions", "field_normalizations")),
                report.get("field_arithmetic_operations"), report.get("atomic_arithmetic_operations")))
            check = sample.get("verification", {})
            accuracy.append((*key, sample["status"], check.get("mode"), check.get("maximum_absolute_error"),
                             json.dumps(check.get("failures", [])), sample.get("error")))
    parts = [panel("Arithmetic work breakdown",
                   "Captured counts for every sample. Data and coefficient multiplications are separate; field and component totals overlap. Expand to inspect additions, subtractions, and normalization.",
                   '<details><summary>Full arithmetic breakdown across sizes and cutoffs</summary>' +
                   table(("Algorithm", "Cutoff", "n", "Data multiplications", "Coefficient multiplications", "Additions", "Subtractions", "Normalizations", "Field steps", "Atomic steps"), breakdown) + '</details>'),
             panel("Accuracy and verification evidence", "Every output is checked by the independent exact oracle outside the observed algorithm. Timing and memory probes are disabled for these sweep samples.",
                   '<details><summary>Accuracy for every executed sample</summary>' +
                   table(("Algorithm", "Cutoff", "n", "Status", "Verification mode", "Maximum absolute error", "Failed entries", "Error"), accuracy) + '</details>'),
             '<section class="panel"><h2>Per-sample primitive and recursion evidence</h2><p class="note">Primitive clocks were disabled: zero profile seconds mean clocks were off. Call counts and inclusive arithmetic remain captured. Inclusive profiles overlap.</p>']
    for line in data["series"]:
        parts.append(f'<details><summary>{text(line["algorithm"])} · cutoff {text(line["cutoff"])} · {len(line["samples"])} samples</summary>')
        for sample in line["samples"]:
            report = sweep_sample_report(line, sample)
            parts.append(f'<details><summary>n={text(sample["size"])} · {text(sample["status"])}</summary>')
            depths = sorted(report.get("products_by_depth", {}).items(), key=lambda item: int(item[0]))
            parts.append('<h3>Data products by recursion depth</h3>' + bar_chart("Data products by recursion depth",
                         [(f"depth {depth}", count) for depth, count in depths], "field products"))
            parts.append(algorithm_details(line["algorithm"], report))
            parts.append('<details><summary>Recurrence prediction and declared scalar costs</summary><pre>' +
                         text(json.dumps({"prediction": sample["prediction"], "work_model": line.get("work_model")}, indent=2)) + '</pre></details></details>')
        parts.append('</details>')
    parts.append('</section>')
    return "".join(parts)


def arithmetic_count(report, category):
    """Read an arithmetic counter, respecting only explicitly declared sparse zeros."""
    default = (0 if report.get("counter_convention") ==
               "omitted arithmetic categories have zero observed calls" else None)
    return report.get("counts", {}).get(category, default)


def experiment_sections(reports):
    """Reuse the full single-pair evidence view inside a combined sweep report."""
    return render_html(reports, title="", source_name="", source_hash="", generated="", sections_only=True)


def render_html(reports, *, title, source_name, source_hash, generated, sections_only=False):
    """Assemble a portable report that prioritizes arithmetic over hardware diagnostics."""
    if reports.get("schema") == "matrix-multiplication-work-sweep-v1":
        return render_work_sweep(reports, title=title, source_name=source_name,
                                 source_hash=source_hash, generated=generated)
    names = list(reports)
    exponents = [(name, report.get("mathematical_model", {}).get("square_arithmetic_exponent"))
                 for name, report in reports.items()]
    products = [(name, arithmetic_count(report, "field_multiplications")) for name, report in reports.items()]
    badges = []
    for name, report in reports.items():
        status = report.get("status", "unknown")
        color = status if status in ("passed", "failed", "error") else "unknown"
        badges.append(f'<span class="badge {color}">{text(name)} · {text(status)}</span>')
    parts = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>{text(title)}</title><style>{STYLE}</style></head><body><main>",
             f'<header><div class="eyebrow">Matrix multiplication · experiment report</div><h1>{text(title)}</h1>',
             '<p class="muted">Mathematical efficiency first. Execution diagnostics are recorded separately.</p>',
             f'<p class="note">Generated {text(generated)} · Source: {text(source_name)} · {len(names)} algorithms</p>',
             '<div class="summary">' + "".join(badges) + "</div></header>"]
    if sections_only:
        parts = []
    parts.extend([f'<p class="callout">{text(comparison_warning(reports))}</p>', '<div class="grid">',
             panel("Square arithmetic exponents · theoretical", "Declared base dimensions and product counts determine log(rank)/log(base size); classical multiplication is cubic. These are calculated family exponents, not inferred from executions. Missing models stay unavailable.",
                   bar_chart("Square arithmetic exponents", exponents, "exponent", maximum=3)),
             panel("Observed data multiplications", "Executed data-dependent multiplications, including padding and classical leaves. Complex-field and real-field products have different component costs; multiplying by fixed coefficients is shown separately.",
                   bar_chart("Observed data products", products, "field products")), "</div>"])
    rows = []
    for name, report in reports.items():
        rows.append((name, arithmetic_count(report, "field_multiplications"), arithmetic_count(report, "coefficient_multiplications"),
                     arithmetic_count(report, "field_additions"), arithmetic_count(report, "field_subtractions"), arithmetic_count(report, "field_normalizations"),
                     report.get("atomic_arithmetic_operations")))
    parts.append(panel("Arithmetic work breakdown", "Field and atomic counters are overlapping descriptions of the same work. Do not sum those layers. Omitted categories are zero only when the report explicitly declares that counting convention; older missing counters remain unavailable.",
                       table(("Algorithm", "Data multiplications", "Coefficient multiplications", "Additions", "Subtractions", "Normalizations", "Atomic steps"), rows)))
    depth_entries = [(f"{name} · depth {depth}", value) for name, report in reports.items()
                     for depth, value in sorted(report.get("products_by_depth", {}).items(), key=lambda item: str(item[0]).zfill(10))]
    if depth_entries:
        parts.append(panel("Data products by recursion depth", "Counts show where leaf products actually execute, not hypothetical work at every internal node. Cutoff and padding determine the observed depth.",
                           bar_chart("Data products by recursion depth", depth_entries, "field products")))
    parts.append('<div class="grid">')
    runtime = [(name, report.get("execution_measurements", {}).get("elapsed_algorithm_call_seconds")) for name, report in reports.items()]
    memory = []
    for name, report in reports.items():
        metric = report.get("execution_measurements", {}).get("python_traced_peak_bytes", {})
        memory.append((name, metric.get("value") if metric.get("status") == "measured" else None))
    parts.append(panel("Secondary · elapsed time", "Seconds for the public Python call, including cold setup and observation; excluding the correctness oracle. A single observed run is not a production benchmark.",
                       bar_chart("Elapsed algorithm call time", runtime, "seconds")))
    parts.append(panel("Secondary · Python allocation peak", "Bytes traced during the call, including setup and observation; excludes pre-existing inputs and untraced native allocations. This is not GPU or whole-process memory.",
                       bar_chart("Traced Python allocation peak", memory, "bytes")))
    parts.append("</div>")
    diagnostics = []
    for name, report in reports.items():
        execution = report.get("execution_measurements", {})
        accuracy = execution.get("accuracy", {})
        memory_status = execution.get("python_traced_peak_bytes", {}).get("status", "not_recorded")
        diagnostics.append((name, accuracy.get("mode"), accuracy.get("maximum_absolute_error"), memory_status,
                            execution.get("classical_equivalent_field_operations_per_second")))
    parts.append(panel("Accuracy and diagnostic scope", "The conventional equivalent rate is 2*M*N*K / elapsed seconds, not actual work or hardware FLOP/s. Accuracy uses the independent oracle. Missing and infinite display errors remain explicit.",
                       table(("Algorithm", "Verification mode", "Maximum absolute error", "Memory probe status", "Equivalent field ops/s"), diagnostics)))
    hardware_rows = []
    for name, report in reports.items():
        execution = report.get("execution_measurements", {})
        for metric in ("process_peak_memory_bytes", "gpu_peak_memory_bytes", "hardware_memory_traffic_bytes", "hardware_instruction_count"):
            observation = execution.get(metric, {})
            hardware_rows.append((name, metric, observation.get("status", "not_recorded"), observation.get("value")))
    parts.append('<section class="panel"><h2>Per-algorithm evidence</h2>')
    parts.extend(algorithm_details(name, report) for name, report in reports.items())
    parts.append("<details><summary>Hardware metric availability</summary>" + table(("Algorithm", "Metric", "Status", "Value"), hardware_rows) + "</details></section>")
    if sections_only:
        return "".join(parts)
    parts.append('<section class="panel"><h2>Reproducibility</h2><p><a href="measurements.json">Original JSON snapshot</a> · Use your browser’s Print command to save a PDF.</p>')
    parts.append(f'<p class="note hash">Source SHA-256: <code>{text(source_hash)}</code></p>')
    parts.append('<details><summary>Report data (trace events omitted from this view)</summary><pre>' + text(json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False)) + "</pre></details></section>")
    parts.append('<footer>Offline report generated automatically by matrix_multiplication.reporting. The source report supplies measurements and model claims; rendering does not independently certify an algorithm or establish a record.</footer></main></body></html>')
    return "".join(parts)


def attach_reference_reports(raw, sources):
    """Embed exact earlier JSON and its provenance without overwriting any source.

    Only explicitly supplied files are included. Earlier pair experiments retain
    their own input scope and diagnostic measurements, separate from sweep counts.
    Their full original JSON text also preserves any trace envelope and events.
    """
    data = read_reports(raw)
    if data.get("schema") != "matrix-multiplication-work-sweep-v1":
        raise ValueError("earlier reference reports can only be attached to a work sweep")
    references = data.setdefault("reference_reports", [])
    for source in sources:
        source = Path(source)
        previous_raw = source.read_bytes()
        previous = read_reports(previous_raw)
        if previous.get("schema") == "matrix-multiplication-work-sweep-v1":
            raise ValueError("reference reports must be single-pair experiments, not nested sweeps")
        references.append({"source_name": source.name, "source_sha256": sha256(previous_raw).hexdigest(),
                           "source_json": previous_raw.decode("utf-8")})
    return (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_data_report(raw, *, source_name, output_root=None, title="Matrix multiplication experiment", timestamp=None, reference_reports=()):
    """Render JSON bytes into a unique dated folder, preserving its exact snapshot."""
    if reference_reports:
        raw = attach_reference_reports(raw, reference_reports)
    reports = read_reports(raw)
    timestamp = timestamp or datetime.now().astimezone()
    rendered = render_html(reports, title=title, source_name=source_name,
                           source_hash=sha256(raw).hexdigest(), generated=timestamp.isoformat(timespec="seconds"))
    root = Path(output_root) if output_root is not None else ROOT / "outputs"
    directory = root / timestamp.strftime("%Y-%m-%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "measurements.json").write_bytes(raw)
    destination = directory / "index.html"
    destination.write_text(rendered, encoding="utf-8")
    return destination


def write_report(source, *, output_root=None, title="Matrix multiplication experiment", timestamp=None):
    """Create a unique dated folder and preserve the JSON used to render its HTML."""
    source = Path(source)
    return write_data_report(source.read_bytes(), source_name=source.name, output_root=output_root,
                             title=title, timestamp=timestamp)

