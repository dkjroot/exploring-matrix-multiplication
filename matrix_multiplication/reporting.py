"""Render experiment evidence automatically as offline HTML with inline SVG charts.

Only the Python standard library is needed. Charts preserve the distinction
between an algorithm-family exponent, measured field work, and secondary runtime
diagnostics. Missing values are displayed as unavailable, never filled with zero.
Each generation creates a unique dated output directory with the original JSON.
"""

from datetime import datetime
from fractions import Fraction
from hashlib import sha256
from html import escape
import json
from math import isfinite, log10
from pathlib import Path

from .report_explanations import composition_guide, reading_guide, terminology_guide
from .report_interactivity import range_chart_script


ROOT = Path(__file__).resolve().parents[1]
# Okabe and Ito's Color Universal Design palette, reordered so the first four
# series use black, orange, blue, and bluish green rather than adjacent blue/purple.
# Source and attribution: https://jfly.uni-koeln.de/color/#pallet
COLORS = ("#000000", "#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442")
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
.chart-legend { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }
.chart-series[data-hidden="true"] { display:none; }
.legend-button[aria-pressed="false"] { opacity:.5; text-decoration:line-through; }
.series-controls { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin:12px 0; }
.series-controls output { font-size:14px; color:var(--muted); }
.legend-button { display:flex; align-items:center; gap:8px; padding:8px 10px;
background:white; color:var(--ink); border:1px solid var(--line); border-radius:6px;
font:inherit; font-size:14px; cursor:pointer; text-align:left; }
.legend-button:hover,.legend-button:focus-visible { background:#edf2f7;
border-color:var(--ink); outline:2px solid var(--ink); outline-offset:2px; }
svg.legend-swatch { width:52px; height:18px; margin:0; flex:none; }
.chart-axis-label { font-size:15px; font-weight:600; }
.range-control { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin:14px 0; }
.range-control label { display:flex; flex-wrap:wrap; align-items:center; gap:12px; flex:1; }
.range-control input { flex:1; min-width:160px; max-width:400px; accent-color:#6554c0; }
.range-control output { font-variant-numeric:tabular-nums; }
.range-control button { font:inherit; padding:5px 10px; cursor:pointer; }
@media(max-width:800px) { .grid { grid-template-columns:1fr; } main { padding:20px 14px; } .panel { padding:16px; } }
@media print { body { background:white; } main { max-width:none; padding:0; } .grid { display:block; }
.panel { margin:16px 0!important; break-inside:avoid; } .table-wrap { overflow:visible; } a { color:inherit; }
.work-chart .chart-series { visibility:visible!important; } }
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


def axis_number(value):
    """Shorten large graph ticks; exact integers remain in tables and tooltips."""
    return f'{float(value):.3g}' if value >= 1_000_000 else display(value)


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


def bar_chart(title, entries, unit, *, maximum=None, category_label="Algorithm / series"):
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
    width, left, plot, row_height = 720, 240, 315, 48
    # A one-row depth chart still needs enough height for its rotated axis name.
    plot_height = max(row_height * len(entries), len(category_label) * 7 + 20)
    plot_bottom = 32 + plot_height
    height = plot_bottom + 48
    title_markup = text(title)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title_markup}">',
             f"<title>{title_markup}</title>",
             f"<desc>Linear bars start at zero. Units: {text(unit)}. Values are labelled in every row.</desc>"]
    for tick in range(5):
        x = left + plot * tick / 4
        parts.append(f'<line x1="{x}" x2="{x}" y1="25" y2="{plot_bottom}" stroke="#e4e8ef"/>')
        parts.append(f'<text x="{x}" y="16" text-anchor="middle" class="axis">{text(display(ceiling * tick / 4))}</text>')
    for index, (label, value) in enumerate(entries):
        y = 32 + (plot_height - row_height * len(entries)) / 2 + index * row_height
        short_label = label if len(label) <= 28 else label[:25] + "…"
        label_size = 14 if len(short_label) > 22 else 16
        parts.append(f'<text x="40" y="{y + 19}" font-size="{label_size}"><title>{text(label)}</title>{text(short_label)}</text>')
        numeric = number(value)
        if numeric is not None:
            size = numeric / ceiling * plot
            parts.append(f'<rect x="{left}" y="{y}" width="{size:.5f}" height="27" rx="4" fill="{COLORS[index % len(COLORS)]}"><title>{text(label)}: {text(display(value))} {text(unit)}</title></rect>')
        value_text = display(value) if numeric is not None else "Unavailable"
        parts.append(f'<text x="{left + plot + 16}" y="{y + 19}" font-size="16">{text(value_text)}</text>')
    axis_name = {"exponent": "Theoretical exponent p", "field products": "Data multiplications (count)",
                 "operations": "Arithmetic operations (count)", "seconds": "Elapsed time (seconds)",
                 "bytes": "Python allocations (bytes)"}.get(unit, unit)
    parts.append(f'<text class="chart-axis-label" data-axis="x" x="{left + plot / 2}" y="{height - 12}" text-anchor="middle">{text(axis_name)}</text>')
    parts.append(f'<text class="chart-axis-label" data-axis="y" transform="translate(17 {(32 + plot_bottom) / 2}) rotate(-90)" text-anchor="middle">{text(category_label)}</text>')
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
        signatures.append(json.dumps([experiment.get(key) for key in keys]
                                     + [experiment.get('size_policy', 'padded')], sort_keys=True))
    if len(set(signatures)) > 1:
        return "These algorithms have different inputs, cutoffs, size policies, tolerances, or profiling settings. Side-by-side charts describe the runs; they do not establish a fair comparison."
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
            ("Size policy", experiment.get("size_policy", "padded (legacy report)")),
            ("Fringe data products", report.get("counts", {}).get("fringe_data_products", 0)),
            ("Unpadded classical product baseline", experiment.get("classical_data_product_baseline")),
            ("Padded input cells (baseline only)", report.get("counts", {}).get("padded_input_cells")),
            ("Square family exponent", model.get("square_arithmetic_exponent")),
            ("Base dimension / bilinear rank", f'{display(model.get("base_dimension"))} / {display(model.get("base_bilinear_rank"))}'),
            ("Arithmetic domain", model.get("arithmetic_domain")),
            ("Recurrence", model.get("recurrence")),
            ("Base identity evidence (reported)", model.get("identity_validation")),
            ("Accuracy mode", accuracy.get("mode")),
            ("Maximum absolute error", accuracy.get("maximum_absolute_error")),
            ("Trace retained / dropped", f'{display(report.get("retained_trace_events"))} / {display(report.get("dropped_trace_events"))}')]
    body += table(("Property", "Reported value"), rows)
    if report.get("measurement"):
        body += '<p class="note">Measurement source: ' + text(measurement_label(report, experiment.get("cutoff"))) + '.</p>'
    if model.get("schedule"):
        body += '<h4>Repeating composition schedule</h4>'
        body += f'<p>{text(" → ".join(model["schedule"]))} → repeat</p>'
        body += '<p>Follow the names from left to right as the calculation moves to smaller blocks. When the last named stage has been used, the next deeper level starts again with the first. Each stage performs the recipe on all the problems at that depth.</p>'
        budget = report.get("composition_budget")
        if budget:
            body += table(("Depth", "Stage", "Nodes", "Square size", "Radix", "Rank"),
                          [(level["depth"], level["stage"], level["nodes"], level["size"],
                            level["radix"], level["rank"]) for level in budget["levels"]])
            body += table(("Metric", "Predicted", "Captured", "Matches"),
                          [(key, budget[key], report.get("counts", {}).get(key, 0)
                            if key == "field_multiplications" else report.get(key),
                            report.get("composition_budget_matches", {}).get(key))
                           for key in ("field_multiplications", "field_arithmetic_operations", "atomic_arithmetic_operations")])
        body += '<p class="note">To calculate the family exponent, we first work out how much one complete cycle shrinks the dimensions and how many descendant products it creates. Padding and stopping part-way through a cycle can make small measured examples grow differently from that long-term model. See the size and leaf tables before interpreting a fitted slope.</p>'
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
        body += '<p>A primitive is a reusable building block in our code, such as adding scalar numbers, splitting a matrix into blocks, or evaluating one recipe. These counts tell you how often each building block was called. Different levels of the hierarchy call one another, so adding every primitive’s count would not give a total arithmetic cost.</p>'
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
        body += '<p>Inclusive time covers a function and the child functions it calls. Exclusive time removes the measured child-call time. For example, a block multiplication includes the scalar work inside it; that scalar work also has its own profile. Do not add inclusive times across the hierarchy, because they overlap. These timings include our reference interpreter and measurement machinery, rather than describing an optimized hardware kernel.</p>'
        body += table(("Primitive", "Calls", "Inclusive seconds", "Exclusive seconds"), rows) + "</details>"
    return body


def point_marker(x, y, group, color, *, radius=4.5):
    """Draw distinct circle/square/triangle/diamond markers without relying on hue."""
    shape = group % 4
    attrs = f'fill="{color}" stroke="{color}" stroke-width="1.5"'
    if group % 8 >= 4:
        attrs = f'fill="white" stroke="{color}" stroke-width="2"'
    if shape == 0:
        return f'<circle cx="{x}" cy="{y}" r="{radius}" {attrs}/>'
    if shape == 1:
        return f'<rect x="{x - radius}" y="{y - radius}" width="{2 * radius}" height="{2 * radius}" {attrs}/>'
    points = ([(x, y - radius - 1), (x + radius + 1, y + radius), (x - radius - 1, y + radius)]
              if shape == 2 else [(x, y - radius - 1), (x + radius + 1, y), (x, y + radius + 1), (x - radius - 1, y)])
    return f'<polygon points="{" ".join(f"{px},{py}" for px, py in points)}" {attrs}/>'


def legend_swatch(line):
    """Match each legend's line style and marker to its plotted series."""
    group, kind = line["color"], line.get("kind", "observed")
    color = COLORS[group % len(COLORS)]
    body = (f'<path d="M18 9h16 M26 2v14" stroke="{color}" stroke-width="2"/>' if kind == "prediction" else
            f'<path d="M2 9h48" stroke="{color}" stroke-width="2.5"' +
            (' stroke-dasharray="6 4"' if kind == "ideal" else ' stroke-dasharray="2 5"' if kind == 'projected' else ' stroke-dasharray="10 3 2 3"' if kind == 'upper_bound' else '') + '/>')
    if kind == "observed":
        body += point_marker(26, 9, group, color, radius=4)
    return f'<svg class="legend-swatch" viewBox="0 0 52 18" aria-hidden="true">{body}</svg>'


def metric_axis_label(metric):
    """Name the counted arithmetic layer without implying a hardware measurement."""
    return {"field_multiplications": "Data multiplications (count)",
            "field_arithmetic_operations": "Field arithmetic steps (count)",
            "atomic_arithmetic_operations": "Component / atomic steps (count)"}[metric]


def curve_chart(title, series, *, logarithmic=False, y_label="Arithmetic operations (count)"):
    """Plot measured points and model curves with explicit linear or log-log axes.

Solid lines connect sampled executions, dashed lines show the ideal recurrence
curve, and crosses show size-policy-aware model predictions at measured dimensions.
No value on a theoretical curve is represented as a captured execution count.
"""
    series = [{**line, 'line_id': f'l{index}'} for index, line in enumerate(series)]
    full_series = series
    limits = next((line['range_limits'] for line in series if line.get('range_limits')), [])
    bounded = any(line.get('kind') == 'upper_bound' for line in series)
    if limits:
        series = [{**line, 'points': [(x, y) for x, y in line['points']
                                     if x <= limits[0] and line.get('kind') not in ('projected', 'upper_bound')]}
                  for line in series]
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
    left, top, width, height = 120, 35, 700, 260
    parts = [f'<svg viewBox="0 0 900 365" role="img" aria-label="{text(title)}"><title>{text(title)}</title>',
             f'<desc>{"Log-log" if logarithmic else "Linear"} axes. Solid measured points; dashed ideal model; crosses are size-policy-aware predictions.</desc>']
    # Tick actual sampled dimensions, rather than incidental integer points in
    # the dense ideal curve. This makes the padding boundaries easier to read.
    sampled_xs = {x for line in series if line.get("kind") != "ideal"
                  for x, _ in line["points"] if number(x) not in (None, 0)}
    xs = sorted(sampled_xs or {x for x, _ in all_points})
    if len(xs) > 8:
        xs = [xs[round(index * (len(xs) - 1) / 7)] for index in range(8)]
    for value in xs:
        x = left + (transform(value) - xmin) / (xmax - xmin) * width
        parts.append(f'<line x1="{x}" x2="{x}" y1="{top}" y2="{top + height}" stroke="#e4e8ef"/><text x="{x}" y="{top + height + 23}" text-anchor="middle" font-size="14">{text(axis_number(value))}</text>')
    for index in range(5):
        transformed = ymin + (ymax - ymin) * index / 4
        value = 10 ** transformed if logarithmic else transformed
        y = top + height - height * index / 4
        parts.append(f'<line x1="{left}" x2="{left + width}" y1="{y}" y2="{y}" stroke="#e4e8ef"/><text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="14">{text(display(value))}</text>')
    scale_label = " (log scale)" if logarithmic else ""
    parts.append(f'<text class="chart-axis-label" data-axis="x" x="470" y="350" text-anchor="middle">Requested square dimension n{scale_label}</text>')
    parts.append(f'<text class="chart-axis-label" data-axis="y" transform="translate(22 165) rotate(-90)" text-anchor="middle">{text(y_label)}{scale_label}</text>')
    for line in series:
        points = [(x, y) for x, y in line["points"] if number(x) not in (None, 0) and number(y) is not None and (not logarithmic or y > 0)]
        color = COLORS[line["color"] % len(COLORS)]
        mapped = [(left + (transform(x) - xmin) / (xmax - xmin) * width,
                   top + height - (transform(y) - ymin) / (ymax - ymin) * height) for x, y in points]
        kind = line.get("kind", "observed")
        parts.append(f'<g class="chart-series" data-series-id="{line["line_id"]}" data-series-group="g{line["color"]}" data-series-kind="{text(kind)}">')
        if kind != "prediction" and mapped:
            coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in mapped)
            dash = ' stroke-dasharray="8 6"' if kind == "ideal" else ' stroke-dasharray="2 5"' if kind == 'projected' else ""
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"{dash}><title>{text(line["label"])}</title></polyline>')
        if kind not in ("ideal", "projected"):
            for (size, work), (x, y) in zip(points, mapped):
                tooltip = text(f'{line["label"]}: n={display(size)}, work={display(work)}')
                if kind == "prediction":
                    parts.append(f'<g stroke="{color}" stroke-width="2"><title>{tooltip}</title><line x1="{x - 7}" x2="{x + 7}" y1="{y}" y2="{y}"/><line x1="{x}" x2="{x}" y1="{y - 7}" y2="{y + 7}"/></g>')
                else:
                    parts.append(f'<g><title>{tooltip}</title>{point_marker(x, y, line["color"], color)}</g>')
        parts.append('</g>')
    parts.append("</svg>")
    legend = ('<div class="series-controls" role="group" aria-label="Choose visible lines">'
              '<button type="button" data-series-select="all">Select all</button>'
              '<button type="button" data-series-select="none">Select none</button>'
              f'<output data-series-status aria-live="polite">{len(series)} of {len(series)} lines enabled</output></div>'
              '<div class="chart-legend" aria-label="Chart legend">') + "".join(
        f'<button type="button" class="legend-button" data-series-id="{line["line_id"]}" data-series-group="g{line["color"]}" aria-pressed="true" aria-label="Toggle {text(line["label"])}">{legend_swatch(line)}<span>{text(line["label"])}</span></button>'
        for line in series) + "</div>"
    # :has scopes hover/focus to this chart. All series in the same group stay
    # visible, including ideal curves and execution-policy predictions. Hover
    # isolation uses CSS; click selections are reapplied after range redraws.
    rules = []
    for group in sorted({line["color"] for line in series}):
        trigger = f'.legend-button[data-series-group="g{group}"][aria-pressed="true"]'
        rules.append(f'.work-chart:has({trigger}:hover) .chart-series:not([data-series-group="g{group}"]),'
                     f'.work-chart:not(:has(.legend-button:hover)):has({trigger}:focus-visible) .chart-series:not([data-series-group="g{group}"])'
                     '{visibility:hidden;}')
    observed = [line["points"] for line in series if line.get("kind", "observed") == "observed"]
    overlap = any(a == b and a for index, a in enumerate(observed) for b in observed[index + 1:])
    note = '<p class="note">Click legend entries to turn individual lines on or off. Select none, then enable two entries to compare them. Hover an entry or focus it with Tab to temporarily isolate its related visible curves; moving away restores your selection. Selections survive range changes and Reset.</p>'
    if overlap:
        note += '<p class="note">Some measured series have identical counts and overlap. Isolate their legend entries to see each one separately.</p>'
    attributes, controls = '', ''
    if limits:
        # Integers travel as decimal strings so browser JSON parsing cannot round
        # authoritative counts above 2^53. JavaScript converts only coordinates.
        payload = {'title': title, 'yLabel': y_label, 'logarithmic': logarithmic,
                   'limits': [str(n) for n in limits], 'colors': list(COLORS), 'bounded': bounded,
                   'series': [{**{k: line[k] for k in ('label', 'color', 'line_id')},
                               'kind': line.get('kind', 'observed'),
                               'points': [[str(x), str(y)] for x, y in line['points']
                                          if number(x) not in (None, 0) and number(y) is not None]}
                              for line in full_series]}
        attributes = ' data-range-chart="' + text(json.dumps(payload, separators=(',', ':'), allow_nan=False)) + '"'
        controls = ('<div class="range-control"><label>Extend dimension range '
                    f'<input type="range" min="0" max="{len(limits)-1}" step="1" value="0" '
                    f'aria-label="Maximum dimension for {text(title)}" /></label>'
                    f'<output aria-live="polite">Sampled range through n={text(display(limits[0]))}</output>'
                    '<button type="button" data-range-reset>Reset</button></div>')
        note += ('<p class="note">' + ('Drag to reveal proved upper bounds for integer inputs with |entry| ≤ 4. These bound the worst case; they do not predict the observed curve. Dash-dot lines connect separately bounded integer dimensions; the connecting segments are not themselves guarantees. Bounds can overestimate actual work because they allow cancellation to be absent. '
                 if bounded else 'Drag to reveal a dotted extension: exact schedule predictions, without matrix execution. ') +
                 'The range doubles at each step. Lines connect selected integer sizes; not every large-size fluctuation is sampled. '
                 'Measured points stay unchanged. Coordinates and axis labels are rounded for display.</p>'
                 '<noscript><p class="note">Enable JavaScript to extend this graph; the sampled graph and exact JSON remain available offline.</p></noscript>')
    return '<div class="work-chart"'+attributes+'><style>' + "".join(rules) + '</style>' + controls + "".join(parts) + legend + note + '</div>'


def projected_series(line, index, metric, label):
    """Adapt supported exact schedule counts into a distinct unexecuted curve."""
    projection = line.get('range_projection', {})
    if projection.get('status') != 'projected':
        return None
    if metric not in ('field_multiplications', 'coefficient_multiplications', 'field_additions',
                      'field_subtractions', 'field_normalizations',
                      'field_arithmetic_operations', 'atomic_arithmetic_operations'):
        return None
    points = [(p['size'], p['counts'][metric] if metric in p['counts'] else p[metric])
              for p in projection['points'] if p['size'] >= projection['sampled_maximum_size']]
    return {'label': label+' · projected schedule', 'color': index, 'kind': 'projected',
            'points': points, 'range_limits': projection['range_limits']}


def measured_series(lines, metric):
    """Extract only successful executed counts, keeping field and atomic layers apart."""
    result = []
    for index, line in enumerate(lines):
        points = []
        for sample in line["samples"]:
            if sample["status"] != "passed":
                continue
            observation = sample["observation"]
            value = observation["counts"].get(metric, 0) if metric.startswith('field_') and metric != 'field_arithmetic_operations' else observation.get(metric)
            if number(value) is not None:
                points.append((sample["size"], value))
        result.append({"label": f'{line["algorithm"]}, cutoff {line["cutoff"]}',
                       "color": index, "points": points, "kind": "observed"})
        projected = projected_series(line, index, metric, f'{line["algorithm"]}, cutoff {line["cutoff"]}')
        if projected:
            result.append(projected)
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
        projected = projected_series(line, index, metric, f'cutoff {line["cutoff"]}')
        if projected:
            result.append(projected)
        for kind, label, points in (
            ("ideal", "ideal model", [(point["size"], point.get(key)) for point in
                [*line.get("ideal_curve", []), *line.get('range_projection', {}).get('ideal_points', [])]]),
            ("prediction", "execution-policy prediction", [(sample["size"], sample["prediction"].get(f"padded_{key}")) for sample in line["samples"]]),
        ):
            result.append({"label": f'cutoff {line["cutoff"]} · {label}', "color": index,
                           "kind": kind, "points": [(x, y) for x, y in points if number(y) is not None]})
    return result


def measurement_label(sample, cutoff):
    """Name the actual evidence source while keeping older reports readable."""
    measurement = sample.get("measurement", {})
    kind = measurement.get("kind", "executed")
    if kind == "setup_error":
        return "Setup failed before execution"
    source = measurement.get("source_cutoff", cutoff)
    return f"Reused from cutoff {source}" if kind == "reused" else f"Executed at cutoff {source}"


def size_effects_section(lines):
    """Show where each finite step originates, including unsupported attribution.

    Geometry is discrete model evidence, not a memory/bit-work observation.
    Preserve older report readability by leaving absent metadata unavailable;
    do not rewrite old JSON or silently assert a callback's traversal from rank.
    """
    rows, transitions, cutoff_notes = [], [], []
    for line in lines:
        for sample in line['samples']:
            evidence = sample.get('size_effects', {})
            if evidence.get('status') not in ('known_reference_policy', 'declared_model_only'):
                continue
            padded = ' × '.join(map(str, evidence['padded_dimensions']))
            leaf = ' × '.join(map(str, evidence['leaf_dimensions']))
            trace = [evidence['padded_dimensions']]+[level['child_dimensions'] for level in evidence['levels']]
            path = ' → '.join('×'.join(map(str, shape)) for shape in trace)
            cores = ' → '.join('×'.join(map(str, level.get('core_dimensions', level['parent_dimensions'])))
                               for level in evidence['levels'])
            stages = ' → '.join(f"{level['stage']} ({level['child_products_per_node']} child products)" for level in evidence['levels'])
            rows.append((line['cutoff'], sample['size'], evidence.get('size_policy', 'padded'), padded, leaf, evidence['recursion_levels'],
                         evidence['extra_zero_input_cells'], 'Yes' if evidence['padding_added'] else 'No',
                         evidence.get('fringe_data_products', 0), path, cores or 'No split', stages or 'Classical leaf', evidence['status']))
            transition = evidence.get('transition_from_previous_sample')
            if transition:
                transitions.append((line['cutoff'], transition.get('from_requested_size'),
                    transition.get('to_requested_size'), transition['kind'],
                    transition.get('observed_data_products_equal'), transition['reason']))
            if evidence.get('padding_triggered_recursion') and evidence['status'] == 'known_reference_policy':
                cutoff_notes.append(f"n={sample['size']}, cutoff {line['cutoff']}: the requested matrix already fits the cutoff, "
                    f"but padding to {padded} happens first and triggers recursion. This is our current ordering choice, not a mathematical requirement.")
    if not rows:
        return ''
    body = ('<details><summary>Where this algorithm’s steps come from</summary>'
            '<p>A recipe tells us how to split a block once we have chosen to apply it. Our implementation separately decides '
            'whether to enlarge the original matrix with zeros and when to stop splitting. This table lets you follow those '
            'decisions from the requested size to the final leaf. Each three-number shape means (rows, shared inner dimension, '
            'columns), as explained in the glossary. The path is calculated from the recorded schedule; the arithmetic counters '
            'and output checks provide separate execution evidence. A row marked “declared_model_only” describes the model '
            'but does not prove how an unfamiliar replacement function actually handled sizes.</p>'
            +table(('Cutoff', 'Requested n', 'Size policy', 'Executed shape', 'Classical leaf shape', 'Levels',
                    'Extra zero input cells', 'Padding added?', 'Fringe data products', 'Dimension path', 'Core path', 'Stage path', 'Attribution'), rows))
    if transitions:
        body += table(('Cutoff', 'Previous n', 'Current n', 'Budget transition', 'Observed data counts equal?', 'Why'), transitions)
    body += ''.join('<p class="callout">'+text(note)+'</p>' for note in cutoff_notes)
    body += ('<p>The default fringe policy tests the cutoff on the actual block, applies the recipe to a divisible core, '
             'and computes the remaining dot terms explicitly. No input zeros are added. At every level the core rounds '
             'each axis down by less than its stage block count; the fringe count includes all levels and their node multiplicities. '
             'For square matrices and fixed stages this adds only quadratic work per node, preserving the justified family exponent. '
             'A stage with an empty core falls back to classical multiplication. The optional padded baseline instead enlarges '
             'the whole request to a covering schedule prefix before checking the cutoff. Earlier saved reports retain that policy. '
             'This is a portable size-handling choice, not a Python/GPU/home-computer restriction. Zero padding preserves the '
             'answer and the fixed-cutoff asymptotic exponent, while sometimes adding substantial work in a finite example. '
             'Two requests with equal field counts can still produce different-sized numbers or different memory peaks.</p></details>')
    return body


def mathematical_work_explanation():
    """Share the work-unit definition and limitations between report families."""
    return ('The score compares the same input under a common exact-arithmetic model: classical work divided by candidate work. '
            'One means equal estimated work; two means half the work of classical multiplication. Higher is better. '
            'It is not a speedup prediction or a measure of proximity to an optimal algorithm. Each 64-bit word product, addition, '
            'subtraction, carry/borrow step, comparison, shift and sign step has weight one. Those weights are a declared convention, '
            'not measured instruction prices. The reference schedule uses schoolbook multiplication, binary restoring division and '
            'Euclidean GCD. Rational addition shares denominator factors; multiplication cancels common factors before forming products. '
            'Reduction and exact division are charged, as are coefficient arithmetic and exact complex components. '
            'We check each replayed result against the original scalar result without changing the algorithm. '
            'Memory, movement, preparation, observation and verification costs are separate. Partial or opaque arithmetic gets no headline score. '
            'The exact numerator and denominator of each ratio are saved in JSON. Rankings can change with inputs, representation, cutoff or another cost model. '
            'Scores describe the explicit executable schedule, including fixed-coefficient scales even by one; equivalent identities can have different schedules. '
            'They do not establish the minimum work of every possible implementation of an identity.')


def traffic_explanation():
    """Define logical payload transfers without implying hardware traffic."""
    return ('This model counts reads and writes at scalar arithmetic ports, plus explicit matrix/vector copies, zero initialization and stores. '
            'A copy reads and writes the value once; repeated uses count repeatedly. Integers use 64-bit magnitude words, '
            'fractions include numerator and denominator, and exact complex pairs include both components. '
            'For example, copying a 4,096-bit integer counts 64 words read and 64 written. Python may actually copy only a reference: '
            'we are describing logical mathematical payload flow, not measuring cache or DRAM traffic. '
            'Internal scalar temporaries, pointer/object overhead, setup, validation, observer replay and verification are excluded. '
            'There is no extrapolation from a few measured operand sizes. Unknown payloads remain unavailable.')


def comparison_value(comparison):
    """Convert an exact ratio only for plotting; raw evidence stays integral."""
    if comparison.get('status') != 'compared':
        return None
    ratio = comparison['relative_efficiency']
    try:
        value = float(Fraction(ratio['numerator'], ratio['denominator']))
        return value if value > 0 and isfinite(value) else None
    except OverflowError:
        return None


def implementation_comparison_sections(lines):
    """Show separate work, allocation-memory, and logical-movement comparisons.

    Main curves use the smallest requested cutoff, matching the arithmetic
    overview. Every cutoff retains exact raw evidence in the expandable table.
    Older input files get no invented score or modelled movement measurement.
    """
    baseline_cutoff = min(line['cutoff'] for line in lines)
    baseline = [line for line in lines if line['cutoff'] == baseline_cutoff]
    work_curves, memory_curves, movement_curves, rows, statistics, traffic_rows = [], [], [], [], [], []
    has_work = any('mathematical_work' in sample.get('observation', {}) for line in lines for sample in line['samples'])
    for index, line in enumerate(baseline):
        scores, memory, movement = [], [], []
        for sample in line['samples']:
            if sample['status'] != 'passed':
                continue
            observation = sample.get('observation', {})
            value = comparison_value(sample.get('work_comparison', {}))
            if value is not None:
                scores.append((sample['size'], value))
            peak = sample.get('execution_measurements', {}).get('python_traced_peak_bytes', {})
            if peak.get('status') == 'measured':
                memory.append((sample['size'], peak['value']))
            traffic = observation.get('data_movement', {})
            if traffic.get('status') == 'modelled':
                movement.append((sample['size'], traffic['logical_word_transfers']))
        label = line['algorithm']+f", cutoff {line['cutoff']}"
        work_curves.append({'label': label, 'color': index, 'points': scores})
        memory_curves.append({'label': label, 'color': index, 'points': memory})
        movement_curves.append({'label': label, 'color': index, 'points': movement})
    for line in lines:
        for sample in line['samples']:
            observation = sample.get('observation', {})
            work = observation.get('mathematical_work')
            traffic = observation.get('data_movement', {})
            if work is None:
                continue
            score = sample.get('work_comparison', {})
            ratio = score.get('relative_efficiency')
            exact_ratio = f"{ratio['numerator']}/{ratio['denominator']}" if ratio else None
            rows.append((line['algorithm'], line['cutoff'], sample['size'], sample['status'], work['status'],
                         work.get('total_work_units'), score.get('baseline_work_units'), exact_ratio,
                         traffic.get('logical_word_transfers'), traffic.get('status', 'not_recorded')))
            for key, value in work.get('schedule_statistics', {}).items():
                statistics.append((line['algorithm'], line['cutoff'], sample['size'], key, value))
            for category, counts in traffic.get('by_category', {}).items():
                traffic_rows.append((line['algorithm'], line['cutoff'], sample['size'], category,
                                     counts.get('scalar_reads', 0), counts.get('scalar_writes', 0),
                                     counts.get('word_reads', 0), counts.get('word_writes', 0),
                                     counts.get('unknown_payload_reads', 0)+counts.get('unknown_payload_writes', 0)))
    parts = []
    if has_work:
        parts.append('<nav aria-label="Candidate comparisons"><a href="#mathematical-work">Work efficiency</a> · '
                     '<a href="#comparison-memory">Memory</a> · <a href="#logical-movement">Logical data movement</a></nav>')
        if not any(line['algorithm'] == 'naive' for line in baseline):
            sizes = sorted({x for curve in work_curves for x, _ in curve['points']})
            work_curves.append({'label': 'Classical reference = 1', 'color': len(baseline), 'kind': 'prediction',
                                'points': [(size, 1) for size in sizes]})
        parts.append('<div id="mathematical-work"></div>'+panel('Mathematical work efficiency', mathematical_work_explanation()+
                           f' These curves use cutoff {baseline_cutoff}; the table below retains all cutoffs. '
                           'The classical denominator reuses a verified classical sample where available, or comes from a separate verified execution on identical inputs, outside candidate measurement scopes.',
                           curve_chart('Mathematical work efficiency', work_curves, y_label='Classical / candidate work (ratio)')
                           if any(curve['points'] for curve in work_curves) else '<p class="note">Unavailable: no complete verified work comparisons.</p>'))
    if has_work or any(curve['points'] for curve in memory_curves):
        parts.append('<div id="comparison-memory"></div>'+panel('Measured Python memory growth',
            'This separate graph records the highest Python-traced allocation total during each public multiplication call. '
            'Lower is better within this measurement scope. Inputs already exist; temporaries, cold preparation and observation machinery contribute. '
            'The new arithmetic replay also contributes observer allocations, so this is not intrinsic algorithm storage or a whole-process/GPU peak. '
            'Reused rows copy original evidence. Disabled memory probes remain unavailable. There is deliberately no extrapolation slider: '
            'we have no justified model of allocator behavior or peak live payload storage at unexecuted sizes.',
            curve_chart('Python allocation peaks', memory_curves, y_label='Traced peak allocation (bytes)')
            if any(curve['points'] for curve in memory_curves) else '<p class="note">Unavailable: no verified Python allocation observations; the probe may be disabled.</p>'))
    if has_work:
        parts.append('<div id="logical-movement"></div>'+panel('Logical data movement', traffic_explanation()+' Lower transfer counts are better within this declared schedule.',
                           curve_chart('Logical data movement', movement_curves, y_label='Logical 64-bit word reads + writes (model)')
                           if any(curve['points'] for curve in movement_curves) else '<p class="note">Unavailable: no complete logical payload observations.</p>'))
        parts.append('<details><summary>Exact work scores, coverage and schedule statistics</summary>'+table(
            ('Algorithm', 'Cutoff', 'n', 'Execution', 'Work coverage', 'Candidate work units', 'Classical work units',
             'Exact relative efficiency', 'Logical word transfers', 'Movement coverage'), rows)
            +table(('Algorithm', 'Cutoff', 'n', 'Reference schedule statistic', 'Count'), statistics)
            +table(('Algorithm', 'Cutoff', 'n', 'Movement category', 'Scalar reads', 'Scalar writes', 'Word reads',
                    'Word writes', 'Unknown payload accesses'), traffic_rows)+'</details>')
    return ''.join(parts)


def resource_scaling_sections(lines):
    """Compare observed operand/memory costs and separately labelled projections.

    Older JSON omits these fields and simply has no additional sections. No
    measurement is backfilled from a model. The charts share the accessible
    palette, markers, labelled axes and legend isolation of the original report.
    """
    parts, rows = [], []
    baseline_cutoff = min(line['cutoff'] for line in lines)
    baseline = [line for line in lines if line['cutoff'] == baseline_cutoff]
    for category, label in (('field_additions', 'Measured additions'), ('field_subtractions', 'Measured subtractions')):
        parts.append(panel(label, 'Each point counts calls to the addition or subtraction operation while the code was running. This is useful because saving products often requires extra sums. A recipe may also express a subtraction by multiplying by a negative coefficient and then adding; our counters record the operations that actually ran, so that form appears as a coefficient product and an addition.',
                           curve_chart(label, measured_series(baseline, category), y_label=label+' (count)')))
    for metric, heading, unit in (
            ('equivalent_word_multiplications', '64-bit multiplication equivalents', '64-bit word products (model)'),
            ('equivalent_word_additions_subtractions', '64-bit addition and subtraction equivalents', '64-bit word sums (model)'),
            ('observed_max_operand_bits', 'Observed operand growth', 'Operand magnitude (bits)')):
        curves = []
        for index, line in enumerate(baseline):
            points = [(s['size'], s['observation']['operand_costs'][metric]) for s in line['samples']
                      if s['status'] == 'passed' and 'operand_costs' in s['observation']]
            curves.append({'label': line['algorithm']+f", cutoff {line['cutoff']}", 'color': index, 'points': points})
            bounds = line.get('resource_bounds', {})
            if bounds.get('status') == 'bounded':
                bound_key = {'equivalent_word_multiplications': 'word_multiplications_upper_bound',
                             'equivalent_word_additions_subtractions': 'word_additions_subtractions_upper_bound',
                             'observed_max_operand_bits': 'operand_bits_upper_bound'}[metric]
                curves.append({'label': line['algorithm']+f" · proved upper bound, |entry| ≤ {bounds['input_absolute_bound']}",
                    'color': index, 'kind': 'upper_bound', 'range_limits': bounds['range_limits'],
                    'points': [(p['size'], p[bound_key]) for p in bounds['points']]})
        if any(curve['points'] for curve in curves):
            parts.append(panel(heading, 'We inspect the exact numbers used by each scalar operation and record how many bits they need. These original word-cost graphs retain their earlier definition for comparison with saved evidence and proved bounds. Multiplying two 128-bit integers means two chunks times two chunks, or four word-pair products. Fractions use unreduced numerator and denominator expressions. This original model excludes GCD, cancellation, division and carry work; the separate mathematical work score above includes a declared schedule for those operations. Neither model counts CPU instructions or Python’s internal big-integer work.',
                curve_chart(heading, curves, logarithmic=True, y_label=unit)))
    for line in lines:
        for sample in line['samples']:
            cost = sample['observation'].get('operand_costs')
            if cost:
                rows.append((line['algorithm'], line['cutoff'], sample['size'],
                    cost['observed_max_operand_bits'], cost['observed_max_result_bits'], cost['observed_max_denominator_bits'],
                    cost['equivalent_word_multiplications'], cost['word_multiplications_by_role'].get('data', 0),
                    cost['word_multiplications_by_role'].get('coefficient', 0), cost['status']))
    if rows:
        parts.append(panel('Operand-size and word-model evidence', 'This table shows the largest values seen during each call, including the numerator and denominator sizes of fractions. We keep those values exact; recording their bit widths does not round them or convert them to 64-bit numbers. A zero uses one magnitude bit in our storage convention, and sign-handling work is not estimated. The data and coefficient columns are parts of the “All word products” total, so do not add them to that total again. “Partial” means some operations, such as an exact division used for normalization, do not yet have a word-cost model.',
                           table(('Algorithm', 'Cutoff', 'n', 'Max operand bits', 'Max result bits', 'Max denominator bits',
                                  'All word products', 'Data word products', 'Coefficient word products', 'Model status'), rows)))
    bound_rows = []
    for line in baseline:
        bounds = line.get('resource_bounds', {})
        if bounds.get('status') != 'bounded':
            continue
        lookup = {p['size']: p for p in bounds['points']}
        for sample in line['samples']:
            if (sample.get('resource_bounds_check_passed') is not True
                    or 'operand_costs' not in sample.get('observation', {})
                    or sample['size'] not in lookup):
                continue
            observed, upper = sample['observation']['operand_costs'], lookup[sample['size']]
            bound_rows.append((line['algorithm'], sample['size'], observed['observed_max_operand_bits'],
                upper['operand_bits_upper_bound'], observed['equivalent_word_multiplications'],
                upper['word_multiplications_upper_bound'], observed['equivalent_word_additions_subtractions'],
                upper['word_additions_subtractions_upper_bound']))
    if bound_rows:
        parts.append(panel('How conservative are the resource bounds?',
            'Compare each captured execution with its bound at the same dimension. The bound covers any integer inputs with entry magnitudes at most 4, using this fixed recipe and cutoff. Triangle inequalities allow every combination to grow without cancellation; reduced denominators divide exact common-denominator bounds. A uniform enclosure covers all recursive children. These choices can make the bound substantially larger than an actual observation, especially for fractional recipes. A large bound does not prove that actual operands or work grow that quickly. Each saved point is justified independently; connecting segments are not guarantees at other dimensions. The scalar-call schedule and every measured width and word-work total are checked against the model. Python allocation peaks, time, GCD, division, carry and allocation work are excluded. This table uses the smallest cutoff, matching the resource charts.',
            table(('Algorithm', 'n', 'Observed operand bits', 'Operand bit bound', 'Observed word products',
                   'Word product bound', 'Observed word sums', 'Word sum bound'), bound_rows)))
    for line in lines:
        projection = line.get('scaling_projection', {})
        if projection.get('status') != 'projected':
            continue
        # Each category uses an exact integer recurrence. Dashed curves and the
        # explicit legend label prevent these huge, unexecuted sizes appearing as
        # empirical measurements. Only display uses approximate log coordinates.
        curves = [{'label': label+' · projection', 'color': index, 'kind': 'ideal',
                   'points': [(p['size'], p['counts'][category]) for p in projection['points']]}
                  for index, (category, label) in enumerate((
                      ('field_multiplications', 'Data multiplications'), ('field_additions', 'Additions'),
                      ('field_subtractions', 'Subtractions'), ('coefficient_multiplications', 'Coefficient products'),
                      ('field_normalizations', 'Normalizations')))]
        parts.append('<details><summary>'+text(f"{line['algorithm']}, cutoff {line['cutoff']} · beyond measured sizes")+'</summary>'
            +panel('Exact recurrence projections', 'These matrices were not executed. Instead, we evaluate the recipe’s counting rule using exact integers: how many child products each stage requests, how many entries its combinations touch, and where the fixed cutoff stops recursion. This lets us inspect counts for sizes too large to run here. The recurrence printed below explains how the construction continues beyond the graph; a finite graph does not itself prove the exponent. We do not extend the measured operand widths or memory peaks to these sizes, because they need further input- and representation-dependent models.',
                   '<p><code>'+text(projection['recurrence'])+'</code></p>'
                   +curve_chart('Projected arithmetic categories', curves, logarithmic=True, y_label='Projected field calls (count)')
                   +table(('n', 'Data multiplications', 'Additions', 'Subtractions', 'Coefficient products', 'Normalizations',
                           'Hypothetical 64-bit input/output bytes'),
                          [(p['size'], *(p['counts'][key] for key in ('field_multiplications', 'field_additions',
                           'field_subtractions', 'coefficient_multiplications', 'field_normalizations')),
                           p['input_output_64bit_baseline_bytes']) for p in projection['points']])
                   +'<p>For a simple storage comparison, the last column imagines keeping the two inputs and one output as three n×n arrays of 64-bit numbers. Our exact integers and fractions can take more space, and an algorithm may need temporary arrays too. That column is therefore neither Fraction storage nor an algorithm’s peak memory. Preparing a fixed recipe is reported separately from applying it. Asking the generator to search harder or build a higher tensor power is another experiment with its own parameters.</p>')+'</details>')
    return ''.join(parts)


def render_work_sweep(data, *, title, source_name, source_hash, generated):
    """Show measured count growth, declared ideal curves, and padding/cutoff effects."""
    lines = data["series"]
    configuration = data["configuration"]
    names = list(dict.fromkeys(line["algorithm"] for line in lines))
    samples = [sample for line in lines for sample in line["samples"]]
    passed = sum(sample["status"] == "passed" for sample in samples)
    matches = sum(sample.get("prediction_matches_observed_data_multiplications") is True for sample in samples)
    step_matches = sum(sample.get("prediction_matches_observed_arithmetic_steps") is True for sample in samples)
    executed = sum(sample.get("measurement", {}).get("kind", "executed") == "executed" for sample in samples)
    reused = sum(sample.get("measurement", {}).get("kind") == "reused" for sample in samples)
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>{text(title)}</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">Measured arithmetic growth</div>',
             f'<h1>{text(title)}</h1><p>This report runs matrix-multiplication recipes at several sizes and counts the work they perform. Changing the classical leaf cutoff lets us explore how far to continue splitting before using direct row-by-column multiplication. Wall-clock time is not plotted in these growth comparisons; the main question is how the arithmetic work changes.</p>',
             f'<p class="note">Generated {text(generated)} · Sizes {text(configuration.get("sizes"))} · Cutoffs {text(configuration.get("cutoffs"))} · Seed {text(configuration.get("seed"))}</p>',
             f'<div class="summary"><span class="badge">{len(samples)} requested configurations</span><span class="badge">{executed} actual executions</span><span class="badge">{reused} reused measurements</span><span class="badge">{passed} configurations verified</span><span class="badge">{matches} multiplication budgets match the recurrence</span><span class="badge">{step_matches} full arithmetic budgets match</span></div></header>',
             '<p class="note">Sometimes two cutoff settings lead to exactly the same calculation. For example, a radix-four recipe may reach 1×1 leaves at both cutoff 1 and cutoff 2. When we know that the padding, stage sequence and leaf dimensions are identical, we run that calculation once and copy its verified evidence into the equivalent rows. The Measurement source column identifies which cutoff actually ran. Every requested setting still has its own model and graph; a copied row is not presented as a second independent execution.</p>',
             reading_guide(), terminology_guide(), implementation_comparison_sections(lines),
             '<p class="callout">The theoretical exponent comes from the recipe and its justified counting rule. The measured slopes below describe only the finite range we executed; they are not proofs of an asymptotic exponent. A log-log graph makes a pure power law appear straight, which helps compare growth rates. Padding, leaf choices and extra additions can make the observed points depart from that straight line.</p>',
             panel("What does cutoff mean?", "The cutoff is the dimension threshold for stopping recursion and using ordinary row-by-column multiplication. A stage that cannot form nonempty blocks also stops.",
                   '<p>For Strassen on a 16×16 matrix, cutoff 4 splits into 8×8 blocks, then 4×4 blocks, and multiplies those leaves classically. Cutoff 1 continues down to individual numbers. More recursion can save data multiplications while adding combination work.</p><p>The naive algorithm never recurses, so its cutoff does nothing and its curves overlap. For AlphaEvolve at size 16, cutoff 2 still produces 1×1 leaves. At unpadded size 8, the size-2 child blocks can be classical leaves. An empty divisible core can produce a leaf larger than the cutoff. The exact-count table records the actual leaf size.</p>')]
    parts.append(panel('Why can the graphs have steps?',
        'Recursion alone does not force a staircase. The exponent describes asymptotic growth, not an exact smooth cost at every integer size.',
        table(('Source', 'What is inherent?', 'What is our choice?', 'Platform necessity'), (
            ('Recursive recipe', 'Each stage prescribes integer equal-block divisions and a number of child products.',
             'How arbitrary sizes are adapted to those stages.', 'The formula is mathematical; the adaptation is portable.'),
            ('Zero padding', 'Each formula needs compatible core dimensions before a stage executes.',
             'Default: round the core down and finish the fringe explicitly. Optional baseline: round the whole request up.',
             'Required by this traversal policy, not universally required by the algorithm or hardware.'),
            ('Classical cutoff', 'Recursion must have a terminating base case.',
             'Default: test actual dimensions before splitting. The padded baseline tests enlarged dimensions; a fixed cutoff preserves the family exponent.',
             'A configurable interpreter choice, not a hardware constraint.'),
            ('Smooth theory curve', 'The recurrence and asymptotic bound apply under their stated assumptions.',
             'Draw an unpadded continuous comparison curve; it is not an executable matrix at fractional sizes.',
             'A presentation model; discrete predictions describe the actual policy.'),
            ('64-bit word-cost model', 'Exact operands can grow or shrink as arithmetic and cancellation change their values.',
             'Round observed magnitude widths up to 64-bit words; crossing a word boundary changes the modelled word count.',
             'A chosen cost unit, not a claim about native CPU/GPU limbs or an added field multiplication.'),
            ('Memory and operand measurements', 'Exact representation needs enough space for the values actually produced.',
             'Measure Python-traced allocation peaks, including observer overhead; record observed widths.',
             'Memory can depend on interpreter/backend allocation behavior. This report does not attribute each memory jump to padding.'),
            ('Sampled graph', 'Captured operation counts are integers.',
             'Connect only requested sizes with straight guides; sparse sampling can hide boundaries.',
             'A plotting choice, not extra algorithm work.')))
        +'<p>For Strassen at cutoff one, the historical padded baseline makes n=5 through n=8 all use 343 data products, '
         'and n=9 use 2,401. The default fringe policy instead uses 110 at n=5: 49 for the 4×4 core plus '
         '5³−4³=61 remaining dot terms. It uses 343 at n=8 and 560 at n=9. The equal-halves formula is inherent; '
         'rounding 9 all the way to 16 was our size-handling choice. Core and stage changes can still cause '
         'finite jumps or dips; removing padding does not assert a smooth or monotone cost curve.</p>'
        '<p>Inspect “Where this algorithm’s steps come from” beside each comparison, when size metadata is available, for the actual stage and leaf path. '
        'A plateau in field counts need not be a plateau in bit costs or memory. These policies and their alternatives '
        'preserve exact arithmetic; no rounding of numerical values is involved.</p>'))
    models = {line["algorithm"]: line["mathematical_model"] for line in lines}
    schedules = [(name, " → ".join(model["schedule"]), model["base_dimension"],
                  model["base_bilinear_rank"], model["arithmetic_domain"])
                 for name, model in models.items() if model.get("schedule")]
    if schedules:
        parts.append(panel("Composition schedules", "A schedule lets different recipes take turns as recursion reaches smaller problems. One cycle means following every stage in that schedule once, after which the same order begins again. The worked example below explains how their size reductions and product counts combine.",
                           composition_guide()+table(("Report name", "Recipes used in order", "Size reduction after a full cycle", "Descendant products after a full cycle", "Kinds of numbers supported"), schedules)))
    parts.append(panel("Square arithmetic exponents · theoretical",
                       "Each algorithm has a growth exponent p calculated from its justified recurrence. Roughly, its leading arithmetic grows like n raised to p as square matrices become large. A smaller exponent can eventually save work, even if its recipe has expensive constants at small sizes. Each justified algorithm provides an upper bound on omega (ω), the infimum over matrix algorithms in the stated arithmetic setting. The values here are not omega itself and are not measured slopes. Keeping the leaf cutoff fixed preserves each family exponent; changing it with matrix size would change the assumptions.",
                       bar_chart("Square arithmetic exponents", [(name, model.get("square_arithmetic_exponent")) for name, model in models.items()], "exponent", maximum=3)))
    baseline_cutoff = min(line["cutoff"] for line in lines)
    baseline = [line for line in lines if line["cutoff"] == baseline_cutoff]
    for metric, heading in (("field_arithmetic_operations", "Measured field arithmetic steps"),
                            ("atomic_arithmetic_operations", "Measured component / atomic steps")):
        curves = measured_series(baseline, metric)
        body = curve_chart(heading, curves, y_label=metric_axis_label(metric))
        body += '<details><summary>Log-log view</summary>' + curve_chart(heading, curves, logarithmic=True, y_label=metric_axis_label(metric)) + '</details>'
        explanation = (f"These runs use cutoff {baseline_cutoff}. A field step acts on one mathematical number: a data product, multiplication by a recipe coefficient, addition, subtraction, or normalization that removes a known scale factor. This total shows work that the data-product count alone would miss. A complex field step may need several real scalar calls, so comparisons across real and complex recipes need the component graph too."
                       if metric == "field_arithmetic_operations" else
                       f"These runs use cutoff {baseline_cutoff}. We count the smaller scalar calls exposed by the implementation, including real arithmetic and exact integer quotient/remainder operations. Exact complex pairs expose their real components; a native Python complex call stays opaque because we do not observe its internal instructions. The field and component graphs describe the same calculation at different levels, so do not add the two totals.")
        parts.append(panel(heading, explanation, body))
    for name in names:
        group = [line for line in lines if line["algorithm"] == name]
        curves = comparison_series(group, "field_multiplications")
        model = group[0]["mathematical_model"]
        body = f'<p>Theoretical family exponent: <strong>{text(display(model["square_arithmetic_exponent"]))}</strong> ({text(model["exponent_expression"])}). {text(model.get("parameter_source", "Declared scheme parameters"))}.</p>'
        body += curve_chart(f"{name} data multiplications", curves, y_label=metric_axis_label("field_multiplications"))
        body += '<details><summary>Log-log view of multiplications</summary>' + curve_chart(f"{name} data multiplications", curves, logarithmic=True, y_label=metric_axis_label("field_multiplications")) + '</details>'
        body += size_effects_section(group)
        body += '<details><summary>All arithmetic steps for these block sizes</summary>'
        body += '<p>The product count is only one part of the bill. At each level, we also form input combinations and assemble outputs. Those operations touch block entries, and some recipes multiply them by fixed coefficients or remove a common scale factor. The total-work model adds these costs at every level. Its smooth curve is therefore more informative than drawing a rescaled n<sup>p</sup> line. The constants come from our explicit operation schedule, rather than being adjusted to make the graph fit the measurements.</p>'
        for metric, heading in (("field_arithmetic_operations", "Field steps"), ("atomic_arithmetic_operations", "Component / atomic steps")):
            step_curves = comparison_series(group, metric)
            body += f'<h3>{heading}</h3>' + curve_chart(heading, step_curves, y_label=metric_axis_label(metric))
            body += '<details><summary>Log-log view of arithmetic steps</summary>' + curve_chart(heading, step_curves, logarithmic=True, y_label=metric_axis_label(metric)) + '</details>'
        body += '</details>'
        rows = []
        for line in group:
            fits = line.get("empirical_fits", {})
            rows.append((line["cutoff"], model["square_arithmetic_exponent"],
                         fits.get("field_multiplications", {}).get("slope"),
                         fits.get("field_arithmetic_operations", {}).get("slope"),
                         fits.get("atomic_arithmetic_operations", {}).get("slope")))
        body += table(("Leaf cutoff", "Theoretical exponent", "Measured multiplication slope", "Measured field-step slope", "Measured component-step slope"), rows)
        body += '<p class="note">A fitted slope summarizes how quickly the successful measured counts grow across the sizes in this experiment. It can be misleading if several sizes share one padded problem, or if combination work is still a large part of the cost. Multiplication slopes and total-step slopes answer different questions for that reason. No finite fitted slope establishes a new mathematical bound; the recurrence must justify the long-term exponent separately.</p>'
        if model.get("schedule"):
            body += '<p class="note">This recipe repeats several stages, so stopping at different points in that order can give different finite costs. We show exact discrete predictions at each sampled size. A smooth ideal curve is omitted rather than pretending one uniform stage describes all those phases. The dimension and stage paths above show which phase each calculation reaches.</p>'
        if group[0]["work_model"].get("unavailable_reason"):
            body += '<p class="callout">Coefficient setup unavailable: ' + text(group[0]["work_model"]["unavailable_reason"]) + '</p>'
        parts.append(panel(f"{name} · measured versus theoretical multiplications", "Solid points are the counts captured while this algorithm ran; the lines between them are visual guides, not measurements of every intervening size. Dashed curves, when available, show a continuous comparison model, separate from the actual discrete traversal. Crosses show what the discrete counting rule predicts after applying the actual size policy and cutoff. When a cross lands on a measured point, that is evidence that execution followed the model. The size-path table explains the difference between the requested problem and the one we actually executed.", body))
    rows = []
    for line in lines:
        for sample in line["samples"]:
            observation, predicted = sample["observation"], sample["prediction"]
            counts = observation["counts"]
            padded_label = (" × ".join(str(x) for x in predicted["padded_dimensions"])
                            if "padded_dimensions" in predicted else predicted.get("padded_dimension"))
            rows.append((line["algorithm"], line["cutoff"], sample["size"], padded_label,
                         predicted.get("effective_leaf_dimension"), counts.get("field_multiplications", 0),
                         predicted.get("padded_data_multiplications"), counts.get("coefficient_multiplications", 0),
                         observation.get("field_arithmetic_operations"), observation.get("atomic_arithmetic_operations"), sample["status"],
                         measurement_label(sample, line["cutoff"])))
    parts.append(panel("Exact sampled counts and effective blocks", "Use this table when you want the exact values behind the graph. Data multiplications act on input-dependent values; coefficient multiplications use the fixed numbers in the recipe. Field steps and component steps overlap, so they must not be added. The cutoff is a maximum leaf size, not a promise to reach a block of exactly that size: the stage’s integer divisions determine which leaf sizes are available.",
                       table(("Algorithm", "Cutoff", "n", "Executed dimensions", "Actual leaf n", "Data multiplications", "Execution prediction", "Coefficient multiplications", "Field steps", "Component steps", "Output check", "Measurement source"), rows)))
    parts.append(sweep_evidence(data))
    parts.append(resource_scaling_sections(lines))
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
    parts.append(range_chart_script())
    parts.append('<footer>Observed work excludes input generation, the exact oracle, report calculations, and measurement bookkeeping. Counts assume unit-cost scalar operations; integer bit complexity is separate.</footer></main></body></html>')
    return "".join(parts)


def sweep_sample_report(line, sample):
    """Adapt captured sweep evidence to the shared single-experiment view.

    Shapes and classical baseline follow the sweep's square input contract.
    Accuracy, primitive counters and optional per-call resources are captured
    evidence. No absent measurement is invented; older fields stay unavailable.
    """
    size = sample["size"]
    budget = sample.get("prediction", {}).get("composition_budget")
    extra = {}
    if budget:
        extra = {"composition_budget": budget, "composition_budget_matches": {
            "field_multiplications": sample.get("prediction_matches_observed_data_multiplications"),
            **{key: sample["observation"].get(key) == budget[key]
               for key in ("field_arithmetic_operations", "atomic_arithmetic_operations")}}}
    return {**sample["observation"], **extra, "status": sample["status"], "error": sample.get("error"),
            "measurement": sample.get("measurement", {}),
            "mathematical_model": line["mathematical_model"],
            "experiment": {"input_shapes": [[size, size], [size, size]], "cutoff": line["cutoff"],
                           "input_repr_sha256": sample.get("input_repr_sha256"),
                           "classical_data_product_baseline": size ** 3, "primitive_timing_enabled": False},
            "execution_measurements": {**sample.get('execution_measurements', {}), "accuracy": sample.get("verification", {})}}


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
                             json.dumps(check.get("failures", [])), sample.get("error"),
                             measurement_label(sample, line["cutoff"])))
    parts = [panel("Arithmetic work breakdown",
                   "Expand the table to inspect all the operations for every requested size and cutoff. A smaller data-product count may come with many more coefficient products or sums, so those categories are kept separate. Normalization means removing a known scale factor from the result. The field and component totals describe the same calculation at different levels and must not be added together.",
                   '<details><summary>Full arithmetic breakdown across sizes and cutoffs</summary>' +
                   table(("Algorithm", "Cutoff", "n", "Data multiplications", "Coefficient multiplications", "Additions", "Subtractions", "Normalizations", "Field steps", "Atomic steps"), breakdown) + '</details>'),
             panel("Accuracy and verification evidence", "After an algorithm returns its answer, an independently written row-by-column calculation checks it. The checker’s work is outside the algorithm’s observed arithmetic. Integer and fraction results must agree exactly; separately labelled floating-point diagnostics use a tolerance. Equivalent-cutoff rows identify the original execution whose evidence they reuse. The primitive clocks are disabled in this size sweep, while optional per-call allocation probes have their own measurement boundaries.",
                   '<details><summary>Accuracy for every executed sample</summary>' +
                   table(("Algorithm", "Cutoff", "n", "Status", "Verification mode", "Maximum absolute error", "Failed entries", "Error", "Measurement source"), accuracy) + '</details>'),
             '<section class="panel"><h2>Per-sample primitive and recursion evidence</h2><p class="note">Open an algorithm, then a matrix size, to see the building blocks that ran. Primitive clocks were disabled in this sweep, so zero profile seconds mean we did not time those calls—not that they took no time. Call counts and arithmetic totals were still recorded. A parent primitive includes work done by its children; those inclusive profiles overlap and must not be summed.</p>']
    for line in data["series"]:
        parts.append(f'<details><summary>{text(line["algorithm"])} · cutoff {text(line["cutoff"])} · {len(line["samples"])} samples</summary>')
        for sample in line["samples"]:
            report = sweep_sample_report(line, sample)
            parts.append(f'<details><summary>n={text(sample["size"])} · {text(sample["status"])}</summary>')
            depths = sorted(report.get("products_by_depth", {}).items(), key=lambda item: int(item[0]))
            parts.append('<h3>Data products by recursion depth</h3>' + bar_chart("Data products by recursion depth",
                         [(f"depth {depth}", count) for depth, count in depths], "field products", category_label="Recursion depth"))
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


def work_and_traffic_evidence(reports):
    """Expose the same model definitions and exact details in single-pair reports."""
    work, traffic, rows, statistics = [], [], [], []
    for name, report in reports.items():
        cost = report.get('mathematical_work')
        if cost is None:
            continue
        movement = report.get('data_movement', {})
        comparison = report.get('work_comparison', {})
        work.append((name, comparison_value(comparison)))
        traffic.append((name, movement.get('logical_word_transfers')))
        ratio = comparison.get('relative_efficiency')
        rows.append((name, report.get('status'), cost['status'], cost.get('total_work_units'),
                     comparison.get('baseline_work_units'),
                     f"{ratio['numerator']}/{ratio['denominator']}" if ratio else None,
                     movement.get('status'), movement.get('logical_word_transfers'), str(cost['unmodelled_calls'])))
        for key, value in cost.get('schedule_statistics', {}).items():
            statistics.append((name, key, value))
    if not rows:
        return ''
    return (panel('Mathematical work efficiency', mathematical_work_explanation(),
                  bar_chart('Mathematical work efficiency', work, 'classical work / candidate work'))
            +panel('Logical data movement', traffic_explanation(),
                   bar_chart('Logical word transfers', traffic, 'logical 64-bit word reads + writes'))
            +'<details><summary>Exact work and movement evidence</summary>'+table(
                ('Algorithm', 'Execution', 'Work coverage', 'Candidate work units', 'Classical work units',
                 'Exact relative efficiency', 'Movement coverage', 'Logical transfers', 'Unmodelled calls'), rows)
            +table(('Algorithm', 'Reference schedule statistic', 'Count'), statistics)+'</details>')


def operand_evidence(reports):
    """Expose word costs in individual-pair reports without inventing older data."""
    rows, primitive_rows, products = [], [], []
    for name, report in reports.items():
        cost = report.get('operand_costs')
        if not cost:
            continue
        rows.append((name, cost['status'], cost['observed_max_operand_bits'], cost['observed_max_result_bits'],
                     cost['observed_max_denominator_bits'], cost['equivalent_word_multiplications'],
                     cost['equivalent_word_additions_subtractions'], str(cost['unmodelled_calls'])))
        products.append((name, cost['equivalent_word_multiplications'] if cost['modelled_atomic_operations'] else None))
        for primitive, values in cost.get('primitive_word_costs', {}).items():
            primitive_rows.append((name, primitive, values['calls'], values['word_multiplications'], values['word_additions_subtractions']))
    if not rows:
        return ''
    return panel('Exact operand sizes and 64-bit word equivalents',
        'Our numbers stay exact while we record their bit widths. This original model breaks integers into 64-bit chunks and counts chunk pairs. Fractions use unreduced numerator and denominator expressions. GCD, cancellation, division and carry propagation remain excluded from this original estimate; the separate mathematical work model includes its own explicit schedule for these costs. Keeping both preserves earlier evidence and resource-bound definitions. These are not CPU instruction counts. Floating diagnostics remain unmodelled rather than being treated as exact integers.',
        bar_chart('64-bit multiplication equivalents', products, 'modelled word products')
        +table(('Algorithm', 'Model status', 'Max operand bits', 'Max result bits', 'Max denominator bits',
                'Word products', 'Word sums', 'Unmodelled calls'), rows)
        +'<details><summary>Word-cost estimates by atomic primitive</summary>'
        +table(('Algorithm', 'Primitive', 'Modelled calls', 'Word products', 'Word sums'), primitive_rows)+'</details>')


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
    else:
        parts.extend([reading_guide(), terminology_guide()])
    parts.extend([f'<p class="callout">{text(comparison_warning(reports))}</p>', '<div class="grid">',
             panel("Square arithmetic exponents · theoretical", "These values describe the algorithm families, not the time or growth of this single input pair. For a repeating square recipe, we calculate the exponent from the number of descendant products and the factor by which their dimensions shrink. Ordinary row-by-column multiplication has exponent three. A successful small run checks an implementation, but cannot determine its asymptotic exponent. If the source data has no mathematical model, we leave that value unavailable.",
                   bar_chart("Square arithmetic exponents", exponents, "exponent", maximum=3)),
             panel("Observed data multiplications", "This is the number of input-dependent field products counted while each algorithm ran on this input pair. It includes work on any padded zeros and the direct products at classical leaves. Fixed-coefficient products appear in the breakdown below. A complex field product may need several real scalar operations, so this chart alone does not compare all the work of real and complex recipes.",
                   bar_chart("Observed data products", products, "field products")), "</div>"])
    rows = []
    for name, report in reports.items():
        rows.append((name, arithmetic_count(report, "field_multiplications"), arithmetic_count(report, "coefficient_multiplications"),
                     arithmetic_count(report, "field_additions"), arithmetic_count(report, "field_subtractions"), arithmetic_count(report, "field_normalizations"),
                     report.get("atomic_arithmetic_operations")))
    parts.append(panel("Arithmetic work breakdown", "The table shows the work that a data-product count alone would hide: coefficient products, sums, differences and normalization. Field and atomic counters are overlapping descriptions of the same work. Do not sum those layers. In new reports, an omitted arithmetic category means no calls were observed only when the report explicitly declares that convention. Older reports may simply lack the counter; those missing values remain unavailable instead of being presented as zero.",
                       table(("Algorithm", "Data multiplications", "Coefficient multiplications", "Additions", "Subtractions", "Normalizations", "Atomic steps"), rows)))
    depth_entries = [(f"{name} · depth {depth}", value) for name, report in reports.items()
                     for depth, value in sorted(report.get("products_by_depth", {}).items(), key=lambda item: str(item[0]).zfill(10))]
    if depth_entries:
        parts.append(panel("Data products by recursion depth", "Depth zero is the original problem, depth one is its child problems, and each further split increases the depth. This chart counts the field products that actually execute, usually in the leaves where splitting stops. It does not count a larger block-product request again as if it were an extra scalar product. Padding and the cutoff determine how deep this particular run goes.",
                           bar_chart("Data products by recursion depth", depth_entries, "field products", category_label="Algorithm and recursion depth")))
    parts.append('<div class="grid">')
    runtime = [(name, report.get("execution_measurements", {}).get("elapsed_algorithm_call_seconds")) for name, report in reports.items()]
    memory = []
    for name, report in reports.items():
        metric = report.get("execution_measurements", {}).get("python_traced_peak_bytes", {})
        memory.append((name, metric.get("value") if metric.get("status") == "measured" else None))
    parts.append(panel("Secondary · elapsed time", "This clock surrounds the public Python multiplication call. Its reading includes input checks, padding, any recipe preparation done inside the call and our observation machinery. The independent answer checker runs afterward, outside this clock. Cache state and measurement overhead can affect the result, so a single observed run is not a production benchmark or a measure of an optimized GPU implementation.",
                       bar_chart("Elapsed algorithm call time", runtime, "seconds")))
    parts.append(panel("Secondary · Python allocation peak", "We record the highest Python-traced allocation total during the call. The inputs already exist when tracking starts, while temporary arrays, numbers, setup and observation may create new allocations. Native libraries and GPU storage may be invisible to this probe. Treat the value as a measurement with those boundaries, not the total memory of the process or the exact storage required by the mathematical algorithm.",
                       bar_chart("Traced Python allocation peak", memory, "bytes")))
    parts.append("</div>")
    parts.append(work_and_traffic_evidence(reports))
    parts.append(operand_evidence(reports))
    diagnostics = []
    for name, report in reports.items():
        execution = report.get("execution_measurements", {})
        accuracy = execution.get("accuracy", {})
        memory_status = execution.get("python_traced_peak_bytes", {}).get("status", "not_recorded")
        diagnostics.append((name, accuracy.get("mode"), accuracy.get("maximum_absolute_error"), memory_status,
                            execution.get("classical_equivalent_field_operations_per_second")))
    parts.append(panel("Accuracy and diagnostic scope", "The answer checker independently compares the returned matrix with ordinary row-by-column multiplication. Exact results require equality; floating diagnostics report their error and tolerance. The equivalent rate divides the conventional classical work estimate, 2*M*N*K, by elapsed seconds. It is a common comparison unit, not a count of this algorithm’s actual work or the chip’s floating-point instructions. Missing measurements and nonfinite errors stay explicit rather than being converted into reassuring zeros.",
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
