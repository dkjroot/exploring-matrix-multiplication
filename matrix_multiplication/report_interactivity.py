"""Offline line selection and range controls, with display-only coordinates.

Python prepares all authoritative counts as arbitrary-precision integers. Browser
code only selects points, formats their decimal strings and draws SVG coordinates.
No measured result, arithmetic budget, fit or resource estimate is recalculated.
"""


def range_chart_script():
    """Return fixed browser code; all report data lives in escaped HTML attributes."""
    return r'''<script>
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  /** Create SVG nodes through the DOM so labels are always inert text. */
  function node(tag, attrs = {}, content = null) {
    const result = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) result.setAttribute(key, String(value));
    if (content !== null) result.textContent = String(content);
    return result;
  }
  /** Preserve exact integer tooltip strings, including counts beyond 2^53. */
  function exact(value) {
    return /^\d+$/.test(value) ? BigInt(value).toLocaleString('en-GB') : value;
  }
  /** Axis labels are explicitly rounded secondary display coordinates. */
  function tick(value) {
    return value >= 1e6 ? value.toExponential(2) : value.toLocaleString('en-GB', {maximumSignificantDigits: 5});
  }
  /** Match the static chart's circle, square, triangle and diamond markers. */
  function marker(x, y, index, color) {
    const attrs = {fill: index % 8 >= 4 ? 'white' : color, stroke: color, 'stroke-width': 1.5};
    const radius = 4.5;
    if (index % 4 === 0) return node('circle', {...attrs, cx: x, cy: y, r: radius});
    if (index % 4 === 1) return node('rect', {...attrs, x: x-radius, y: y-radius, width: 2*radius, height: 2*radius});
    const points = index % 4 === 2
      ? [[x,y-radius-1],[x+radius+1,y+radius],[x-radius-1,y+radius]]
      : [[x,y-radius-1],[x+radius+1,y],[x,y+radius+1],[x-radius-1,y]];
    return node('polygon', {...attrs, points: points.map(p => p.join(',')).join(' ')});
  }
  /** Reapply per-line choices whenever a slider replaces its SVG. */
  function applySelection(chart) {
    const buttons = [...chart.querySelectorAll('.legend-button[data-series-id]')];
    const enabled = new Set(buttons.filter(b => b.getAttribute('aria-pressed') === 'true')
      .map(b => b.getAttribute('data-series-id')));
    for (const line of chart.querySelectorAll('.chart-series[data-series-id]')) {
      line.setAttribute('data-hidden', String(!enabled.has(line.getAttribute('data-series-id'))));
    }
    chart.querySelector('[data-series-status]').textContent = enabled.size+' of '+buttons.length+' lines enabled';
  }
  for (const chart of document.querySelectorAll('.work-chart')) {
    const buttons = [...chart.querySelectorAll('.legend-button[data-series-id]')];
    for (const button of buttons) {
      button.addEventListener('click', () => {
        button.setAttribute('aria-pressed', String(button.getAttribute('aria-pressed') !== 'true'));
        applySelection(chart);
      });
    }
    for (const control of chart.querySelectorAll('[data-series-select]')) {
      control.addEventListener('click', () => {
        const enabled = String(control.getAttribute('data-series-select') === 'all');
        for (const button of buttons) button.setAttribute('aria-pressed', enabled);
        applySelection(chart);
      });
    }
  }
  for (const chart of document.querySelectorAll('[data-range-chart]')) {
    const data = JSON.parse(chart.getAttribute('data-range-chart'));
    const slider = chart.querySelector('input[type="range"]');
    const output = chart.querySelector('output');
    const original = chart.querySelector(':scope > svg').cloneNode(true);
    /** Draw selected points with new axes, keeping measured data immutable. */
    function redraw() {
      const position = Number(slider.value);
      const limitText = data.limits[position];
      const maximum = Number(limitText);
      const sampled = data.limits[0];
      slider.setAttribute('aria-valuetext', 'Maximum dimension '+exact(limitText));
      output.textContent = position === 0 ? 'Sampled range through n='+exact(sampled)
        : 'Through n='+exact(limitText)+(data.bounded ? ' · upper bounds beyond n=' : ' · projected beyond n=')+exact(sampled);
      if (position === 0) {
        chart.querySelector(':scope > svg').replaceWith(original.cloneNode(true));
        applySelection(chart);
        return;
      }
      const lines = data.series.map(line => ({...line, points: line.points.filter(([x,y]) =>
        Number(x) <= maximum && Number.isFinite(Number(x)) && Number.isFinite(Number(y))
        && (!data.logarithmic || Number(y) > 0))}));
      const points = lines.flatMap(line => line.points);
      if (!points.length) return;
      const transform = data.logarithmic ? Math.log10 : Number;
      const xmin = Math.min(...points.map(([x]) => transform(Number(x))));
      const xmax = Math.max(transform(maximum), xmin+1e-12);
      const ymin = data.logarithmic ? Math.min(...points.map(([,y]) => transform(Number(y)))) : 0;
      const ymax = Math.max(...points.map(([,y]) => transform(Number(y))), ymin+1);
      const left = 120, top = 35, width = 700, height = 260;
      /** Convert a saved point to approximate SVG coordinates only. */
      function location(point) {
        return [left+(transform(Number(point[0]))-xmin)/(xmax-xmin)*width,
                top+height-(transform(Number(point[1]))-ymin)/(ymax-ymin)*height];
      }
      const svg = node('svg', {viewBox: '0 0 900 365', role: 'img', 'aria-label': data.title});
      svg.append(node('title', {}, data.title), node('desc', {},
        data.bounded ? 'Solid markers are measured. Dash-dot lines are proved upper bounds at selected dimensions, not predicted observations. Coordinates are rounded.'
          : 'Solid markers are measured. Dotted lines are exact schedule projections. Coordinates are rounded.'));
      // Tick selected actual integer dimensions rather than invent fractional matrices.
      const dimensions = [...new Set(lines.filter(line => line.kind !== 'ideal')
        .flatMap(line => line.points.map(([x]) => x)))].sort((a,b) => Number(a)-Number(b));
      // Space ticks in screen coordinates. Dense small samples must not crowd
      // the left edge when the slider reveals much larger dimensions.
      const ticks = dimensions.length <= 8 ? dimensions : [...new Set(Array.from({length:8}, (_,i) => {
        const target = xmin+(xmax-xmin)*i/7;
        return dimensions.reduce((best, dimension) =>
          Math.abs(transform(Number(dimension))-target) < Math.abs(transform(Number(best))-target) ? dimension : best);
      }))];
      for (const dimension of ticks) {
        const x = location([dimension, '1'])[0];
        svg.append(node('line', {x1:x, x2:x, y1:top, y2:top+height, stroke:'#e4e8ef'}),
          node('text', {x, y:top+height+23, 'text-anchor':'middle', 'font-size':14}, tick(Number(dimension))));
      }
      for (let i=0; i<5; i++) {
        const value = ymin+(ymax-ymin)*i/4;
        const y = top+height-height*i/4;
        svg.append(node('line', {x1:left, x2:left+width, y1:y, y2:y, stroke:'#e4e8ef'}),
          node('text', {x:left-12, y:y+5, 'text-anchor':'end', 'font-size':14}, tick(data.logarithmic ? 10**value : value)));
      }
      // A visible boundary records where execution stops and projection begins.
      const boundary = location([sampled, '1'])[0];
      svg.append(node('line', {x1:boundary, x2:boundary, y1:top, y2:top+height,
        stroke:'#6554c0', 'stroke-dasharray':'3 5', 'data-measured-boundary':sampled}));
      const suffix = data.logarithmic ? ' (log scale)' : '';
      svg.append(node('text', {class:'chart-axis-label', 'data-axis':'x', x:470, y:350, 'text-anchor':'middle'},
        'Requested square dimension n'+suffix),
        node('text', {class:'chart-axis-label', 'data-axis':'y', transform:'translate(22 165) rotate(-90)', 'text-anchor':'middle'}, data.yLabel+suffix));
      for (const line of lines) {
        const color = data.colors[line.color % data.colors.length];
        const group = node('g', {class:'chart-series', 'data-series-id':line.line_id, 'data-series-group':'g'+line.color, 'data-series-kind':line.kind});
        if (line.kind !== 'prediction' && line.points.length) {
          const attrs = {points:line.points.map(location).map(p => p.join(',')).join(' '),
            fill:'none', stroke:color, 'stroke-width':2.5};
          if (line.kind === 'ideal') attrs['stroke-dasharray'] = '8 6';
          if (line.kind === 'projected') attrs['stroke-dasharray'] = '2 5';
          if (line.kind === 'upper_bound') attrs['stroke-dasharray'] = '10 3 2 3';
          const path = node('polyline', attrs);
          path.append(node('title', {}, line.label));
          group.append(path);
        }
        for (const point of line.points) {
          if (line.kind === 'ideal') continue;
          const [x,y] = location(point);
          const symbol = node('g');
          symbol.append(node('title', {}, line.label+': n='+exact(point[0])+', work='+exact(point[1])));
          if (line.kind === 'observed') symbol.append(marker(x,y,line.color,color));
          else if (line.kind === 'projected' || line.kind === 'upper_bound') symbol.append(node('circle', {cx:x, cy:y, r:3, fill:'white', stroke:color}));
          else symbol.append(node('path', {d:`M${x-7} ${y}h14 M${x} ${y-7}v14`, stroke:color, 'stroke-width':2}));
          group.append(symbol);
        }
        svg.append(group);
      }
      chart.querySelector(':scope > svg').replaceWith(svg);
      applySelection(chart);
    }
    slider.addEventListener('input', redraw);
    chart.querySelector('[data-range-reset]').addEventListener('click', () => { slider.value='0'; redraw(); });
    slider.setAttribute('aria-valuetext', 'Maximum dimension '+exact(data.limits[0]));
  }
})();
</script>'''
