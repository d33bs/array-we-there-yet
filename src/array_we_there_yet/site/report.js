(function () {
  'use strict';

  var DATA = JSON.parse(document.getElementById('report-data').textContent);
  var hidden = new Set(); // layout keys, "backend/layout", that the filter hides
  var state = { logY: true };
  var redraws = []; // functions that draw one figure
  var CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

  // ---------- helpers ----------

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === 'text') node.textContent = attrs[key];
      else if (key === 'html') node.innerHTML = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    (children || []).forEach(function (child) { node.appendChild(child); });
    return node;
  }

  // A black line disappears on a dark page, so black follows the text color.
  function plotColor(color) {
    return color === '#000000' ? css('--text') : color;
  }

  function visible(item) {
    return !hidden.has(item.backend + '/' + item.layout);
  }

  function twoDigits(value) {
    return value >= 100 ? Math.round(value).toLocaleString('en-US') : Number(value.toPrecision(2)).toString();
  }

  function duration(seconds) {
    if (seconds >= 3600) return twoDigits(seconds / 3600) + ' h';
    if (seconds >= 60) return twoDigits(seconds / 60) + ' min';
    return twoDigits(seconds) + ' s';
  }

  function dollars(amount) {
    if (amount < 0.01) return '$' + amount.toFixed(4);
    if (amount < 1) return '$' + amount.toFixed(3);
    if (amount < 10) return '$' + amount.toFixed(2);
    return '$' + Math.round(amount).toLocaleString('en-US');
  }

  function bytesText(gigabytes) {
    if (gigabytes >= 0.1) return twoDigits(gigabytes) + ' GB';
    if (gigabytes >= 0.001) return twoDigits(gigabytes * 1e3) + ' MB';
    return twoDigits(gigabytes * 1e6) + ' KB';
  }

  function baseLayout(title) {
    var text = css('--text');
    var grid = css('--grid');
    return {
      title: { text: title, font: { size: 13, color: text } },
      margin: { l: 58, r: 10, t: 36, b: 46 },
      height: 290,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: text, size: 11 },
      showlegend: false,
      hovermode: 'closest',
      xaxis: { gridcolor: grid, linecolor: grid, zerolinecolor: grid },
      yaxis: { gridcolor: grid, linecolor: grid, zerolinecolor: grid, nticks: 6 }
    };
  }

  function seriesTrace(series, yLabel, options) {
    var trace = {
      type: 'scatter',
      mode: 'lines+markers',
      name: series.label,
      x: series.x,
      y: series.y,
      line: { color: plotColor(series.color), dash: series.dash, width: 2 },
      marker: { color: plotColor(series.color), size: 6 },
      hovertemplate: '<b>' + series.label + '</b><br>%{x:,}<br>' + yLabel + ': %{y:.3g}<extra></extra>'
    };
    if (options && options.dash) trace.line.dash = options.dash;
    if (options && options.name) trace.name = options.name;
    if (series.low && series.high) {
      trace.error_y = {
        type: 'data',
        symmetric: false,
        array: series.high.map(function (high, i) { return Math.max(high - series.y[i], 0); }),
        arrayminus: series.y.map(function (y, i) { return Math.max(y - series.low[i], 0); }),
        thickness: 1,
        width: 3,
        color: plotColor(series.color)
      };
    }
    return trace;
  }

  // Keep the axes tidy when the filter hides every layout in a panel.
  function keepTidy(layout, traces, values) {
    if (values && values.length) {
      var low = Math.log10(Math.min.apply(null, values));
      var high = Math.log10(Math.max.apply(null, values));
      layout.xaxis.type = 'log';
      layout.xaxis.range = [low - 0.08, high + 0.08];
    }
    if (!traces.length) {
      layout.yaxis.type = 'linear';
      layout.yaxis.range = [0, 1];
      layout.yaxis.showticklabels = false;
      layout.annotations = [{
        text: 'No layouts shown', showarrow: false, xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
        font: { color: css('--muted'), size: 12 }
      }];
    }
  }

  function shell(container, draw) {
    container.classList.add('ready');
    redraws.push(draw);
    draw();
  }

  // ---------- facet grids: the three overview views ----------

  function facet(container, spec) {
    container.classList.add('facet-grid');
    var cells = spec.panels.map(function (panel) {
      var cell = el('div', { 'class': 'panel-plot' });
      container.appendChild(cell);
      return { panel: panel, cell: cell };
    });
    shell(container, function () {
      cells.forEach(function (entry) {
        var panel = entry.panel;
        var yLabel = panel.yLabel + ' (' + spec.axisNote + ')';
        var traces = panel.series.filter(visible).map(function (series) {
          return seriesTrace(series, panel.yLabel);
        });
        var layout = baseLayout(panel.title);
        var dims = panel.series.length ? panel.series[0].x : [];
        layout.xaxis.tickvals = dims;
        layout.xaxis.ticktext = dims.map(String);
        layout.xaxis.title = { text: 'Feature count', standoff: 4 };
        layout.yaxis.type = state.logY ? 'log' : 'linear';
        layout.yaxis.title = { text: yLabel, standoff: 4 };
        keepTidy(layout, traces, dims);
        Plotly.react(entry.cell, traces, layout, CONFIG);
      });
    });
  }

  // ---------- row scaling ----------

  function rowScaling(container, spec) {
    container.classList.add('facet-grid');
    var cells = spec.panels.map(function (panel) {
      var cell = el('div', { 'class': 'panel-plot' });
      container.appendChild(cell);
      return { panel: panel, cell: cell };
    });
    shell(container, function () {
      cells.forEach(function (entry) {
        var traces = entry.panel.series.filter(visible).map(function (series) {
          return seriesTrace(series, 'Median seconds');
        });
        var layout = baseLayout(entry.panel.title);
        layout.xaxis.title = { text: 'Rows', standoff: 4 };
        layout.yaxis.type = 'log';
        layout.yaxis.title = { text: 'Median seconds', standoff: 4 };
        var rowValues = entry.panel.series.length ? entry.panel.series[0].x : [];
        keepTidy(layout, traces, rowValues);
        Plotly.react(entry.cell, traces, layout, CONFIG);
      });
    });
  }

  // ---------- storage size against time, default and compact ----------

  function profiles(container, spec) {
    container.classList.add('facet-grid');
    var cells = spec.panels.map(function (panel) {
      var cell = el('div', { 'class': 'panel-plot' });
      container.appendChild(cell);
      return { panel: panel, cell: cell };
    });
    shell(container, function () {
      cells.forEach(function (entry) {
        var byKey = {};
        entry.panel.points.filter(visible).forEach(function (point) {
          (byKey[point.key] = byKey[point.key] || []).push(point);
        });
        var traces = Object.keys(byKey).map(function (key) {
          var points = byKey[key].slice().sort(function (a, b) { return a.profile < b.profile ? 1 : -1; });
          var first = points[0];
          return {
            type: 'scatter',
            mode: 'lines+markers',
            name: first.label,
            x: points.map(function (p) { return p.x; }),
            y: points.map(function (p) { return p.y; }),
            text: points.map(function (p) { return p.profile; }),
            line: { color: plotColor(first.color), width: 1.5 },
            marker: {
              size: 9,
              symbol: first.layout === 'wide' ? 'square' : 'circle',
              color: points.map(function (p) { return p.profile === 'default' ? plotColor(first.color) : 'rgba(0,0,0,0)'; }),
              line: { color: plotColor(first.color), width: 2 }
            },
            hovertemplate: '<b>' + first.label + '</b> (%{text})<br>%{x:.3g}x raw float32<br>%{y:.3g} s<extra></extra>'
          };
        });
        var layout = baseLayout(entry.panel.title);
        layout.xaxis.type = 'log';
        layout.xaxis.title = { text: 'Storage size (x raw float32)', standoff: 4 };
        layout.yaxis.type = 'log';
        layout.yaxis.title = { text: 'Median seconds', standoff: 4 };
        if (!traces.length) keepTidy(layout, traces, [1, 10]);
        Plotly.react(entry.cell, traces, layout, CONFIG);
      });
    });
  }

  // ---------- explorer: absolute values ----------

  function explorer(container, spec) {
    var operation = el('select', { 'aria-label': 'Operation' });
    spec.operations.forEach(function (op) {
      operation.appendChild(el('option', { value: op.id, text: op.title }));
    });
    var profile = el('select', { 'aria-label': 'Profile' });
    var options = spec.profiles.length > 1 ? ['default', 'compact', 'both'] : ['default'];
    options.forEach(function (name) {
      profile.appendChild(el('option', { value: name, text: name.charAt(0).toUpperCase() + name.slice(1) }));
    });
    var controls = el('div', { 'class': 'controls-row' }, [
      el('label', { 'class': 'toggle', text: 'Operation ' }, [operation]),
      el('label', { 'class': 'toggle', text: 'Profile ' }, [profile])
    ]);
    var plot = el('div', { 'class': 'panel-plot' });
    container.appendChild(controls);
    container.appendChild(plot);

    function draw() {
      var op = operation.value;
      var wanted = profile.value === 'both' ? ['default', 'compact'] : [profile.value];
      var isSize = op === 'storage_size';
      var rows = spec.rows.filter(function (row) {
        return row.operation === (isSize ? 'write' : op) && wanted.indexOf(row.profile) >= 0 && visible(row);
      });
      var byKey = {};
      rows.forEach(function (row) {
        var key = row.key + '/' + row.profile;
        (byKey[key] = byKey[key] || []).push(row);
      });
      var traces = Object.keys(byKey).map(function (key) {
        var group = byKey[key].slice().sort(function (a, b) { return a.dimensions - b.dimensions; });
        var first = group[0];
        var y = group.map(function (row) { return isSize ? row.bytes : row.median; });
        var series = {
          label: first.label + (wanted.length > 1 ? ' (' + first.profile + ')' : ''),
          color: first.color,
          dash: first.profile === 'compact' ? 'dot' : first.dash,
          x: group.map(function (row) { return row.dimensions; }),
          y: y,
          low: isSize ? null : group.map(function (row) { return row.low; }),
          high: isSize ? null : group.map(function (row) { return row.high; })
        };
        return seriesTrace(series, isSize ? 'Bytes' : 'Median seconds');
      });
      var title = operation.options[operation.selectedIndex].text;
      var layout = baseLayout(title);
      layout.height = 440;
      layout.showlegend = true;
      layout.legend = { font: { size: 11 } };
      layout.margin.r = 8;
      var dims = rows.length ? Array.from(new Set(rows.map(function (row) { return row.dimensions; }))).sort(function (a, b) { return a - b; }) : [];
      layout.xaxis.tickvals = dims;
      layout.xaxis.ticktext = dims.map(String);
      layout.xaxis.title = { text: 'Feature count', standoff: 4 };
      layout.yaxis.type = state.logY ? 'log' : 'linear';
      layout.yaxis.title = { text: isSize ? 'Storage size (bytes)' : 'Median seconds', standoff: 4 };
      keepTidy(layout, traces, dims.length ? dims : spec.rows.map(function (row) { return row.dimensions; }));
      Plotly.react(plot, traces, layout, CONFIG);
    }
    operation.addEventListener('change', draw);
    profile.addEventListener('change', draw);
    shell(container, draw);
  }

  // ---------- real-world calculator ----------

  function realWorld(container, spec) {
    var inputs = {};
    function field(id, label, value, min, max, step, log) {
      var number = el('input', { type: 'number', id: 'rw-' + id, min: min, max: max, step: step, value: value });
      var range = el('input', { type: 'range', 'aria-label': label + ' slider' });
      var out = el('output', { text: '' });
      function toRange(v) { return log ? 100 * Math.log10(v / min) / Math.log10(max / min) : v; }
      function fromRange(v) { return log ? min * Math.pow(max / min, v / 100) : Number(v); }
      range.min = log ? 0 : min;
      range.max = log ? 100 : max;
      range.step = log ? 0.5 : step;
      range.value = toRange(value);
      number.addEventListener('input', function () {
        var v = parseFloat(number.value);
        if (!isNaN(v)) { range.value = toRange(Math.min(Math.max(v, min), max)); update(); }
      });
      range.addEventListener('input', function () {
        var v = fromRange(range.value);
        number.value = log ? Number(v.toPrecision(3)) : v;
        update();
      });
      inputs[id] = number;
      return el('label', { text: label }, [number, range, out]);
    }
    var calc = el('div', { 'class': 'calc' }, [
      field('size', 'File size (GB) for CSV wide', spec.datasetGb, 0.1, 1000, 0.1, true),
      field('uses', 'Uses', spec.uses, 1, 100000, 1, true),
      field('price', 'Egress price ($ per GB)', spec.egressDollarsPerGb, 0, 0.2, 0.005, false),
      field('speed', 'Download speed (MB/s)', spec.downloadMbPerSecond, 10, 1000, 10, false)
    ]);
    var headline = el('p', { 'class': 'calc-headline' });
    var charts = el('div', { 'class': 'calc-charts' });
    var timePlot = el('div', { 'class': 'panel-plot' });
    var costPlot = el('div', { 'class': 'panel-plot' });
    charts.appendChild(timePlot);
    charts.appendChild(costPlot);
    var table = el('div', { 'class': 'calc-table' });
    [calc, headline, charts, table].forEach(function (node) { container.appendChild(node); });

    function params() {
      return {
        size: Math.max(parseFloat(inputs.size.value) || spec.datasetGb, 0.001),
        uses: Math.max(parseFloat(inputs.uses.value) || 1, 1),
        price: Math.max(parseFloat(inputs.price.value) || 0, 0),
        speed: Math.max(parseFloat(inputs.speed.value) || 1, 1)
      };
    }

    function compute(p) {
      var scale = p.size / spec.datasetGb;
      return spec.layouts.map(function (layout) {
        var sizeGb = layout.sizeGb * scale;
        var read = layout.readSeconds * scale;
        var download = sizeGb * 1000 / p.speed;
        var total = download + read;
        var egress = sizeGb * p.price;
        return {
          layout: layout, sizeGb: sizeGb, download: download, read: read, total: total,
          egress: egress, timeSpent: total * p.uses, egressSpent: egress * p.uses
        };
      });
    }

    function draw() {
      var p = params();
      var all = compute(p);
      var find = function (backend, layout) {
        return all.filter(function (row) {
          return row.layout.backend === backend && row.layout.layout === layout && row.layout.profile === 'default';
        })[0];
      };
      var base = find('csv', 'wide');
      var best = find('parquet', 'fixed_array');
      if (base && best) {
        headline.textContent = 'For a ' + twoDigits(p.size) + ' GB CSV wide file, Parquet fixed_array takes ' +
          duration(best.total) + ' per use instead of ' + duration(base.total) + ' and costs ' +
          dollars(best.egress) + ' instead of ' + dollars(base.egress) + ' in egress. Over ' +
          Math.round(p.uses).toLocaleString('en-US') + ' uses that saves ' +
          duration(base.timeSpent - best.timeSpent) + ' and ' + dollars(base.egressSpent - best.egressSpent) + '.';
      } else {
        headline.textContent = '';
      }
      var rows = all.filter(function (row) { return visible(row.layout); })
        .sort(function (a, b) { return b.total - a.total; });
      var names = rows.map(function (row) { return row.layout.label; });
      var text = css('--text');
      var grid = css('--grid');

      var timeLayout = baseLayout('Time for one use (lower is better)');
      timeLayout.height = 70 + 30 * rows.length;
      timeLayout.barmode = 'stack';
      timeLayout.margin.l = 190;
      timeLayout.showlegend = true;
      timeLayout.legend = { orientation: 'h', y: -0.18 };
      timeLayout.xaxis.title = { text: 'Seconds', standoff: 4 };
      timeLayout.yaxis = { autorange: 'reversed', gridcolor: grid, automargin: true };
      var xmax = rows.length ? Math.max.apply(null, rows.map(function (row) { return row.total; })) * 1.25 : 1;
      timeLayout.xaxis.range = [0, xmax];
      Plotly.react(timePlot, [
        { type: 'bar', orientation: 'h', name: 'Download', y: names, x: rows.map(function (row) { return row.download; }), marker: { color: '#56B4E9' },
          hovertemplate: '%{y}<br>Download: %{x:.3g} s<extra></extra>' },
        { type: 'bar', orientation: 'h', name: 'Read into memory', y: names, x: rows.map(function (row) { return row.read; }), marker: { color: '#0072B2' },
          text: rows.map(function (row) { return duration(row.total); }), textposition: 'outside', cliponaxis: false, textfont: { color: text },
          hovertemplate: '%{y}<br>Read: %{x:.3g} s<extra></extra>' }
      ], timeLayout, CONFIG);

      var costLayout = baseLayout('Egress over ' + Math.round(p.uses).toLocaleString('en-US') + ' uses (lower is better)');
      costLayout.height = timeLayout.height;
      costLayout.margin.l = 20;
      costLayout.xaxis.title = { text: 'Dollars', standoff: 4 };
      costLayout.yaxis = { autorange: 'reversed', showticklabels: false, gridcolor: grid };
      var cmax = rows.length ? Math.max.apply(null, rows.map(function (row) { return row.egressSpent; })) * 1.25 : 1;
      costLayout.xaxis.range = [0, cmax || 1];
      Plotly.react(costPlot, [
        { type: 'bar', orientation: 'h', y: names, x: rows.map(function (row) { return row.egressSpent; }), marker: { color: '#D55E00' },
          text: rows.map(function (row) { return dollars(row.egressSpent); }), textposition: 'outside', cliponaxis: false, textfont: { color: text },
          hovertemplate: '%{y}<br>%{x:$,.2f}<extra></extra>' }
      ], costLayout, CONFIG);

      var head = ['Layout', 'Size', 'Download', 'Read into memory', 'Total time', 'Egress cost',
        'Time spent', 'Egress spent', 'Time saved', 'Egress cost saved'];
      var body = rows.map(function (row) {
        var isBase = base && row.layout.key === base.layout.key;
        return [row.layout.label, bytesText(row.sizeGb), duration(row.download), duration(row.read), duration(row.total),
          dollars(row.egress), duration(row.timeSpent), dollars(row.egressSpent),
          isBase || !base ? 'baseline' : duration(base.timeSpent - row.timeSpent),
          isBase || !base ? 'baseline' : dollars(base.egressSpent - row.egressSpent)];
      });
      var t = el('table');
      t.appendChild(el('thead', {}, [el('tr', {}, head.map(function (h) { return el('th', { text: h }); }))]));
      t.appendChild(el('tbody', {}, body.map(function (cells) {
        return el('tr', {}, cells.map(function (c) { return el('td', { text: c }); }));
      })));
      table.replaceChildren(t);
      Object.keys(inputs).forEach(function (id) {
        var out = inputs[id].parentNode.querySelector('output');
        out.textContent = ({ size: twoDigits(p.size) + ' GB', uses: Math.round(p.uses).toLocaleString('en-US'),
          price: '$' + p.price.toFixed(3) + ' per GB', speed: Math.round(p.speed) + ' MB/s' })[id];
      });
    }
    var update = draw;
    shell(container, draw);
  }

  // ---------- filter bar ----------

  function buildFilters() {
    var bar = document.getElementById('filters');
    if (!bar) return;
    if (!DATA.layouts.length) { bar.hidden = true; return; }
    var status = el('span', { 'class': 'filters-status' });
    var body = el('div', { 'class': 'filters-body' });
    var chips = {};

    function refresh() {
      Object.keys(chips).forEach(function (key) {
        chips[key].setAttribute('aria-pressed', hidden.has(key) ? 'false' : 'true');
      });
      status.textContent = (DATA.layouts.length - hidden.size) + ' of ' + DATA.layouts.length + ' layouts shown';
      redraws.forEach(function (draw) { draw(); });
    }

    function setWhere(test) {
      hidden.clear();
      DATA.layouts.forEach(function (layout) { if (!test(layout)) hidden.add(layout.key); });
      refresh();
    }

    DATA.backends.forEach(function (backend) {
      var group = el('div', { 'class': 'fgroup' });
      var toggle = el('button', { type: 'button', 'class': 'backend-btn', text: backend.label, title: 'Show or hide ' + backend.label });
      toggle.addEventListener('click', function () {
        var mine = DATA.layouts.filter(function (layout) { return layout.backend === backend.id; });
        var anyHidden = mine.some(function (layout) { return hidden.has(layout.key); });
        mine.forEach(function (layout) { if (anyHidden) hidden.delete(layout.key); else hidden.add(layout.key); });
        refresh();
      });
      group.appendChild(toggle);
      var chipRow = el('span', { 'class': 'chips' });
      group.appendChild(chipRow);
      DATA.layouts.filter(function (layout) { return layout.backend === backend.id; }).forEach(function (layout) {
        var stroke = layout.color === '#000000' ? 'currentColor' : layout.color;
        var svg = '<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="' + stroke +
          '" stroke-width="2.5"' + (layout.dash === 'dash' ? ' stroke-dasharray="6 4"' : '') + '/></svg>';
        var chip = el('button', { type: 'button', 'class': 'chip', 'aria-pressed': 'true', title: layout.label, html: svg });
        chip.appendChild(document.createTextNode(layout.layout));
        chip.addEventListener('click', function () {
          if (hidden.has(layout.key)) hidden.delete(layout.key); else hidden.add(layout.key);
          refresh();
        });
        chips[layout.key] = chip;
        chipRow.appendChild(chip);
      });
      body.appendChild(group);
    });

    function action(text, handler) {
      var button = el('button', { type: 'button', text: text });
      button.addEventListener('click', handler);
      return button;
    }
    var logBox = el('input', { type: 'checkbox', id: 'log-y' });
    logBox.checked = true;
    logBox.addEventListener('change', function () { state.logY = logBox.checked; refresh(); });
    var collapse = action('Hide filters', function () {
      var collapsed = bar.classList.toggle('collapsed');
      collapse.textContent = collapsed ? 'Show filters' : 'Hide filters';
    });
    var head = el('div', { 'class': 'filters-head' }, [
      el('span', {}, [el('strong', { text: 'Key and filter ' }), status]),
      el('span', { 'class': 'filters-actions' }, [
        action('All', function () { setWhere(function () { return true; }); }),
        action('Wide only', function () { setWhere(function (l) { return l.layout === 'wide'; }); }),
        action('Array-like only', function () { setWhere(function (l) { return l.layout !== 'wide'; }); }),
        el('label', { 'class': 'toggle', text: 'Log scale' }, [logBox]),
        collapse
      ])
    ]);
    bar.appendChild(head);
    bar.appendChild(body);
    status.textContent = DATA.layouts.length + ' of ' + DATA.layouts.length + ' layouts shown';
  }

  // ---------- start ----------

  var builders = {
    combined: function (node) { facet(node, DATA.facets.combined); },
    backend_wide: function (node) { facet(node, DATA.facets.backend_wide); },
    wide_layouts: function (node) { facet(node, DATA.facets.wide_layouts); },
    row_scaling: function (node) { rowScaling(node, DATA.rowScaling); },
    profiles: function (node) { profiles(node, DATA.profiles); },
    explorer: function (node) { explorer(node, DATA.explorer); },
    real_world: function (node) { realWorld(node, DATA.realWorld); }
  };

  function start() {
    var nodes = document.querySelectorAll('.figure[data-figure]');
    if (typeof window.Plotly === 'undefined') {
      nodes.forEach(function (node) {
        node.textContent = 'The interactive plots need Plotly from cdn.plot.ly. Check the connection and reload.';
        node.className = 'figure figure-fallback';
      });
      return;
    }
    buildFilters();
    nodes.forEach(function (node) {
      var build = builders[node.getAttribute('data-figure')];
      try {
        if (build) build(node);
      } catch (error) {
        node.textContent = 'This plot could not be drawn: ' + error.message;
        node.className = 'figure figure-error';
      }
    });
    var dark = window.matchMedia('(prefers-color-scheme: dark)');
    if (dark.addEventListener) {
      dark.addEventListener('change', function () { redraws.forEach(function (draw) { draw(); }); });
    }
  }

  window.addEventListener('load', start);
})();
