#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
"""
Autokernel dashboard — lightweight HTTP server for tracking optimization progress.

Usage:
    python3 dashboard.py              # default port 8420
    python3 dashboard.py 9000         # custom port

Then open http://localhost:8420 in a browser.
Auto-refreshes every 30 seconds. Reads results.tsv on each request.
"""

import csv
import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8420
PROJECT_DIR = Path(__file__).parent
RESULTS_FILE = PROJECT_DIR / "results.tsv"

STALL_COLUMNS = ["stall_math", "stall_wait", "stall_scoreboard", "stall_barrier"]

# Vertical annotation markers on charts. Each entry is (iteration_index, label).
# Add new markers here when significant context changes happen mid-run.
MARKERS = [
    (19, "Added CUDA docs to context"),
]


def read_results():
    """Parse results.tsv into a list of dicts."""
    if not RESULTS_FILE.exists():
        return []
    rows = []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for key in ["duration_us", "vs_sdpa", "sm_pct"] + STALL_COLUMNS:
                try:
                    row[key] = float(row.get(key, 0))
                except (ValueError, TypeError):
                    row[key] = 0.0
            row["iteration"] = len(rows)
            rows.append(row)
    return rows


def is_loop_running():
    """Check if the autokernel tmux session is alive."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", "autokernel"],
            capture_output=True, timeout=2,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_api_data():
    """Build JSON payload for the dashboard."""
    rows = read_results()
    if not rows:
        return {"rows": [], "summary": {}}

    kept = [r for r in rows if r.get("status") == "keep"]
    best = min(kept, key=lambda r: r["duration_us"]) if kept else rows[0]
    latest = rows[-1]

    summary = {
        "total_iterations": len(rows),
        "kept": len(kept),
        "discarded": len([r for r in rows if r.get("status") == "discard"]),
        "crashed": len([r for r in rows if r.get("status") == "crash"]),
        "best_duration_us": best["duration_us"],
        "best_vs_sdpa": best["vs_sdpa"],
        "best_sm_pct": best["sm_pct"],
        "best_commit": best.get("commit", ""),
        "baseline_duration_us": rows[0]["duration_us"] if rows else 0,
        "baseline_vs_sdpa": rows[0]["vs_sdpa"] if rows else 0,
        "current_top_stall": latest.get("top_stall", ""),
        "latest_stalls": {col: latest.get(col, 0) for col in STALL_COLUMNS},
    }

    return {
        "rows": rows,
        "summary": summary,
        "loop_running": is_loop_running(),
        "markers": [{"iteration": i, "label": l} for i, l in MARKERS],
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>autokernel dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { font-size: 18px; color: #58a6ff; margin-bottom: 4px; }
  .subtitle { font-size: 12px; color: #484f58; margin-bottom: 20px; }
  .cards { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; min-width: 150px; flex: 1; }
  .card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 28px; font-weight: bold; color: #f0f6fc; margin-top: 4px; }
  .card .sub { font-size: 12px; color: #8b949e; margin-top: 2px; }
  .card .value.good { color: #3fb950; }
  .card .value.warn { color: #d29922; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .chart-box.wide { grid-column: 1 / -1; }
  .chart-box h2 { font-size: 13px; color: #8b949e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .chart-box canvas { max-height: 240px; }
  .chart-box.wide canvas { max-height: 200px; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  th { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; padding: 10px 12px; text-align: left; border-bottom: 1px solid #30363d; }
  td { padding: 8px 12px; font-size: 13px; border-bottom: 1px solid #21262d; }
  tr.keep td:first-child { border-left: 3px solid #3fb950; }
  tr.discard td:first-child { border-left: 3px solid #f85149; }
  tr.crash td:first-child { border-left: 3px solid #d29922; }
  .status-keep { color: #3fb950; }
  .status-discard { color: #f85149; }
  .status-crash { color: #d29922; }
  .no-data { color: #484f58; font-size: 14px; text-align: center; padding: 60px; }
  .loop-status { display: inline-flex; align-items: center; gap: 6px; margin-left: 12px; font-size: 12px; vertical-align: middle; }
  .loop-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .loop-dot.running { background: #3fb950; box-shadow: 0 0 6px #3fb950; animation: pulse 2s infinite; }
  .loop-dot.stopped { background: #f85149; box-shadow: 0 0 6px #f85149; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } .chart-box.wide { grid-column: 1; } }
</style>
</head>
<body>

<h1>autokernel dashboard <span class="loop-status" id="loopStatus"></span></h1>
<div class="subtitle" id="subtitle"></div>

<div class="cards" id="cards"></div>

<div class="charts">
  <div class="chart-box">
    <h2>Kernel Duration (us) &mdash; lower is better</h2>
    <canvas id="chartDuration"></canvas>
  </div>
  <div class="chart-box">
    <h2>vs cuDNN SDPA &mdash; higher is better</h2>
    <canvas id="chartSdpa"></canvas>
  </div>
  <div class="chart-box">
    <h2>SM Throughput %</h2>
    <canvas id="chartSm"></canvas>
  </div>
  <div class="chart-box">
    <h2>Latest Stall Breakdown</h2>
    <canvas id="chartStallsBar"></canvas>
  </div>
  <div class="chart-box wide">
    <h2>Stall Evolution Over Iterations</h2>
    <canvas id="chartStallsLine"></canvas>
  </div>
</div>

<h2 style="font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Experiment Log (newest first)</h2>
<table>
  <thead>
    <tr><th>#</th><th>Commit</th><th>Duration</th><th>vs SDPA</th><th>SM %</th><th>Top Stall</th><th>Status</th><th>Description</th></tr>
  </thead>
  <tbody id="logBody"></tbody>
</table>

<script>
const STALL_META = {
  stall_math:       { label: 'Math Throttle',  color: '#f85149' },
  stall_wait:       { label: 'Wait',           color: '#d29922' },
  stall_scoreboard: { label: 'Scoreboard',     color: '#58a6ff' },
  stall_barrier:    { label: 'Barrier',        color: '#79c0ff' },
};

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { color: '#21262d' }, ticks: { color: '#484f58', font: { size: 10 } } },
    y: { grid: { color: '#21262d' }, ticks: { color: '#484f58', font: { size: 10 } } },
  },
};

let charts = {};
let _rows = [];  // stash for tooltip access
let _markers = [];  // annotation markers

function buildAnnotations(rows) {
  const annot = {};
  _markers.forEach((m, i) => {
    // Find the x-axis position: markers use iteration index
    // Only show if we have enough rows
    if (m.iteration >= rows.length) return;
    annot['marker' + i] = {
      type: 'line',
      xMin: m.iteration,
      xMax: m.iteration,
      borderColor: '#8b949e80',
      borderWidth: 1,
      borderDash: [4, 4],
      label: {
        display: true,
        content: m.label,
        position: 'start',
        backgroundColor: '#161b22',
        color: '#8b949e',
        font: { size: 10, family: 'monospace' },
        padding: { top: 2, bottom: 2, left: 4, right: 4 },
        borderRadius: 3,
      },
    };
  });
  return annot;
}

function makeChart(id, config) {
  const ctx = document.getElementById(id).getContext('2d');
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, config);
}

// W&B-style tooltip: shows experiment details on hover
function experimentTooltip(context) {
  const idx = context[0]?.dataIndex;
  if (idx == null || !_rows[idx]) return '';
  const r = _rows[idx];
  const lines = [
    `#${r.iteration}  ${r.commit || ''}  [${r.status}]`,
    `Duration: ${typeof r.duration_us === 'number' ? r.duration_us.toFixed(1) : r.duration_us} us`,
    `vs SDPA: ${typeof r.vs_sdpa === 'number' ? r.vs_sdpa.toFixed(2) + 'x' : r.vs_sdpa}`,
    `SM: ${typeof r.sm_pct === 'number' ? r.sm_pct.toFixed(1) : r.sm_pct}%`,
    `Top stall: ${(r.top_stall || '').replace(/_/g, ' ')}`,
    ``,
    r.description || '',
  ];
  return lines;
}

const tooltipPlugin = {
  tooltip: {
    enabled: true,
    backgroundColor: '#161b22',
    borderColor: '#30363d',
    borderWidth: 1,
    titleColor: '#f0f6fc',
    bodyColor: '#c9d1d9',
    bodyFont: { family: 'monospace', size: 11 },
    titleFont: { family: 'monospace', size: 12, weight: 'bold' },
    padding: 10,
    callbacks: { footer: experimentTooltip },
  },
};

function renderCards(s) {
  const improvement = s.baseline_duration_us > 0
    ? ((s.baseline_duration_us - s.best_duration_us) / s.baseline_duration_us * 100).toFixed(1)
    : 0;
  const sdpaClass = s.best_vs_sdpa >= 1.0 ? 'good' : 'warn';
  const stallLabel = (s.current_top_stall || 'n/a').replace(/_/g, ' ');

  document.getElementById('cards').innerHTML = `
    <div class="card">
      <div class="label">Iteration</div>
      <div class="value">${s.total_iterations}</div>
      <div class="sub">${s.kept} kept &middot; ${s.discarded} disc &middot; ${s.crashed} crash</div>
    </div>
    <div class="card">
      <div class="label">Best Duration</div>
      <div class="value good">${s.best_duration_us.toFixed(1)} us</div>
      <div class="sub">${improvement}% faster than baseline</div>
    </div>
    <div class="card">
      <div class="label">vs cuDNN SDPA</div>
      <div class="value ${sdpaClass}">${s.best_vs_sdpa.toFixed(2)}x</div>
      <div class="sub">target: &gt;1.50x</div>
    </div>
    <div class="card">
      <div class="label">SM Throughput</div>
      <div class="value">${s.best_sm_pct.toFixed(1)}%</div>
      <div class="sub">ceiling: ~97%</div>
    </div>
    <div class="card">
      <div class="label">#1 Bottleneck</div>
      <div class="value warn" style="font-size:18px">${stallLabel}</div>
      <div class="sub">current top stall</div>
    </div>
  `;
}

function renderLineChart(id, label, color, rows, accessor, extraDatasets) {
  const datasets = [{
    label: label,
    data: rows.map(accessor),
    borderColor: color,
    backgroundColor: color + '20',
    pointBackgroundColor: rows.map(r =>
      r.status === 'keep' ? '#3fb950' : r.status === 'crash' ? '#d29922' : '#f85149'),
    pointRadius: 4,
    tension: 0.3,
    fill: true,
  }];
  if (extraDatasets) datasets.push(...extraDatasets);

  makeChart(id, {
    type: 'line',
    data: { labels: rows.map(r => r.iteration), datasets },
    options: {
      ...chartDefaults,
      plugins: {
        ...tooltipPlugin,
        legend: { display: !!extraDatasets, labels: { color: '#8b949e' } },
        annotation: { annotations: buildAnnotations(rows) },
      },
    },
  });
}

function renderStallBar(stalls) {
  const keys = Object.keys(STALL_META);
  const labels = keys.map(k => STALL_META[k].label);
  const values = keys.map(k => stalls[k] || 0);
  const colors = keys.map(k => STALL_META[k].color);

  makeChart('chartStallsBar', {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderWidth: 0, borderRadius: 3 }],
    },
    options: {
      ...chartDefaults,
      indexAxis: 'y',
      scales: {
        x: { grid: { color: '#21262d' }, ticks: { color: '#484f58', callback: v => v + '%' }, min: 0 },
        y: { grid: { display: false }, ticks: { color: '#c9d1d9', font: { size: 11, family: 'monospace' } } },
      },
    },
  });
}

function renderStallEvolution(rows) {
  const keys = Object.keys(STALL_META);
  // Normalize so all 4 categories sum to 100% at each iteration
  const normalized = rows.map(r => {
    const total = keys.reduce((s, k) => s + (r[k] || 0), 0);
    const result = {};
    keys.forEach(k => { result[k] = total > 0 ? (r[k] || 0) / total * 100 : 0; });
    return result;
  });

  const datasets = keys.map(k => ({
    label: STALL_META[k].label,
    data: normalized.map(r => r[k]),
    borderColor: STALL_META[k].color,
    backgroundColor: STALL_META[k].color + '80',
    pointRadius: 0,
    tension: 0.3,
    fill: true,
    stack: 'stalls',
  }));

  makeChart('chartStallsLine', {
    type: 'line',
    data: { labels: rows.map(r => r.iteration), datasets },
    options: {
      ...chartDefaults,
      plugins: {
        legend: { display: true, labels: { color: '#8b949e', font: { size: 11 } } },
        annotation: { annotations: buildAnnotations(rows) },
      },
      scales: {
        x: { grid: { color: '#21262d' }, ticks: { color: '#484f58', font: { size: 10 } }, title: { display: true, text: 'Iteration', color: '#484f58' } },
        y: { stacked: true, grid: { color: '#21262d' }, ticks: { color: '#484f58', callback: v => v + '%' }, min: 0, max: 100, title: { display: true, text: 'Stall %', color: '#484f58' } },
      },
    },
  });
}

function renderTable(rows) {
  const tbody = document.getElementById('logBody');
  const markerSet = new Set(_markers.map(m => m.iteration));
  const markerMap = Object.fromEntries(_markers.map(m => [m.iteration, m.label]));
  const reversed = [...rows].reverse();
  tbody.innerHTML = reversed.map(r => {
    const mark = markerSet.has(r.iteration)
      ? `<span title="${markerMap[r.iteration]}" style="cursor:help;color:#8b949e;margin-left:4px">|</span>`
      : '';
    return `
    <tr class="${r.status}">
      <td>${r.iteration}${mark}</td>
      <td><code>${r.commit || ''}</code></td>
      <td>${typeof r.duration_us === 'number' ? r.duration_us.toFixed(1) : r.duration_us}</td>
      <td>${typeof r.vs_sdpa === 'number' ? r.vs_sdpa.toFixed(2) + 'x' : r.vs_sdpa}</td>
      <td>${typeof r.sm_pct === 'number' ? r.sm_pct.toFixed(1) : r.sm_pct}%</td>
      <td>${(r.top_stall || '').replace(/_/g, ' ')}</td>
      <td><span class="status-${r.status}">${r.status}</span></td>
      <td>${r.description || ''}</td>
    </tr>`;
  }).join('');
}

async function refresh() {
  try {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    const { rows, summary } = data;

    document.getElementById('subtitle').textContent =
      'Last refreshed: ' + new Date().toLocaleTimeString() + ' \u2022 auto-refreshes every 30s';

    if (!rows || rows.length === 0) {
      document.getElementById('cards').innerHTML = '<div class="no-data">Waiting for first experiment result in results.tsv...</div>';
      return;
    }

    _rows = rows;  // stash for tooltip access
    _markers = data.markers || [];

    const running = data.loop_running;
    document.getElementById('loopStatus').innerHTML = running
      ? '<span class="loop-dot running"></span><span style="color:#3fb950">loop running</span>'
      : '<span class="loop-dot stopped"></span><span style="color:#f85149">loop stopped</span>';

    renderCards(summary);
    renderLineChart('chartDuration', 'Duration (us)', '#58a6ff', rows, r => r.duration_us);
    renderLineChart('chartSdpa', 'vs SDPA', '#3fb950', rows, r => r.vs_sdpa, [{
      label: 'Target (1.5x)',
      data: rows.map(() => 1.5),
      borderColor: '#f8514940',
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false,
    }]);
    renderLineChart('chartSm', 'SM %', '#a371f7', rows, r => r.sm_pct);
    renderStallBar(summary.latest_stalls);
    renderStallEvolution(rows);
    renderTable(rows);
  } catch (e) {
    console.error('Refresh failed:', e);
  }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            data = get_api_data()
            body = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def main():
    server = ReusableHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"autokernel dashboard running at http://localhost:{PORT}")
    print(f"Reading results from {RESULTS_FILE}")
    print("Auto-refreshes every 30 seconds. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
