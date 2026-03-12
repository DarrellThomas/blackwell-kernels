#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
"""
Autokernel dashboard — lightweight HTTP server for tracking optimization progress.

Usage:
    python3 dashboard.py              # default port 8420
    python3 dashboard.py 9000         # custom port

Then open http://localhost:8420 in a browser.
Auto-refreshes every 3 minutes. Reads results/<kernel>.tsv on each request.
"""

import csv
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8420
PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"

STALL_COLUMNS = ["stall_math", "stall_wait", "stall_scoreboard", "stall_barrier"]

# Per-kernel configuration
KERNEL_CONFIG = {
    "attention": {
        "ref_label": "vs cuDNN SDPA",
        "ref_column": "vs_sdpa",
        "ref_target": 1.5,
        "gpu": 1,
        "tsv_file": Path("/data/src/blackwell-kernels/results/attention.tsv"),
        "heartbeat": Path("/data/src/blackwell-kernels/.autokernel.attention.alive"),
        "theoretical_floor_us": 38,
        "achievable_ceiling_us": 53,
        "markers": [
            (19, "Added CUDA docs to context"),
        ],
    },
    "attention-r2": {
        "ref_label": "vs cuDNN SDPA",
        "ref_column": "vs_sdpa",
        "ref_target": 1.5,
        "gpu": 1,
        "tsv_file": Path("/data/src/blackwell-kernels-gpu1/results/attention.tsv"),
        "heartbeat": Path("/data/src/blackwell-kernels-gpu1/.autokernel.attention.alive"),
        "theoretical_floor_us": 38,
        "achievable_ceiling_us": 53,
        "markers": [],
    },
    "gemm": {
        "ref_label": "vs cuBLAS",
        "ref_column": "vs_ref",
        "ref_target": 1.0,
        "gpu": 0,
        "tsv_file": Path("/data/src/blackwell-kernels-gemm/results/gemm.tsv"),
        "heartbeat": Path("/data/src/blackwell-kernels-gemm/.autokernel.gemm.alive"),
        "theoretical_floor_us": 614,
        "achievable_ceiling_us": 614,
        "markers": [],
    },
}

DEFAULT_KERNEL_CONFIG = {
    "ref_label": "vs Reference",
    "ref_column": "vs_ref",
    "ref_target": 1.0,
    "markers": [],
}


def list_kernels():
    """List available kernels that have TSV data."""
    kernels = []
    for name, config in sorted(KERNEL_CONFIG.items()):
        tsv = config.get("tsv_file")
        if tsv and tsv.exists():
            kernels.append(name)
    # Also check local results/ for any not in KERNEL_CONFIG
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.tsv")):
            if f.stem not in kernels:
                kernels.append(f.stem)
    return kernels


def read_results(kernel="attention"):
    """Parse results TSV into a list of dicts."""
    config = KERNEL_CONFIG.get(kernel, {})
    results_file = config.get("tsv_file")
    # Fallback to local results/ directory
    if not results_file or not results_file.exists():
        results_file = RESULTS_DIR / f"{kernel}.tsv"
    if not results_file.exists():
        results_file = PROJECT_DIR / "results.tsv"
    if not results_file.exists():
        return []
    rows = []
    config = KERNEL_CONFIG.get(kernel, DEFAULT_KERNEL_CONFIG)
    ref_col = config["ref_column"]
    with open(results_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for key in ["duration_us", ref_col, "vs_sdpa", "vs_ref", "sm_pct"] + STALL_COLUMNS:
                try:
                    row[key] = float(row.get(key, 0))
                except (ValueError, TypeError):
                    row[key] = 0.0
            row["iteration"] = len(rows)
            rows.append(row)
    return rows


def get_kernel_status(kernel):
    """Check kernel loop status and uptime from heartbeat file.

    Heartbeat file contains the loop start epoch on its first line.
    mtime is updated on each heartbeat touch.
    Returns (status, start_epoch): status is 'running'/'stale'/'stopped',
    start_epoch is float or None.
    """
    config = KERNEL_CONFIG.get(kernel, {})
    heartbeat = config.get("heartbeat")
    if not heartbeat or not heartbeat.exists():
        return "stopped", None
    try:
        age = time.time() - heartbeat.stat().st_mtime
        # Read start epoch from file content
        start_epoch = None
        try:
            content = heartbeat.read_text().strip()
            if content:
                start_epoch = float(content.split()[0])
        except (ValueError, OSError):
            pass
        if age < 900:
            return "running", start_epoch
        elif age < 1800:
            return "stale", start_epoch
        else:
            return "stopped", None
    except OSError:
        return "stopped", None


def get_all_loop_status():
    """Return loop status for each kernel."""
    result = {}
    now = time.time()
    for kernel in sorted(KERNEL_CONFIG):
        status, start_epoch = get_kernel_status(kernel)
        info = {"status": status}
        if start_epoch is not None:
            info["start_epoch"] = start_epoch
        config = KERNEL_CONFIG.get(kernel, {})
        heartbeat = config.get("heartbeat")
        if heartbeat and heartbeat.exists():
            try:
                info["heartbeat_age"] = now - heartbeat.stat().st_mtime
            except OSError:
                pass
        result[kernel] = info
    return result


def get_api_data(kernel="attention"):
    """Build JSON payload for the dashboard."""
    rows = read_results(kernel)
    config = KERNEL_CONFIG.get(kernel, DEFAULT_KERNEL_CONFIG)
    ref_col = config["ref_column"]

    if not rows:
        return {
            "rows": [], "summary": {}, "kernel": kernel,
            "config": config, "markers": [],
            "loop_status": get_all_loop_status(),
        }

    kept = [r for r in rows if r.get("status") == "keep"]
    best = min(kept, key=lambda r: r["duration_us"]) if kept else rows[0]
    latest = rows[-1]

    best_vs_ref = best.get(ref_col, best.get("vs_sdpa", best.get("vs_ref", 0)))
    if isinstance(best_vs_ref, str):
        try:
            best_vs_ref = float(best_vs_ref)
        except ValueError:
            best_vs_ref = 0.0

    summary = {
        "total_iterations": len(rows),
        "kept": len(kept),
        "discarded": len([r for r in rows if r.get("status") == "discard"]),
        "crashed": len([r for r in rows if r.get("status") == "crash"]),
        "best_duration_us": best["duration_us"],
        "best_vs_ref": best_vs_ref,
        "best_sm_pct": best["sm_pct"],
        "best_commit": best.get("commit", ""),
        "baseline_duration_us": rows[0]["duration_us"] if rows else 0,
        "baseline_vs_ref": rows[0].get(ref_col, rows[0].get("vs_sdpa", rows[0].get("vs_ref", 0))) if rows else 0,
        "current_top_stall": latest.get("top_stall", ""),
        "latest_stalls": {col: latest.get(col, 0) for col in STALL_COLUMNS},
    }

    return {
        "rows": rows,
        "summary": summary,
        "kernel": kernel,
        "config": config,
        "loop_status": get_all_loop_status(),
        "markers": [{"iteration": i, "label": l} for i, l in config.get("markers", [])],
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
  h1 { font-size: 18px; color: #58a6ff; margin-bottom: 4px; display: inline-block; }
  .header-row { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
  .kernel-select { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 4px 10px; font-family: inherit; font-size: 13px; cursor: pointer; }
  .kernel-select:focus { outline: none; border-color: #58a6ff; }
  .subtitle { font-size: 12px; color: #484f58; margin-bottom: 20px; font-variant-numeric: tabular-nums; }
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
  .loop-dot.stale { background: #d29922; box-shadow: 0 0 6px #d29922; animation: pulse 3s infinite; }
  .loop-dot.stopped { background: #f85149; box-shadow: 0 0 6px #f85149; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } .chart-box.wide { grid-column: 1; } }
</style>
</head>
<body>

<div class="header-row">
  <h1>autokernel dashboard</h1>
  <select class="kernel-select" id="kernelSelect" onchange="switchKernel(this.value)">
    <option value="attention">attention</option>
  </select>
  <span id="loopStatusAll" style="display:inline-flex;align-items:center;gap:14px;margin-left:12px;font-size:12px;font-family:monospace"></span>
</div>
<div class="subtitle" id="subtitle"></div>

<div class="cards" id="cards"></div>

<div class="charts">
  <div class="chart-box">
    <h2>Kernel Duration (us) &mdash; lower is better</h2>
    <canvas id="chartDuration"></canvas>
  </div>
  <div class="chart-box">
    <h2 id="refChartTitle">vs Reference &mdash; higher is better</h2>
    <canvas id="chartRef"></canvas>
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
    <tr><th>#</th><th>Commit</th><th>Duration</th><th id="refColHeader">vs Ref</th><th>SM %</th><th>Top Stall</th><th>Status</th><th>Description</th></tr>
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
let _rows = [];
let _markers = [];
let _config = {};
let _currentKernel = 'attention';

function buildAnnotations(rows) {
  const annot = {};
  _markers.forEach((m, i) => {
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

function getRefValue(row) {
  const col = _config.ref_column || 'vs_sdpa';
  return row[col] || row.vs_sdpa || row.vs_ref || 0;
}

function experimentTooltip(context) {
  const idx = context[0]?.dataIndex;
  if (idx == null || !_rows[idx]) return '';
  const r = _rows[idx];
  const refLabel = _config.ref_label || 'vs Ref';
  const refVal = getRefValue(r);
  const lines = [
    `#${r.iteration}  ${r.commit || ''}  [${r.status}]`,
    `Duration: ${typeof r.duration_us === 'number' ? r.duration_us.toFixed(1) : r.duration_us} us`,
    `${refLabel}: ${typeof refVal === 'number' ? refVal.toFixed(2) + 'x' : refVal}`,
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
  const refLabel = _config.ref_label || 'vs Reference';
  const refTarget = _config.ref_target || 1.0;
  const refClass = s.best_vs_ref >= refTarget ? 'good' : 'warn';
  const stallLabel = (s.current_top_stall || 'n/a').replace(/_/g, ' ');
  const ceiling = _config.achievable_ceiling_us || 0;
  const floor = _config.theoretical_floor_us || 0;
  const pctOfCeiling = ceiling > 0 ? (ceiling / s.best_duration_us * 100).toFixed(0) : '—';
  const headroom = ceiling > 0 ? ((s.best_duration_us - ceiling) / s.best_duration_us * 100).toFixed(0) : '—';

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
      <div class="label">${refLabel}</div>
      <div class="value ${refClass}">${s.best_vs_ref.toFixed(2)}x</div>
      <div class="sub">target: &gt;${refTarget.toFixed(1)}x</div>
    </div>
    <div class="card">
      <div class="label">vs Theory</div>
      <div class="value ${headroom <= 10 ? 'good' : 'warn'}">${pctOfCeiling}%</div>
      <div class="sub">ceiling: ${ceiling} us &middot; floor: ${floor} us</div>
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
    data: rows.map(r => r.status === 'crash' ? null : accessor(r)),
    borderColor: color,
    backgroundColor: color + '20',
    pointBackgroundColor: rows.map(r =>
      r.status === 'keep' ? '#3fb950' : r.status === 'crash' ? '#d29922' : '#f85149'),
    pointRadius: rows.map(r => r.status === 'crash' ? 0 : 4),
    tension: 0.3,
    fill: true,
    spanGaps: true,
  }];
  // Separate crash marker dataset — plots at the bottom of the chart
  const crashIndices = rows.filter(r => r.status === 'crash');
  if (crashIndices.length > 0) {
    datasets.push({
      label: 'crash',
      data: rows.map(r => r.status === 'crash' ? 0 : null),
      borderColor: 'transparent',
      backgroundColor: 'transparent',
      pointBackgroundColor: '#d29922',
      pointBorderColor: '#d29922',
      pointStyle: 'crossRot',
      pointRadius: 6,
      pointBorderWidth: 2,
      showLine: false,
    });
  }
  if (extraDatasets) datasets.push(...extraDatasets);

  makeChart(id, {
    type: 'line',
    data: { labels: rows.map(r => r.iteration), datasets },
    options: {
      ...chartDefaults,
      plugins: {
        ...tooltipPlugin,
        legend: { display: !!extraDatasets, labels: { color: '#8b949e', filter: item => item.text !== 'crash' } },
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
    const refVal = getRefValue(r);
    return `
    <tr class="${r.status}">
      <td>${r.iteration}${mark}</td>
      <td><code>${r.commit || ''}</code></td>
      <td>${typeof r.duration_us === 'number' ? r.duration_us.toFixed(1) : r.duration_us}</td>
      <td>${typeof refVal === 'number' ? refVal.toFixed(2) + 'x' : refVal}</td>
      <td>${typeof r.sm_pct === 'number' ? r.sm_pct.toFixed(1) : r.sm_pct}%</td>
      <td>${(r.top_stall || '').replace(/_/g, ' ')}</td>
      <td><span class="status-${r.status}">${r.status}</span></td>
      <td>${r.description || ''}</td>
    </tr>`;
  }).join('');
}

function switchKernel(kernel) {
  _currentKernel = kernel;
  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set('kernel', kernel);
  history.replaceState(null, '', url);
  refresh();
}

async function loadKernels() {
  try {
    const resp = await fetch('/api/kernels');
    const kernels = await resp.json();
    const sel = document.getElementById('kernelSelect');
    sel.innerHTML = kernels.map(k =>
      `<option value="${k}" ${k === _currentKernel ? 'selected' : ''}>${k}</option>`
    ).join('');
  } catch (e) {
    console.error('Failed to load kernels:', e);
  }
}

function getHeartbeatStyle(ageSec) {
  function lerp(a, b, t) { return Math.round(a + (b - a) * t); }
  function rgb(r, g, b) { return 'rgb(' + r + ',' + g + ',' + b + ')'; }
  if (ageSec == null) {
    return { color: rgb(50, 10, 10), phase: '' };
  }
  if (ageSec < 900) {
    // Green phase: 0-15 min, #00FF00 fading to dim green
    var t = ageSec / 900;
    return { color: rgb(lerp(0, 10, t), lerp(255, 50, t), lerp(0, 10, t)), phase: 'running' };
  } else if (ageSec < 1800) {
    // Yellow phase: 15-30 min, bright yellow fading to dim yellow
    var t = (ageSec - 900) / 900;
    return { color: rgb(lerp(255, 50, t), lerp(215, 43, t), lerp(0, 0, t)), phase: 'stale' };
  } else if (ageSec < 2700) {
    // Red phase: 30-45 min, bright red fading to dim red
    var t = (ageSec - 1800) / 900;
    return { color: rgb(lerp(255, 50, t), lerp(48, 10, t), lerp(48, 10, t)), phase: '' };
  } else {
    // 45+ min: static dim red
    return { color: rgb(50, 10, 10), phase: '' };
  }
}

function renderLoopStatus(loopStatus) {
  const el = document.getElementById('loopStatusAll');
  if (!loopStatus) { el.innerHTML = ''; return; }
  const sorted = Object.entries(loopStatus).sort((a, b) => a[0].localeCompare(b[0]));
  el.innerHTML = sorted.map(([kernel, info]) => {
    const age = info.heartbeat_age;
    const style = getHeartbeatStyle(age);
    const animClass = style.phase ? ' ' + style.phase : '';
    return `<span style="display:inline-flex;align-items:center;gap:4px">` +
      `<span class="loop-dot${animClass}" id="hb-dot-${kernel}" style="background:${style.color};box-shadow:0 0 6px ${style.color}"></span>` +
      `<span id="hb-label-${kernel}" style="color:${style.color}">${kernel}</span></span>`;
  }).join('');
}

async function refresh() {
  try {
    const resp = await fetch(`/api/data?kernel=${_currentKernel}`);
    const data = await resp.json();
    const { rows, summary } = data;

    _config = data.config || {};
    _rows = rows;
    _markers = data.markers || [];

    const refLabel = _config.ref_label || 'vs Reference';
    document.getElementById('refChartTitle').textContent = `${refLabel} — higher is better`;
    document.getElementById('refColHeader').textContent = refLabel;

    _lastRefresh = Date.now();
    _loopStatus = data.loop_status || {};
    renderLoopStatus(_loopStatus);
    updateClocks();

    if (!rows || rows.length === 0) {
      document.getElementById('cards').innerHTML = `<div class="no-data">Waiting for first experiment result in results/${_currentKernel}.tsv...</div>`;
      return;
    }

    renderCards(summary);
    const durationExtra = [];
    if (_config.achievable_ceiling_us) {
      durationExtra.push({
        label: `Achievable ceiling (${_config.achievable_ceiling_us} us)`,
        data: rows.map(() => _config.achievable_ceiling_us),
        borderColor: '#3fb95060',
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false,
      });
    }
    if (_config.theoretical_floor_us && _config.theoretical_floor_us !== _config.achievable_ceiling_us) {
      durationExtra.push({
        label: `Hard floor (${_config.theoretical_floor_us} us)`,
        data: rows.map(() => _config.theoretical_floor_us),
        borderColor: '#f8514940',
        borderDash: [2, 3],
        pointRadius: 0,
        fill: false,
      });
    }
    renderLineChart('chartDuration', 'Duration (us)', '#58a6ff', rows, r => r.duration_us, durationExtra.length ? durationExtra : undefined);

    const refTarget = _config.ref_target || 1.0;
    renderLineChart('chartRef', refLabel, '#3fb950', rows, r => getRefValue(r), [{
      label: `Target (${refTarget.toFixed(1)}x)`,
      data: rows.map(() => refTarget),
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

const REFRESH_INTERVAL = 180; // seconds (3 minutes)
let _lastRefresh = 0;
let _loopStatus = {};

function pad2(n) { return String(n).padStart(2, '0'); }

function fmtHHMM(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}:${pad2(m)}`;
}

function fmtMMSS(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${pad2(m)}:${pad2(s)}`;
}

function updateClocks() {
  const now = Date.now();
  const sinceLast = (now - _lastRefresh) / 1000;
  const untilNext = Math.max(0, REFRESH_INTERVAL - sinceLast);

  // Update heartbeat dot colors — age increases each second
  for (const [kernel, info] of Object.entries(_loopStatus)) {
    const baseAge = info.heartbeat_age;
    const currentAge = baseAge != null ? baseAge + sinceLast : null;
    const style = getHeartbeatStyle(currentAge);
    const dot = document.getElementById('hb-dot-' + kernel);
    const label = document.getElementById('hb-label-' + kernel);
    if (dot) {
      dot.style.background = style.color;
      dot.style.boxShadow = '0 0 6px ' + style.color;
      const wantClass = 'loop-dot' + (style.phase ? ' ' + style.phase : '');
      if (dot.className !== wantClass) dot.className = wantClass;
    }
    if (label) label.style.color = style.color;
  }

  // Find the current kernel's loop uptime from server-provided start_epoch
  let uptimePart = '';
  const info = _loopStatus[_currentKernel];
  if (info && info.start_epoch && info.status !== 'stopped') {
    const uptimeSec = now / 1000 - info.start_epoch;
    uptimePart = `loop running for ${fmtHHMM(uptimeSec)} \u2022 `;
  }

  document.getElementById('subtitle').textContent =
    `${_currentKernel} kernel \u2022 ` +
    uptimePart +
    `next refresh in ${fmtMMSS(untilNext)}`;
}

// Read kernel from URL on load
const params = new URLSearchParams(window.location.search);
if (params.has('kernel')) _currentKernel = params.get('kernel');

loadKernels();
refresh();
setInterval(refresh, REFRESH_INTERVAL * 1000);
setInterval(loadKernels, REFRESH_INTERVAL * 1000);
setInterval(updateClocks, 1000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/kernels":
            kernels = list_kernels()
            body = json.dumps(kernels).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/api/data":
            kernel = qs.get("kernel", ["attention"])[0]
            data = get_api_data(kernel)
            body = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path in ("/", "/index.html"):
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
    print(f"Reading results from {RESULTS_DIR}/")
    print("Auto-refreshes every 3 minutes. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
