"""Self-contained HTML dashboards for Kaggle/notebook run artifacts."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    return sorted(keys)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _discover_files(out_dir: Path) -> dict[str, list[str]]:
    groups = {
        "checkpoints": [],
        "profiler_traces": [],
        "metrics": [],
    }
    for path in sorted(out_dir.glob("step-*")):
        if path.is_dir():
            groups["checkpoints"].append(path.name)
    for name in ["train.jsonl", "metrics.jsonl", "gpu_metrics.jsonl", "run.log", "summary.json", "error.json"]:
        if (out_dir / name).exists():
            groups["metrics"].append(name)
    for trace in sorted((out_dir / "torch_traces").glob("*")) if (out_dir / "torch_traces").exists() else []:
        groups["profiler_traces"].append(str(trace.relative_to(out_dir)))
    return groups


def write_dashboard(out_dir: str | Path, filename: str = "run_dashboard.html") -> Path:
    out = Path(out_dir)
    train_rows = _read_jsonl(out / "train.jsonl") or _read_jsonl(out / "metrics.jsonl")
    gpu_rows = _read_jsonl(out / "gpu_metrics.jsonl")
    rows = train_rows or gpu_rows
    numeric_keys = _numeric_keys(rows)
    gpu_keys = [k for k in numeric_keys if k.startswith("gpu/")]
    loss_keys = [
        k for k in numeric_keys
        if k in {"loss", "l_dpo", "l_curv", "reward_acc", "grad_norm/pre_clip", "grad_norm/post_clip", "lr", "dpo/roc_auc", "dpo/error_rate"}
    ]
    throughput_keys = [k for k in numeric_keys if k.startswith("throughput/") or k.startswith("time/")]
    other_keys = [k for k in numeric_keys if k not in set(gpu_keys + loss_keys + throughput_keys + ["step", "micro_step"])]
    files = _discover_files(out)
    latest = rows[-1] if rows else {}

    payload = {
        "rows": rows,
        "gpuRows": gpu_rows,
        "groups": {
            "Training": loss_keys,
            "GPU": gpu_keys,
            "Throughput": throughput_keys,
            "Other": other_keys,
        },
        "latest": latest,
        "files": files,
    }
    data = json.dumps(_json_safe(payload), allow_nan=False)
    title = html.escape(out.name)
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>curvature_dpo run dashboard - {title}</title>
  <style>
    :root {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    body {{ margin: 0; }}
    header {{ padding: 20px 24px; background: #111827; color: white; }}
    main {{ padding: 18px 24px 32px; max-width: 1440px; margin: 0 auto; }}
    h1 {{ margin: 0; font-size: 22px; }}
    h2 {{ font-size: 16px; margin: 22px 0 10px; }}
    .sub {{ color: #cbd5e1; margin-top: 6px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
    .tile {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
    .label {{ color: #64748b; font-size: 12px; }}
    .value {{ font-weight: 700; font-size: 20px; margin-top: 4px; }}
    .panel {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-top: 12px; }}
    canvas {{ width: 100%; height: 300px; display: block; }}
    .checks {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin-bottom: 10px; }}
    label {{ font-size: 13px; white-space: nowrap; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: right; border-bottom: 1px solid #e5e7eb; padding: 6px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }}
    .files {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ border: 1px solid #cbd5e1; border-radius: 999px; padding: 4px 8px; background: white; font-size: 12px; }}
  </style>
</head>
<body>
<header>
  <h1>curvature_dpo Run Dashboard</h1>
  <div class="sub">{title}</div>
</header>
<main>
  <section class="grid" id="summary"></section>
  <section id="charts"></section>
  <section class="panel">
    <h2>Artifacts</h2>
    <div id="files"></div>
  </section>
  <section class="panel">
    <h2>Recent Metrics</h2>
    <div style="overflow:auto"><table id="recent"></table></div>
  </section>
</main>
<script>
const data = {data};
const colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c", "#15803d", "#a16207"];
function fmt(v) {{
  if (v === undefined || v === null || Number.isNaN(v)) return "";
  if (Math.abs(v) >= 1000) return Number(v).toFixed(0);
  if (Math.abs(v) >= 10) return Number(v).toFixed(2);
  return Number(v).toFixed(4);
}}
function addSummary() {{
  const keys = ["step", "loss", "reward_acc", "l_curv", "gpu/mem_allocated_gb", "gpu/mem_divergence_pct", "gpu/gpu_util_pct", "throughput/tokens_per_sec"];
  const root = document.getElementById("summary");
  keys.filter(k => data.latest[k] !== undefined).forEach(k => {{
    const el = document.createElement("div");
    el.className = "tile";
    el.innerHTML = `<div class="label">${{k}}</div><div class="value">${{fmt(data.latest[k])}}</div>`;
    root.appendChild(el);
  }});
}}
function drawChart(canvas, keys) {{
  const ctx = canvas.getContext("2d");
  const rows = data.rows;
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = 300 * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, w, h);
  const pad = 36;
  const xs = rows.map((r, i) => Number(r.step ?? i));
  const selected = keys.filter(k => document.getElementById("check-" + btoa(k))?.checked);
  let values = [];
  selected.forEach(k => rows.forEach(r => {{ if (typeof r[k] === "number") values.push(r[k]); }}));
  if (!rows.length || !selected.length || !values.length) return;
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...values), maxY = Math.max(...values);
  if (minY === maxY) {{ minY -= 1; maxY += 1; }}
  ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, 270); ctx.lineTo(canvas.clientWidth - 8, 270); ctx.stroke();
  selected.forEach((k, idx) => {{
    ctx.strokeStyle = colors[idx % colors.length]; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    rows.forEach((r, i) => {{
      if (typeof r[k] !== "number") return;
      const x = pad + ((xs[i] - minX) / Math.max(maxX - minX, 1)) * (canvas.clientWidth - pad - 12);
      const y = 270 - ((r[k] - minY) / (maxY - minY)) * 250;
      if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
  ctx.fillStyle = "#475569"; ctx.font = "12px sans-serif";
  ctx.fillText(fmt(maxY), 4, 16); ctx.fillText(fmt(minY), 4, 270);
}}
function addChartGroup(name, keys) {{
  if (!keys.length) return;
  const section = document.createElement("section");
  section.className = "panel";
  const checks = keys.map((k, i) => `<label><input id="check-${{btoa(k)}}" type="checkbox" ${{i < 6 ? "checked" : ""}}> ${{k}}</label>`).join("");
  section.innerHTML = `<h2>${{name}}</h2><div class="checks">${{checks}}</div><canvas></canvas>`;
  document.getElementById("charts").appendChild(section);
  const canvas = section.querySelector("canvas");
  section.querySelectorAll("input").forEach(input => input.addEventListener("change", () => drawChart(canvas, keys)));
  drawChart(canvas, keys);
}}
function addFiles() {{
  const root = document.getElementById("files");
  Object.entries(data.files).forEach(([group, files]) => {{
    const div = document.createElement("div");
    div.innerHTML = `<h2>${{group}}</h2><div class="files">${{files.map(f => `<span class="pill">${{f}}</span>`).join("") || "<span class='pill'>none</span>"}}</div>`;
    root.appendChild(div);
  }});
}}
function addRecent() {{
  const rows = data.rows.slice(-20);
  const keys = Array.from(new Set(rows.flatMap(r => Object.keys(r)))).slice(0, 18);
  const table = document.getElementById("recent");
  table.innerHTML = `<thead><tr>${{keys.map(k => `<th>${{k}}</th>`).join("")}}</tr></thead>` +
    `<tbody>${{rows.map(r => `<tr>${{keys.map(k => `<td>${{typeof r[k] === "number" ? fmt(r[k]) : (r[k] ?? "")}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
}}
addSummary();
Object.entries(data.groups).forEach(([name, keys]) => addChartGroup(name, keys));
addFiles();
addRecent();
</script>
</body>
</html>
"""
    path = out / filename
    path.write_text(page)
    return path


class DashboardWriter:
    def __init__(self, cfg: Any, logger) -> None:
        self.enabled = bool(getattr(cfg.dashboard, "enabled", True))
        self.update_every = int(getattr(cfg.dashboard, "update_every", getattr(cfg, "log_every", 10)))
        self.filename = str(getattr(cfg.dashboard, "filename", "run_dashboard.html"))
        self.out_dir = Path(cfg.out_dir)
        self.logger = logger
        self.path = self.out_dir / self.filename
        if self.enabled:
            self.logger.info("Dashboard enabled: %s", self.path)

    def maybe_update(self, step: int, force: bool = False) -> None:
        if not self.enabled:
            return
        if force or step == 1 or step % self.update_every == 0:
            write_dashboard(self.out_dir, self.filename)


__all__ = ["DashboardWriter", "write_dashboard"]
