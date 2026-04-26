"""Run summaries, error records, and Kaggle-friendly artifact archives."""
from __future__ import annotations

import json
import shutil
import subprocess
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from curvature_dpo.utils.dashboard import write_dashboard


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last: dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def write_error_record(out_dir: str | Path, exc: BaseException) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "error.json"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    path.write_text(json.dumps(record, indent=2))
    return path


def write_run_summary(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    train_last = _read_last_jsonl(out / "train.jsonl") or _read_last_jsonl(out / "metrics.jsonl")
    gpu_last = _read_last_jsonl(out / "gpu_metrics.jsonl")
    checkpoints = sorted(p.name for p in out.glob("step-*") if p.is_dir())
    traces = sorted(str(p.relative_to(out)) for p in (out / "torch_traces").glob("*")) if (out / "torch_traces").exists() else []
    files = sorted(p.name for p in out.iterdir()) if out.exists() else []
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(out),
        "status": "failed" if (out / "error.json").exists() else "completed",
        "latest_train_metrics": train_last,
        "latest_gpu_metrics": gpu_last,
        "checkpoints": checkpoints,
        "profiler_traces": traces,
        "files": files,
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def write_run_meta(cfg: Any, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    try:
        from omegaconf import OmegaConf

        (meta / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception:
        (meta / "config_repr.txt").write_text(repr(cfg))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True).strip()
    except Exception:
        sha = "not_available"
    (meta / "git_sha.txt").write_text(sha + "\n")
    try:
        freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            freeze = subprocess.check_output(["python3", "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT)
        except Exception as exc:
            freeze = f"pip freeze unavailable: {exc}\n"
    (meta / "pip_freeze.txt").write_text(freeze)
    return meta


def archive_run(out_dir: str | Path, archive_name: str = "run_artifacts.zip", copy_to_cwd: bool = True) -> Path:
    out = Path(out_dir)
    archive_path = out / archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path == archive_path or not path.is_file():
                continue
            zf.write(path, path.relative_to(out))
    if copy_to_cwd:
        target = Path.cwd() / f"{out.name}_{archive_name}"
        shutil.copy2(archive_path, target)
    return archive_path


def finalize_run_artifacts(cfg: Any, logger, failed: bool = False) -> dict[str, str]:
    out_dir = Path(cfg.out_dir)
    dashboard = write_dashboard(out_dir, cfg.dashboard.filename)
    summary = write_run_summary(out_dir)
    archive = None
    if bool(getattr(cfg.diagnostics, "final_archive", True)):
        archive = archive_run(out_dir, cfg.diagnostics.archive_name)
    result = {
        "dashboard": str(dashboard),
        "summary": str(summary),
    }
    if archive is not None:
        result["archive"] = str(archive)
        result["download_copy"] = str(Path.cwd() / f"{out_dir.name}_{cfg.diagnostics.archive_name}")
    logger.info("Final artifacts: %s", result)
    print("\nFinal artifacts")
    for key, value in result.items():
        print(f"- {key}: {value}")
    latest = _read_last_jsonl(out_dir / "train.jsonl") or _read_last_jsonl(out_dir / "metrics.jsonl")
    if latest:
        print("\nFinal logged metrics")
        for key in sorted(latest):
            print(f"- {key}: {latest[key]}")
    return result


__all__ = ["archive_run", "finalize_run_artifacts", "write_error_record", "write_run_meta", "write_run_summary"]
