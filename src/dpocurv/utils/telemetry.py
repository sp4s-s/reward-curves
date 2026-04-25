"""GPU telemetry, throughput counters, and optional PyTorch profiler traces."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from dpocurv.utils.logging import JsonlMetricWriter


class NvmlProbe:
    def __init__(self, device: str, enabled: bool = True) -> None:
        self._nvml = None
        self._handle = None
        if not enabled:
            return
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            index = torch.cuda.current_device()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            self._nvml = None
            self._handle = None

    def read(self) -> dict[str, float]:
        if self._nvml is None or self._handle is None:
            return {}
        out: dict[str, float] = {}
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            out["gpu_util_pct"] = float(util.gpu)
            out["gpu_mem_util_pct"] = float(util.memory)
        except Exception:
            pass
        try:
            out["gpu_temp_c"] = float(self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            out["gpu_power_w"] = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        except Exception:
            pass
        return out


class GpuTelemetry:
    def __init__(self, cfg: Any, logger, device: str) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.enabled = bool(getattr(cfg.telemetry, "enabled", True))
        self.log_every = int(getattr(cfg.telemetry, "log_every", getattr(cfg, "log_every", 10)))
        self.out_dir = Path(cfg.out_dir)
        self.writer = None
        self.nvml = NvmlProbe(device, enabled=bool(getattr(cfg.telemetry, "show_nvml", True)))
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.last_step = 0
        if self.enabled:
            self.writer = JsonlMetricWriter(self.out_dir / "gpu_metrics.jsonl")
            self.logger.info("GPU telemetry enabled: %s", self.out_dir / "gpu_metrics.jsonl")

    def __enter__(self) -> "GpuTelemetry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def capture(self, metrics: dict[str, Any], step: int, micro_step: int, tokens: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            return metrics
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = now - self.start_time
        step_dt = max(now - self.last_time, 1e-9)
        step_delta = max(step - self.last_step, 1)
        self.last_time = now
        self.last_step = step

        enriched = dict(metrics)
        enriched["time/elapsed_sec"] = elapsed
        enriched["time/steps_per_sec"] = step_delta / step_dt
        enriched["time/sec_per_step"] = step_dt / step_delta
        enriched["micro_step"] = micro_step
        if tokens is not None and step_dt > 0:
            enriched["throughput/tokens_per_sec"] = float(tokens) / step_dt

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            stats = torch.cuda.memory_stats()
            inactive = int(stats.get("inactive_split_bytes.all.current", 0))
            active = int(stats.get("active_bytes.all.current", 0))
            divergence = reserved - allocated
            enriched.update(
                {
                    "gpu/mem_allocated_gb": allocated / 1e9,
                    "gpu/mem_reserved_gb": reserved / 1e9,
                    "gpu/max_mem_allocated_gb": max_allocated / 1e9,
                    "gpu/max_mem_reserved_gb": max_reserved / 1e9,
                    "gpu/mem_divergence_gb": divergence / 1e9,
                    "gpu/mem_divergence_pct": (divergence / reserved * 100.0) if reserved else 0.0,
                    "gpu/inactive_split_gb": inactive / 1e9,
                    "gpu/fragmentation_pct": (inactive / max(active + inactive, 1) * 100.0),
                }
            )
            enriched.update({f"gpu/{k}": v for k, v in self.nvml.read().items()})
        return enriched

    def write(self, metrics: dict[str, Any]) -> None:
        if self.writer is not None:
            self.writer.write(metrics)

    def print_summary(self, metrics: dict[str, Any]) -> None:
        step = int(metrics.get("step", 0))
        if step != 1 and step % self.log_every != 0:
            return
        loss = metrics.get("loss", float("nan"))
        mem = metrics.get("gpu/mem_allocated_gb")
        div = metrics.get("gpu/mem_divergence_pct")
        util = metrics.get("gpu/gpu_util_pct")
        toks = metrics.get("throughput/tokens_per_sec")
        parts = [f"step={step}", f"loss={loss:.4f}" if isinstance(loss, float) else f"loss={loss}"]
        if mem is not None:
            parts.append(f"mem={mem:.2f}GB")
        if div is not None:
            parts.append(f"mem_div={div:.1f}%")
        if util is not None:
            parts.append(f"util={util:.0f}%")
        if toks is not None:
            parts.append(f"tok/s={toks:.0f}")
        self.logger.info(" | ".join(parts))


def profiler_context(cfg: Any, out_dir: str | Path):
    profiler_cfg = getattr(cfg, "profiler", None)
    if profiler_cfg is None or not bool(getattr(profiler_cfg, "enabled", False)):
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    if not hasattr(torch, "profiler"):
        return nullcontext()
    trace_dir = Path(out_dir) / str(getattr(profiler_cfg, "trace_dir", "torch_traces"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(getattr(profiler_cfg, "wait", 5)),
            warmup=int(getattr(profiler_cfg, "warmup", 2)),
            active=int(getattr(profiler_cfg, "active", 3)),
            repeat=int(getattr(profiler_cfg, "repeat", 1)),
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=bool(getattr(profiler_cfg, "record_shapes", False)),
        profile_memory=bool(getattr(profiler_cfg, "profile_memory", True)),
        with_stack=bool(getattr(profiler_cfg, "with_stack", False)),
    )


__all__ = ["GpuTelemetry", "profiler_context"]
