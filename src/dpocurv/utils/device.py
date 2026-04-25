"""Hardware-aware CUDA defaults for single-GPU runs."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceProfile:
    device: str
    name: str
    capability: tuple[int, int] | None
    dtype: torch.dtype
    attn_implementation: Optional[str]
    tf32: bool

    @property
    def dtype_name(self) -> str:
        if self.dtype is torch.bfloat16:
            return "bf16"
        if self.dtype is torch.float16:
            return "fp16"
        return "fp32"


def resolve_device(requested: str = "auto") -> str:
    if requested in {"auto", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def get_device_profile(
    device: str = "auto",
    prefer_bf16: bool = True,
    prefer_flash_attn: bool = True,
) -> DeviceProfile:
    resolved = resolve_device(device)
    if not resolved.startswith("cuda") or not torch.cuda.is_available():
        return DeviceProfile(resolved, "cpu", None, torch.float32, None, False)

    idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    name = torch.cuda.get_device_name(idx)

    # bf16 is reliable on Ampere and newer. T4/P100/V100 should use fp16/fp32.
    if prefer_bf16 and torch.cuda.is_bf16_supported() and major >= 8:
        dtype = torch.bfloat16
    elif major >= 7:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # FlashAttention-2 wheels generally target Ampere/Hopper/Blackwell.
    has_flash_attn = find_spec("flash_attn") is not None
    attn_impl = "flash_attention_2" if prefer_flash_attn and major >= 8 and has_flash_attn else None
    tf32 = major >= 8

    return DeviceProfile(resolved, name, (major, minor), dtype, attn_impl, tf32)


def configure_torch_for_device(profile: DeviceProfile) -> None:
    if not profile.device.startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = profile.tf32
    torch.backends.cudnn.allow_tf32 = profile.tf32
    if hasattr(torch, "set_float32_matmul_precision") and profile.tf32:
        torch.set_float32_matmul_precision("high")


__all__ = ["DeviceProfile", "configure_torch_for_device", "get_device_profile", "resolve_device"]
