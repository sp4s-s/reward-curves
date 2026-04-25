"""Shared runtime helpers for efficient single-GPU training."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader


def create_train_loader(dataset, cfg: Any) -> DataLoader:
    num_workers = int(getattr(cfg, "num_workers", 0))
    pin_memory = bool(getattr(cfg, "pin_memory", False))
    generator = torch.Generator()
    generator.manual_seed(int(getattr(cfg, "seed", 0)))

    kwargs = {
        "batch_size": int(cfg.micro_batch_size),
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_context(model, device: str):
    if not str(device).startswith("cuda"):
        return nullcontext()
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def optimizer_step_ready(micro_step: int, grad_accum: int) -> bool:
    return micro_step > 0 and micro_step % grad_accum == 0


__all__ = ["autocast_context", "create_train_loader", "move_batch", "optimizer_step_ready"]
