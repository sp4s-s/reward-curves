"""Checkpoint save/load helpers.

We save HF-format model + tokenizer (so generation can re-load with
`AutoModelForCausalLM.from_pretrained`) plus a separate `trainer_state.pt`
for optimizer/scheduler state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Optional[Any],
    step: int,
    out_dir: str | Path,
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    import torch

    ckpt_dir = Path(out_dir) / f"step-{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)

    state: Dict[str, Any] = {
        "step": step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if extras:
        state.update(extras)
    torch.save(state, ckpt_dir / "trainer_state.pt")
    return ckpt_dir


def list_checkpoints(out_dir: str | Path) -> List[Path]:
    out = sorted(Path(out_dir).glob("step-*"))
    return [p for p in out if p.is_dir()]


def step_of(ckpt_dir: str | Path) -> int:
    name = Path(ckpt_dir).name
    if not name.startswith("step-"):
        raise ValueError(f"Not a step-* checkpoint dir: {ckpt_dir}")
    return int(name.split("-", 1)[1])


def latest_checkpoint(out_dir: str | Path) -> Optional[Path]:
    ckpts = list_checkpoints(out_dir)
    return ckpts[-1] if ckpts else None


__all__ = ["save_checkpoint", "list_checkpoints", "step_of", "latest_checkpoint"]
