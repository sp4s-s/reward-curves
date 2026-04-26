"""Checkpoint save/load helpers with rolling (keep-last-only) support."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


_ROLLING_NAME = "checkpoint_rolling"
_PREV_NAME = "checkpoint_rolling_prev"


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Optional[Any],
    step: int,
    out_dir: str | Path,
    extras: Optional[Dict[str, Any]] = None,
    keep_last_only: bool = True,
) -> Path:
    """Save a checkpoint.

    When *keep_last_only* is True (default) only one rolling checkpoint is kept
    on disk at any time.  The previous checkpoint is deleted *after* the new one
    is fully written, so a crash during save never loses the previous state.
    """
    import torch

    root = Path(out_dir)

    if keep_last_only:
        ckpt_dir = root / _ROLLING_NAME
        # Write to a staging dir first — crash during write never corrupts
        # the previous good checkpoint.
        staging = root / f"{_ROLLING_NAME}_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        _write_ckpt(model, tokenizer, optimizer, scheduler, step, staging, extras, torch)
        # Rotate: current rolling → prev, staging → rolling
        if ckpt_dir.exists():
            if (root / _PREV_NAME).exists():
                shutil.rmtree(root / _PREV_NAME)
            ckpt_dir.rename(root / _PREV_NAME)
        staging.rename(ckpt_dir)
        # Delete previous only after new one is safely in place
        prev = root / _PREV_NAME
        if prev.exists():
            shutil.rmtree(prev)
        return ckpt_dir
    else:
        ckpt_dir = root / f"step-{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _write_ckpt(model, tokenizer, optimizer, scheduler, step, ckpt_dir, extras, torch)
        return ckpt_dir


def _write_ckpt(model, tokenizer, optimizer, scheduler, step, ckpt_dir, extras, torch):
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


def load_checkpoint(
    optimizer: Any,
    scheduler: Any,
    ckpt_dir: str | Path,
) -> int:
    """Restore optimizer + scheduler from *ckpt_dir* and return the saved step.

    Model weights must be loaded separately by passing *ckpt_dir* as the
    model_name to ``load_policy()``.
    """
    import torch

    state_path = Path(ckpt_dir) / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No trainer_state.pt found in {ckpt_dir}")
    state = torch.load(state_path, map_location="cpu")
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def find_rolling_checkpoint(out_dir: str | Path) -> Optional[Path]:
    """Return the rolling checkpoint path if it exists and is valid."""
    p = Path(out_dir) / _ROLLING_NAME
    return p if (p / "trainer_state.pt").exists() else None


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


__all__ = [
    "find_rolling_checkpoint",
    "latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "save_checkpoint",
    "step_of",
]
