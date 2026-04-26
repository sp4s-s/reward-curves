"""Checkpoint management: keep last-N + best-by-metric checkpoints."""
from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Low-level I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_ckpt(
    model, tokenizer, optimizer, scheduler, step: int,
    ckpt_dir: Path, extras: Optional[Dict], torch,
) -> None:
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


def _safe_rmtree(path: Path) -> None:
    """Delete *path* without raising if it no longer exists."""
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Saves checkpoints and prunes old ones, keeping:

    * the **last** ``keep_last_n`` checkpoints (recency),
    * optionally the **best** checkpoint by a scalar metric.

    The "best" dir is never deleted, even when it falls outside the recency
    window.  The total number of checkpoints on disk is at most
    ``keep_last_n + 1`` (when best ≠ any recent).

    Crash-safety: every save goes to a ``_staging`` directory first and is
    renamed into place atomically, so an interrupted write leaves the
    previous checkpoint intact.
    """

    _STATE_FILE = "ckptmgr_state.json"

    def __init__(
        self,
        out_dir: str | Path,
        keep_last_n: int = 2,
        keep_best: bool = True,
        mode: str = "max",          # "max" for reward_acc, "min" for loss
    ) -> None:
        self.out_dir = Path(out_dir)
        self.keep_last_n = max(keep_last_n, 1)
        self.keep_best = keep_best
        self.mode = mode
        self._recents: Deque[Path] = deque()
        self._best_ckpt: Optional[Path] = None
        self._best_score: Optional[float] = None
        self._restore_state()

    # ── public ────────────────────────────────────────────────────────────────

    def save(
        self,
        model, tokenizer, optimizer, scheduler,
        step: int,
        score: Optional[float] = None,
        extras: Optional[Dict] = None,
    ) -> Path:
        """Write a checkpoint and prune obsolete ones. Returns the new path."""
        import torch

        ckpt_dir = self.out_dir / f"step-{step:06d}"
        staging = self.out_dir / f"step-{step:06d}_staging"

        # Write into staging first
        _safe_rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        _write_ckpt(model, tokenizer, optimizer, scheduler, step, staging, extras, torch)

        # Swap staging → final
        _safe_rmtree(ckpt_dir)
        staging.rename(ckpt_dir)

        # Update best
        is_new_best = False
        old_best = self._best_ckpt
        if self.keep_best and score is not None:
            if self._best_score is None:
                is_new_best = True
            elif self.mode == "max" and score > self._best_score:
                is_new_best = True
            elif self.mode == "min" and score < self._best_score:
                is_new_best = True
            if is_new_best:
                self._best_score = score
                self._best_ckpt = ckpt_dir

        # Add to recency window and prune
        self._recents.append(ckpt_dir)
        self._prune()

        # Delete old best if it is no longer protected
        if is_new_best and old_best is not None:
            self._maybe_delete(old_best)

        self._persist_state()
        return ckpt_dir

    @property
    def best(self) -> Optional[Path]:
        return self._best_ckpt

    @property
    def latest(self) -> Optional[Path]:
        return self._recents[-1] if self._recents else None

    # ── internal ──────────────────────────────────────────────────────────────

    def _prune(self) -> None:
        while len(self._recents) > self.keep_last_n:
            old = self._recents.popleft()
            self._maybe_delete(old)

    def _maybe_delete(self, path: Path) -> None:
        """Delete *path* unless it is the best checkpoint."""
        if path == self._best_ckpt:
            return
        _safe_rmtree(path)

    def _state_path(self) -> Path:
        return self.out_dir / self._STATE_FILE

    def _persist_state(self) -> None:
        """Write manager state to JSON so it survives process restart."""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "keep_last_n": self.keep_last_n,
                "keep_best": self.keep_best,
                "mode": self.mode,
                "recents": [str(p) for p in self._recents],
                "best_ckpt": str(self._best_ckpt) if self._best_ckpt else None,
                "best_score": self._best_score,
            }
            self._state_path().write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # non-fatal

    def _restore_state(self) -> None:
        """Reload recency list + best from a previous session if available."""
        sp = self._state_path()
        if not sp.exists():
            return
        try:
            data = json.loads(sp.read_text())
            for p_str in data.get("recents", []):
                p = Path(p_str)
                if (p / "trainer_state.pt").exists():
                    self._recents.append(p)
            best = data.get("best_ckpt")
            if best:
                bp = Path(best)
                if (bp / "trainer_state.pt").exists():
                    self._best_ckpt = bp
                    self._best_score = data.get("best_score")
        except Exception:
            pass  # corrupt state file — start fresh


# ──────────────────────────────────────────────────────────────────────────────
# Functional helpers (used by trainers / main.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    optimizer: Any,
    scheduler: Any,
    ckpt_dir: str | Path,
) -> int:
    """Restore optimizer + scheduler states and return the saved step.

    Model weights must be loaded separately via ``load_policy(ckpt_dir, …)``.
    """
    import torch

    state_path = Path(ckpt_dir) / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No trainer_state.pt in {ckpt_dir}")
    state = torch.load(state_path, map_location="cpu")
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def find_resume_checkpoint(out_dir: str | Path) -> Optional[Path]:
    """Auto-detect the latest checkpoint to resume from.

    Preference order:
    1. ``ckptmgr_state.json`` → most-recent entry (authoritative)
    2. Newest ``step-XXXXXX`` dir with a ``trainer_state.pt``
    """
    root = Path(out_dir)
    sp = root / CheckpointManager._STATE_FILE
    if sp.exists():
        try:
            data = json.loads(sp.read_text())
            recents = data.get("recents", [])
            for p_str in reversed(recents):
                p = Path(p_str)
                if (p / "trainer_state.pt").exists():
                    return p
        except Exception:
            pass
    # Fallback: scan disk
    candidates = sorted(root.glob("step-*/trainer_state.pt"))
    return candidates[-1].parent if candidates else None


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
    "CheckpointManager",
    "find_resume_checkpoint",
    "latest_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "step_of",
]
