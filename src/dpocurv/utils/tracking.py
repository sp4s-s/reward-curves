"""Optional experiment tracking integrations."""
from __future__ import annotations

from typing import Any


class Tracker:
    def __init__(self) -> None:
        self._wandb = None
        self.enabled = False

    def init(self, cfg: Any, out_dir, logger) -> None:
        if not bool(cfg.wandb.enabled):
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb.enabled=true but wandb is not installed; continuing without W&B.")
            return

        from omegaconf import OmegaConf

        self._wandb = wandb
        self._wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=f"{cfg.experiment.name}_{out_dir.name}",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir),
        )
        self.enabled = True

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


tracker = Tracker()


__all__ = ["tracker", "Tracker"]
