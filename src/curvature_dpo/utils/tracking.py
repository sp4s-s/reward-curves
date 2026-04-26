"""Experiment tracking via Weights & Biases."""
from __future__ import annotations

import os
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
            logger.warning("wandb not installed; continuing without W&B.")
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

    def log_image(self, key: str, path: str, step: int | None = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log({key: self._wandb.Image(path)}, step=step)

    def save_file(self, path: str) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.save(path, base_path=os.path.dirname(path))

    def log_artifact(self, name: str, artifact_type: str, path: str, description: str | None = None) -> None:
        if not (self.enabled and self._wandb is not None):
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type, description=description)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


tracker = Tracker()

__all__ = ["tracker", "Tracker"]
