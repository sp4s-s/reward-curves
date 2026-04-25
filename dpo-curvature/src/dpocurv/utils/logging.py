"""Stdout + file logger and a JSONL metric writer.

Design choices:
- Stdout handler always on, file handler optional (per-run).
- JSONL metric stream is the canonical training log; W&B is best-effort
  (configured later in stage 6) and never the source of truth.
- Loggers are idempotent — re-calling `get_logger` returns the same instance.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def get_logger(
    name: str = "dpocurv",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class JsonlMetricWriter:
    """Append-only JSONL writer for training/eval metrics.

    Use as a context manager so the file handle is flushed on exit even if
    training crashes mid-step.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered so each write is durable enough for tail -f.
        self._fh = open(self.path, "a", buffering=1)

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=float) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "JsonlMetricWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["get_logger", "JsonlMetricWriter"]
