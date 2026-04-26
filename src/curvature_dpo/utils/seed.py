"""Deterministic seeding."""
from __future__ import annotations

import os
import random
import hashlib
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # warn_only=True so an unsupported op logs but does not crash eval.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_rng(seed: int) -> np.random.Generator:
    """Branch a numpy Generator off the master seed for data-side randomness."""
    return np.random.default_rng(seed)


def derive_seed(master: int, *salts: object) -> int:
    """Derive a stable child seed from a master seed and arbitrary salts."""
    payload = repr((int(master), *salts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFFFFFF


__all__ = ["set_seed", "make_rng", "derive_seed"]
