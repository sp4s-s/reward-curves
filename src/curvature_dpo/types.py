"""Shared dataclass types used across stages.

Kept deliberately minimal in stage 0; later stages extend in their own modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """Hyperparameters shared by SFT and DPO trainers.

    Stage-specific subclasses (SFTConfig, DPOConfig, CurvDPOConfig) will be
    introduced in stages 2–4 and will *inherit* from this; callers should not
    rely on these defaults outside of stage-0 smoke tests.
    """

    run_name: str
    out_dir: str
    seed: int = 20260425

    # model
    model_name: str = "EleutherAI/pythia-410m-deduped"
    max_seq_len: int = 512
    bf16: bool = True

    # optim
    micro_batch_size: int = 8
    grad_accum: int = 2
    lr: float = 5.0e-7
    warmup_steps: int = 50
    total_steps: int = 4000

    # logging / checkpointing
    save_every: int = 250
    log_every: int = 10

    # DPO
    beta: float = 0.1

    # Curvature regularizer (active only when lambda > 0)
    curv_lambda: float = 0.0
    curv_n_positions: int = 2
    curv_n_swaps: int = 2
    curv_swap_topk: int = 50

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size * self.grad_accum


@dataclass
class ProbeItem:
    """A single (prompt, response) probe point used for curvature estimation."""

    prompt: str
    response: str
    prompt_ids: List[int] = field(default_factory=list)
    response_ids: List[int] = field(default_factory=list)


@dataclass
class CurvatureSample:
    """Result of one curvature evaluation on a probe item."""

    probe_idx: int
    positions: List[int]
    per_position_curvature: List[float]
    aggregate: float
    swap_distribution: str  # "Q_topk" | "Q_unif" | "Q_rand-id"
    extra: Optional[dict] = None
