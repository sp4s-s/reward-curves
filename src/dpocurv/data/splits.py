"""Deterministic data splitting logic."""
from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset
from dpocurv.utils.seed import make_rng


def get_splits(
    dataset_name: str,
    sft_size: int,
    dpo_size: int,
    probe_size: int,
    oracle_pct: float = 0.2,
    seed: int = 20260425,
):
    """
    Splits the dataset into:
    1. Oracle (20% held-out) - Strictly for gold RM evaluation.
    2. SFT-train - Chosen-only shard from remaining 80%.
    3. DPO-train - Triples shard from remaining 80%.
    4. Probe set - Fixed items for curvature estimation.
    """
    ds = load_dataset(dataset_name, split="train_prefs")
    
    # Deterministic shuffle of indices
    rng = make_rng(seed)
    indices = np.arange(len(ds))
    rng.shuffle(indices)
    
    # 1. Oracle split (strictly held out)
    oracle_end = int(len(ds) * oracle_pct)
    oracle_indices = indices[:oracle_end]
    
    # 2. Training pool (remaining indices)
    train_pool_indices = indices[oracle_end:]
    
    # Slice shards from train pool
    needed = sft_size + dpo_size + probe_size
    if needed > len(train_pool_indices):
        raise ValueError(
            f"Requested {needed} train/probe rows, but only {len(train_pool_indices)} "
            "remain after the oracle holdout."
        )

    # SFT
    sft_indices = train_pool_indices[:sft_size]
    # DPO
    dpo_indices = train_pool_indices[sft_size : sft_size + dpo_size]
    # Probe
    probe_indices = train_pool_indices[sft_size + dpo_size : sft_size + dpo_size + probe_size]
    
    return {
        "oracle": ds.select(oracle_indices.tolist()),
        "sft": ds.select(sft_indices.tolist()),
        "dpo": ds.select(dpo_indices.tolist()),
        "probe": ds.select(probe_indices.tolist()),
        "test": load_dataset(dataset_name, split="test_prefs")
    }
