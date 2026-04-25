"""Overoptimization analysis logic."""
from __future__ import annotations

import torch
from typing import Dict, Sequence


def compute_overoptimization_gap(
    implicit_rewards: torch.Tensor,
    gold_rewards: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes Delta = mean(r_theta) - mean(r_gold).
    Note: Both sets should be z-normalized externally for cross-checkpoint comparison.
    """
    imp_mean = implicit_rewards.mean().item()
    gold_mean = gold_rewards.mean().item()
    
    return {
        "mean_implicit": imp_mean,
        "mean_gold": gold_mean,
        "gap": imp_mean - gold_mean
    }


def compute_best_of_n(
    gold_rewards_matrix: torch.Tensor,  # [num_prompts, n_samples]
    n_values: Sequence[int] = (1, 2, 4),
) -> Dict[str, float]:
    """
    Computes mean(max_n(r_gold)) for different n.
    """
    results = {}
    num_prompts, total_samples = gold_rewards_matrix.shape
    
    for n in n_values:
        if n > total_samples:
            continue
        # Take first n samples for each prompt
        subset = gold_rewards_matrix[:, :n]
        bon_n = subset.max(dim=1).values.mean().item()
        results[f"bon_{n}"] = bon_n
        
    return results
