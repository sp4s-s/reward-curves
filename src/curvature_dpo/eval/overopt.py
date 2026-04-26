"""Overoptimization analysis logic."""
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Sequence, List
from scipy import stats

def compute_overoptimization_gap(
    implicit_rewards: torch.Tensor,
    gold_rewards: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes Delta = mean(r_theta) - mean(r_gold).
    """
    imp_mean = implicit_rewards.mean().item()
    gold_mean = gold_rewards.mean().item()
    
    return {
        "mean_implicit": imp_mean,
        "mean_gold": gold_mean,
        "gap": imp_mean - gold_mean
    }

def compute_best_of_n(
    gold_rewards_matrix: np.ndarray,  # [num_prompts, total_samples]
    n_values: Sequence[int] = (1, 2, 4),
) -> Dict[str, float]:
    """
    Computes mean_x [max over n samples of r_gold(x, y_hat)].
    """
    results = {}
    num_prompts, total_samples = gold_rewards_matrix.shape
    
    for n in n_values:
        if n > total_samples:
            continue
        # For each prompt, take first n samples and find max
        max_scores = np.max(gold_rewards_matrix[:, :n], axis=1)
        results[f"bon_{n}"] = float(np.mean(max_scores))
        
    return results

def compute_goodhart_slope(r_gold: np.ndarray, r_theta: np.ndarray) -> Dict[str, float]:
    """OLS slope of r_gold regressed on r_theta."""
    if len(r_gold) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
    slope, intercept, r_value, p_value, std_err = stats.linregress(r_theta, r_gold)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2)
    }

def lexical_diversity(texts: List[str], n: int = 1) -> float:
    """Distinct-n lexical diversity."""
    if not texts:
        return 0.0
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)
