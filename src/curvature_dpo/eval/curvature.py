"""Curvature estimation logic."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from curvature_dpo.types import ProbeItem, CurvatureSample


def _model_device(model) -> str:
    return str(next(model.parameters()).device)


@torch.no_grad()
def estimate_curvature(
    policy,
    reference_model,
    tokenizer,
    probe_item: ProbeItem,
    n_positions: int = 16,
    n_swaps: int = 8,
    swap_distribution: str = "Q_topk",
    beta: float = 0.1,
    device: str = "cuda"
) -> CurvatureSample:
    """
    Offline curvature estimator for a single probe item.
    Returns detailed swap information for the swap table.
    """
    policy_device = _model_device(policy)
    reference_device = _model_device(reference_model)
    prompt_ids = torch.tensor(probe_item.prompt_ids, dtype=torch.long, device=policy_device)
    response_ids = torch.tensor(probe_item.response_ids, dtype=torch.long, device=policy_device)
    
    full_ids = torch.cat([prompt_ids, response_ids])
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    
    if response_len == 0:
        return CurvatureSample(-1, [], [], 0.0, swap_distribution, extra={"swaps": []})

    def get_logp(ids):
        input_ids = ids.unsqueeze(0)
        logits = policy(input_ids).logits
        ref_logits = reference_model(input_ids.to(reference_device)).logits
        
        shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
        shift_labels = input_ids[:, prompt_len:].contiguous()
        lp = F.log_softmax(shift_logits, dim=-1)
        pi_logp = torch.gather(lp, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).sum()
        
        shift_ref_logits = ref_logits[:, prompt_len-1:-1, :].contiguous()
        ref_lp = F.log_softmax(shift_ref_logits, dim=-1)
        ref_labels = shift_labels.to(reference_device, non_blocking=True)
        ref_logp = torch.gather(ref_lp, dim=-1, index=ref_labels.unsqueeze(-1)).squeeze(-1).sum().to(policy_device)
        
        return pi_logp, ref_logp

    pi_base, ref_base = get_logp(full_ids)
    r_base = beta * (pi_base - ref_base)
    
    effective_len = min(response_len, n_positions)
    positions = torch.linspace(0, response_len - 1, effective_len, dtype=torch.long, device=device).tolist()
    
    per_position_curv = []
    swap_records = []
    
    for pos_idx in positions:
        abs_pos = prompt_len + pos_idx
        orig_token = full_ids[abs_pos].item()
        
        # Sample swaps
        if swap_distribution == "Q_topk":
            with torch.no_grad():
                # Get logits at this position from reference model
                prefix_ids = full_ids[:abs_pos]
                ref_prefix_logits = reference_model(prefix_ids.unsqueeze(0).to(reference_device)).logits
                probs = F.softmax(ref_prefix_logits[0, -1, :], dim=-1)
                k = min(50, probs.size(-1))
                topk = torch.topk(probs, k)
                indices = topk.indices.tolist()
                # Exclude original token
                indices = [idx for idx in indices if idx != orig_token]
                if not indices:
                    continue
                swaps = [indices[i] for i in torch.randperm(len(indices))[:n_swaps]]
        elif swap_distribution == "Q_unif":
            swaps = torch.randint(0, tokenizer.vocab_size, (n_swaps,), device=device).tolist()
        elif swap_distribution == "Q_rand-id":
            # Just random tokens from the vocab, but maybe weighted by frequency? 
            # User said Q_rand-id, usually implies random token ID.
            swaps = torch.randint(0, tokenizer.vocab_size, (n_swaps,), device=device).tolist()
        else:
            raise ValueError(f"Unknown swap_distribution: {swap_distribution}")
        
        sq_diffs = []
        for swap_token in swaps:
            swapped_ids = full_ids.clone()
            swapped_ids[abs_pos] = swap_token
            pi_swapped, ref_swapped = get_logp(swapped_ids)
            r_swapped = beta * (pi_swapped - ref_swapped)
            
            diff = (r_swapped - r_base).item()
            sq_diffs.append(diff**2)
            
            swap_records.append({
                "position": pos_idx,
                "orig_token": orig_token,
                "swap_token": swap_token,
                "r_theta_orig": r_base.item(),
                "r_theta_swap": r_swapped.item(),
                "diff_sq": diff**2
            })
            
        per_position_curv.append(np.mean(sq_diffs))
        
    aggregate = np.mean(per_position_curv)
    
    return CurvatureSample(
        probe_idx=-1, 
        positions=positions,
        per_position_curvature=per_position_curv,
        aggregate=aggregate,
        swap_distribution=swap_distribution,
        extra={"swaps": swap_records, "response_len": response_len}
    )

def compute_bootstrap_ci(data: List[float], n_resamples: int = 1000) -> tuple[float, float]:
    """Computes 95% bootstrap confidence interval for the mean."""
    if not data:
        return 0.0, 0.0
    resampled_means = []
    for _ in range(n_resamples):
        resample = np.random.choice(data, size=len(data), replace=True)
        resampled_means.append(np.mean(resample))
    return float(np.percentile(resampled_means, 2.5)), float(np.percentile(resampled_means, 97.5))
