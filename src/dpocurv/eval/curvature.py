"""Curvature estimation logic."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dpocurv.types import ProbeItem, CurvatureSample


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
    """
    # 1. Prepare base sequence
    prompt_ids = torch.tensor(probe_item.prompt_ids, dtype=torch.long, device=device)
    response_ids = torch.tensor(probe_item.response_ids, dtype=torch.long, device=device)
    
    full_ids = torch.cat([prompt_ids, response_ids])
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    
    # 2. Compute base reward r_theta(x, y)
    def get_reward(ids):
        input_ids = ids.unsqueeze(0)
        logits = policy(input_ids).logits
        ref_logits = reference_model(input_ids).logits
        
        # log pi_theta(y|x)
        shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
        shift_labels = input_ids[:, prompt_len:].contiguous()
        lp = F.log_softmax(shift_logits, dim=-1)
        pi_logp = torch.gather(lp, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).sum()
        
        # log pi_ref(y|x)
        shift_ref_logits = ref_logits[:, prompt_len-1:-1, :].contiguous()
        ref_lp = F.log_softmax(shift_ref_logits, dim=-1)
        ref_logp = torch.gather(ref_lp, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).sum()
        
        return beta * (pi_logp - ref_logp)

    r_base = get_reward(full_ids)
    
    # 3. Choose positions
    effective_len = min(response_len, n_positions)
    if response_len == 0:
        return CurvatureSample(-1, [], [], 0.0, swap_distribution)
    positions = torch.randperm(response_len, device=device)[:effective_len].tolist()
    
    per_position_curv = []
    
    for pos_idx in positions:
        # absolute index in full_ids
        abs_pos = prompt_len + pos_idx
        
        # 4. Sample swaps
        swaps = []
        if swap_distribution == "Q_topk":
            # Get top-k from pi_ref at this position
            with torch.no_grad():
                ref_logits = reference_model(full_ids[:abs_pos].unsqueeze(0)).logits
                k = min(50, ref_logits.size(-1))
                topk_probs = torch.topk(ref_logits[0, -1, :], k)
                swaps = topk_probs.indices.tolist()
                # Sample K from these
                swaps = [swaps[int(i)] for i in torch.randperm(len(swaps))[:n_swaps]]
        elif swap_distribution == "Q_unif":
            swaps = torch.randint(0, tokenizer.vocab_size, (n_swaps,), device=device).tolist()
        else:
            raise ValueError(f"Unknown swap_distribution: {swap_distribution}")
        
        # 5. Compute s_theta^2
        sq_diffs = []
        for swap_token in swaps:
            swapped_ids = full_ids.clone()
            swapped_ids[abs_pos] = swap_token
            r_swapped = get_reward(swapped_ids)
            sq_diffs.append((r_swapped - r_base).pow(2))
            
        per_position_curv.append(torch.stack(sq_diffs).mean().item())
        
    aggregate = sum(per_position_curv) / len(per_position_curv) if per_position_curv else 0.0
    
    return CurvatureSample(
        probe_idx=-1, # caller fills
        positions=positions,
        per_position_curvature=per_position_curv,
        aggregate=aggregate,
        swap_distribution=swap_distribution
    )
