"""DPO and Curv-DPO loss functions."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_logprobs(logits, labels):
    """
    Computes log-probabilities for the labels in the logits.
    Masks labels == -100.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log-probs memory efficiently using cross_entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss = loss.view(shift_labels.shape)
    return -loss.sum(-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
):
    """Standard DPO loss."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Optional: metrics
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, chosen_rewards, rejected_rewards, reward_accuracies


def curvature_loss(
    policy,
    reference_model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    n_positions: int = 2,
    n_swaps: int = 2,
    swap_topk: int = 50,
    device: str = "cuda"
):
    """
    In-loop curvature regularizer.
    Samples swaps for the 'chosen' completion and penalizes squared implicit-reward changes.
    """
    batch_size, seq_len = input_ids.shape
    
    # Calculate base implicit rewards for the chosen completions
    policy_logits = policy(input_ids).logits
    with torch.no_grad():
        ref_logits = reference_model(input_ids).logits
    
    # We need the logprobs per token to calculate the "local" reward change
    # or just the aggregate reward change. The spec says r_theta(x, swap(y)) - r_theta(x, y).
    
    pi_logps_base = compute_logprobs(policy_logits, labels)
    ref_logps_base = compute_logprobs(ref_logits, labels)
    r_base = beta * (pi_logps_base - ref_logps_base)
    
    total_curv_loss = 0.0
    count = 0
    
    # For simplicity and to avoid too many forwards, we sample 1 position and 1 swap per batch element
    # for the regularizer, or iterate a small number.
    for _ in range(n_positions):
        for _ in range(n_swaps):
            # Sample a position in each response (masking prompt)
            # labels != -100 is the response
            # This is a bit tricky to vectorize efficiently without custom kernels
            # but we'll do it per-batch element for clarity.
            
            swapped_input_ids = input_ids.clone()
            swapped_labels = labels.clone()
            for b in range(batch_size):
                resp_indices = (labels[b] != -100).nonzero(as_tuple=True)[0]
                if len(resp_indices) == 0: continue
                pos = resp_indices[torch.randint(0, len(resp_indices), (1,)).item()]
                
                # Sample swap token from top-k of ref_model
                # To avoid extra forwards, we use the logits we already have
                k = min(swap_topk, ref_logits.size(-1))
                logits_pos = max(int(pos.item()) - 1, 0)
                topk_indices = torch.topk(ref_logits[b, logits_pos, :], k).indices
                swap_token = topk_indices[torch.randint(0, k, (1,)).item()]
                swapped_input_ids[b, pos] = swap_token
                swapped_labels[b, pos] = swap_token
            
            # Forward on swapped batch
            pi_logps_swapped = compute_logprobs(policy(swapped_input_ids).logits, swapped_labels)
            # Ref model is frozen, but we still need its logps for the swapped sequence
            with torch.no_grad():
                ref_logps_swapped = compute_logprobs(reference_model(swapped_input_ids).logits, swapped_labels)
            
            r_swapped = beta * (pi_logps_swapped - ref_logps_swapped)
            total_curv_loss += (r_swapped - r_base).pow(2).mean()
            count += 1
            
    return total_curv_loss / count if count > 0 else torch.tensor(0.0).to(device)
