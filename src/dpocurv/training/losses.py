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
    pi_logps_base: torch.Tensor,
    ref_logps_base: torch.Tensor,
    ref_logits: torch.Tensor,
    beta: float = 0.1,
    n_positions: int = 2,
    n_swaps: int = 2,
    swap_topk: int = 50,
    device: str = "cuda",
    ref_device: str | None = None,
):
    """
    In-loop curvature regularizer.
    Samples swaps for the 'chosen' completion and penalizes squared implicit-reward changes.
    """
    ref_device = ref_device or device
    batch_size, seq_len = input_ids.shape

    r_base = beta * (pi_logps_base - ref_logps_base)

    total_curv_loss = 0.0
    count = 0

    for _ in range(n_positions):
        for _ in range(n_swaps):
            swapped_input_ids = input_ids.clone()
            swapped_labels = labels.clone()
            for b in range(batch_size):
                resp_indices = (labels[b] != -100).nonzero(as_tuple=True)[0]
                if len(resp_indices) == 0:
                    continue
                pos = resp_indices[torch.randint(0, len(resp_indices), (1,)).item()]

                # Sample swap token from top-k of ref_model logits (already on ref_device)
                k = min(swap_topk, ref_logits.size(-1))
                logits_pos = max(int(pos.item()) - 1, 0)
                topk_indices = torch.topk(ref_logits[b, logits_pos, :], k).indices
                swap_token = topk_indices[torch.randint(0, k, (1,)).item()]
                swapped_input_ids[b, pos] = swap_token.to(device)
                swapped_labels[b, pos] = swap_token.to(device)

            # Policy forward on device (cuda:0)
            pi_logps_swapped = compute_logprobs(
                policy(swapped_input_ids).logits,
                swapped_labels,
            )
            # Reference forward on ref_device (cuda:1 if 2xT4), result moved back
            with torch.no_grad():
                ref_logps_swapped = compute_logprobs(
                    reference_model(swapped_input_ids.to(ref_device)).logits,
                    swapped_labels.to(ref_device),
                ).to(device)

            r_swapped = beta * (pi_logps_swapped - ref_logps_swapped)
            total_curv_loss += (r_swapped - r_base).pow(2).mean()
            count += 1

    return total_curv_loss / count if count > 0 else torch.tensor(0.0, device=device)
