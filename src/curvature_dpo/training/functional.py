"""DPO and curvature-regularized DPO loss functions."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of per-token log-probs for non-masked positions."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    return -loss.view(shift_labels.shape).sum(-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss, chosen_rewards, rejected_rewards, accuracy


@torch.no_grad()
def sample_swap_candidates(
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    n_positions: int,
    n_swaps: int,
    swap_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample swap token indices and positions from reference logits.

    Returns candidates (batch, n_positions, n_swaps) and positions (batch, n_positions),
    both on the same device as ref_logits. Caller should free ref_logits after this call.
    """
    batch_size = labels.shape[0]
    device = ref_logits.device
    k = min(swap_topk, ref_logits.size(-1))

    candidates = torch.zeros(batch_size, n_positions, n_swaps, dtype=torch.long, device=device)
    positions = torch.zeros(batch_size, n_positions, dtype=torch.long, device=device)

    for b in range(batch_size):
        resp_idx = (labels[b] != -100).nonzero(as_tuple=True)[0]
        if len(resp_idx) == 0:
            continue
        sampled = resp_idx[torch.randint(0, len(resp_idx), (n_positions,), device=device)]
        for pi, pos in enumerate(sampled):
            positions[b, pi] = pos
            logits_pos = max(int(pos.item()) - 1, 0)
            topk = torch.topk(ref_logits[b, logits_pos, :], k).indices
            candidates[b, pi, :] = topk[torch.randint(0, k, (n_swaps,), device=device)]

    return candidates, positions


def curvature_loss(
    policy,
    reference_model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    pi_logps_base: torch.Tensor,
    ref_logps_base: torch.Tensor,
    swap_candidates: torch.Tensor,
    swap_positions: torch.Tensor,
    beta: float = 0.1,
    device: str = "cuda",
    ref_device: str | None = None,
) -> torch.Tensor:
    """Curvature regularisation term.

    Penalises squared implicit-reward change under single-token swaps sampled
    from the reference distribution. swap_candidates / swap_positions are
    pre-computed (see sample_swap_candidates) so the full reference logit
    tensor does not need to be held in memory during this call.
    """
    ref_device = ref_device or device
    batch_size = input_ids.shape[0]
    n_positions = swap_candidates.shape[1]
    n_swaps = swap_candidates.shape[2]

    r_base = beta * (pi_logps_base - ref_logps_base)

    total: torch.Tensor = torch.tensor(0.0, device=device)
    count = 0

    for pi in range(n_positions):
        for si in range(n_swaps):
            swapped_ids = input_ids.clone()
            swapped_labels = labels.clone()
            for b in range(batch_size):
                pos = int(swap_positions[b, pi].item())
                tok = int(swap_candidates[b, pi, si].item())
                swapped_ids[b, pos] = tok
                swapped_labels[b, pos] = tok

            pi_logps_swapped = compute_logprobs(
                policy(swapped_ids).logits,
                swapped_labels,
            )
            with torch.no_grad():
                ref_logps_swapped = compute_logprobs(
                    reference_model(swapped_ids.to(ref_device)).logits,
                    swapped_labels.to(ref_device),
                ).to(device, non_blocking=True)

            r_swapped = beta * (pi_logps_swapped - ref_logps_swapped)
            total = total + (r_swapped - r_base).pow(2).mean()
            count += 1

    return total / count if count > 0 else torch.tensor(0.0, device=device)
