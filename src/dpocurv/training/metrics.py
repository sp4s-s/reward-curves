"""Training metric helpers kept off the hot path unless a step is logged."""
from __future__ import annotations

import math
from typing import Iterable

import torch


def _trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def parameter_norm(model) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in _trainable_params(model):
        total += p.detach().float().pow(2).sum()
    return float(total.sqrt().item())


@torch.no_grad()
def clone_trainable_params(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in _trainable_params(model)]


@torch.no_grad()
def update_norm(before: list[torch.Tensor] | None, model) -> float | None:
    if before is None:
        return None
    total = torch.zeros((), device=next(model.parameters()).device)
    for old, new in zip(before, _trainable_params(model)):
        total += (new.detach().float() - old.float()).pow(2).sum()
    return float(total.sqrt().item())


def response_lengths(labels: torch.Tensor) -> torch.Tensor:
    return (labels != -100).sum(dim=-1).float()


def binary_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    scores = torch.cat([pos_scores.detach().float(), neg_scores.detach().float()]).cpu()
    labels = torch.cat([torch.ones_like(pos_scores, dtype=torch.float32), torch.zeros_like(neg_scores, dtype=torch.float32)]).cpu()
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[labels == 1].sum().item()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def dpo_batch_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float,
) -> dict[str, float]:
    delta_w = policy_chosen_logps - ref_chosen_logps
    delta_l = policy_rejected_logps - ref_rejected_logps
    margin = delta_w - delta_l
    chosen_reward = beta * delta_w
    rejected_reward = beta * delta_l
    chosen_len = response_lengths(chosen_labels)
    rejected_len = response_lengths(rejected_labels)
    acc = (margin > 0).float().mean()
    auc = binary_auc(chosen_reward, rejected_reward)
    return {
        "policy/chosen_logp_mean": float(policy_chosen_logps.detach().mean().item()),
        "policy/rejected_logp_mean": float(policy_rejected_logps.detach().mean().item()),
        "ref/chosen_logp_mean": float(ref_chosen_logps.detach().mean().item()),
        "ref/rejected_logp_mean": float(ref_rejected_logps.detach().mean().item()),
        "dpo/delta_w_mean": float(delta_w.detach().mean().item()),
        "dpo/delta_l_mean": float(delta_l.detach().mean().item()),
        "dpo/margin_mean": float(margin.detach().mean().item()),
        "dpo/margin_std": float(margin.detach().std(unbiased=False).item()),
        "dpo/chosen_reward_mean": float(chosen_reward.detach().mean().item()),
        "dpo/rejected_reward_mean": float(rejected_reward.detach().mean().item()),
        "dpo/pref_accuracy": float(acc.item()),
        "dpo/error_rate": float((1.0 - acc).item()),
        "dpo/roc_auc": auc,
        "data/chosen_response_len_mean": float(chosen_len.mean().item()),
        "data/rejected_response_len_mean": float(rejected_len.mean().item()),
    }


def gradient_cosine(loss_a: torch.Tensor, loss_b: torch.Tensor, model) -> float:
    params = _trainable_params(model)
    grads_a = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    grads_b = torch.autograd.grad(loss_b, params, retain_graph=True, allow_unused=True)
    dot = torch.zeros((), device=loss_a.device)
    norm_a = torch.zeros((), device=loss_a.device)
    norm_b = torch.zeros((), device=loss_a.device)
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        ga = ga.detach().float()
        gb = gb.detach().float()
        dot += (ga * gb).sum()
        norm_a += ga.pow(2).sum()
        norm_b += gb.pow(2).sum()
    denom = norm_a.sqrt() * norm_b.sqrt()
    if denom.item() == 0:
        return float("nan")
    return float((dot / denom).item())


__all__ = [
    "binary_auc",
    "clone_trainable_params",
    "dpo_batch_metrics",
    "gradient_cosine",
    "parameter_norm",
    "response_lengths",
    "update_norm",
]
