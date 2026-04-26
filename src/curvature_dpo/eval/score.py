"""Scoring logic for evaluation."""
from __future__ import annotations

import torch
from curvature_dpo.training.functional import compute_logprobs


def _model_device(model) -> str:
    return str(next(model.parameters()).device)


@torch.no_grad()
def compute_implicit_rewards(
    policy,
    reference_model,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    beta: float = 0.1,
    device: str = "cuda",
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Computes r_theta(x, y) = beta * [log pi_theta(y|x) - log pi_ref(y|x)]
    for a batch of (prompt, completion) pairs.
    """
    policy.eval()
    reference_model.eval()
    policy_device = _model_device(policy)
    reference_device = _model_device(reference_model)
    
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")

    rewards = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        completion_batch = completions[start : start + batch_size]
        prefixes = [f"PROMPT:\n{p}\n\nRESPONSE:\n" for p in prompt_batch]
        texts = [prefix + completion + tokenizer.eos_token for prefix, completion in zip(prefixes, completion_batch)]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        prefix_enc = tokenizer(prefixes, return_tensors="pt", padding=True, add_special_tokens=True)
        prompt_lens = prefix_enc.attention_mask.sum(dim=1).tolist()

        labels = enc.input_ids.clone()
        for row, prompt_len in enumerate(prompt_lens):
            labels[row, :prompt_len] = -100
        labels = labels.masked_fill(enc.attention_mask == 0, -100)

        policy_inputs = {k: v.to(policy_device, non_blocking=True) for k, v in enc.items()}
        reference_inputs = {k: v.to(reference_device, non_blocking=True) for k, v in enc.items()}
        policy_labels = labels.to(policy_device, non_blocking=True)
        reference_labels = labels.to(reference_device, non_blocking=True)

        logits = policy(**policy_inputs).logits
        ref_logits = reference_model(**reference_inputs).logits

        logp = compute_logprobs(logits, policy_labels)
        ref_logp = compute_logprobs(ref_logits, reference_labels).to(policy_device, non_blocking=True)
        rewards.append(beta * (logp - ref_logp))
        
    return torch.cat(rewards, dim=0)
