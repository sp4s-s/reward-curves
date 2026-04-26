"""Scoring logic for evaluation."""
from __future__ import annotations

import torch
from curvature_dpo.training.losses import compute_logprobs


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
    
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")

    rewards = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        completion_batch = completions[start : start + batch_size]
        prefixes = [f"PROMPT:\n{p}\n\nRESPONSE:\n" for p in prompt_batch]
        texts = [prefix + completion + tokenizer.eos_token for prefix, completion in zip(prefixes, completion_batch)]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_lens = [
            tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids.size(1)
            for prefix in prefixes
        ]

        labels = enc.input_ids.clone()
        for row, prompt_len in enumerate(prompt_lens):
            labels[row, :prompt_len] = -100
        labels = labels.masked_fill(enc.attention_mask == 0, -100)

        logits = policy(**enc).logits
        ref_logits = reference_model(**enc).logits

        logp = compute_logprobs(logits, labels)
        ref_logp = compute_logprobs(ref_logits, labels)
        rewards.append(beta * (logp - ref_logp))
        
    return torch.cat(rewards, dim=0)
