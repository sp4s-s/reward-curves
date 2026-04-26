"""Generation logic for evaluation."""
from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


@torch.no_grad()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    n_samples: int = 1,
    max_new_tokens: int = 256,
    device: str = "cuda",
    batch_size: int = 8,
) -> list[str]:
    """
    Generates completions for a list of prompts.
    Returns one sampled completion per prompt.
    """
    model.eval()
    if n_samples != 1:
        raise ValueError("generate_completions currently supports n_samples=1 for efficient evaluation")
    results: list[str] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Template
    formatted_prompts = [f"PROMPT:\n{p}\n\nRESPONSE:\n" for p in prompts]
    
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            prompt_batch = prompts[start : start + batch_size]
            formatted_batch = formatted_prompts[start : start + batch_size]
            inputs = tokenizer(formatted_batch, return_tensors="pt", padding=True).to(device)
            
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                renormalize_logits=True,
                remove_invalid_values=True,
            )

            input_width = inputs.input_ids.shape[-1]
            for row, _prompt in enumerate(prompt_batch):
                seq = outputs[row]
                results.append(tokenizer.decode(seq[input_width:], skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = old_padding_side
        
    return results
