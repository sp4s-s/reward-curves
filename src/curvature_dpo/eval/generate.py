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
    n_samples: int = 4,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    device: str = "cuda",
    batch_size: int = 8,
) -> list[dict]:
    """
    Generates completions for a list of prompts.
    Returns a list of dicts: {"prompt": str, "completions": list[str]}
    """
    model.eval()
    results = []
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
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_width = inputs.input_ids.shape[-1]
            for row, prompt in enumerate(prompt_batch):
                first = row * n_samples
                last = first + n_samples
                completions = [
                    tokenizer.decode(seq[input_width:], skip_special_tokens=True).strip()
                    for seq in outputs[first:last]
                ]
                results.append({"prompt": prompt, "completions": completions})
    finally:
        tokenizer.padding_side = old_padding_side
        
    return results
