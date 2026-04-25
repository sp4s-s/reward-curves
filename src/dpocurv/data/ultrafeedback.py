"""Data loading and tokenization logic for UltraFeedback."""
from __future__ import annotations

from typing import Any, Dict
import torch
from transformers import PreTrainedTokenizer


def as_text(value: Any, role: str | None = None) -> str:
    """Normalize UltraFeedback string/message-list fields to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value.get("text", "")))
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if role is not None and item.get("role") != role:
                    continue
                parts.append(str(item.get("content", item.get("text", ""))))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(value)


def response_text(value: Any) -> str:
    text = as_text(value, role="assistant")
    return text if text else as_text(value)


def tokenize_dpo_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Tokenizes a DPO triplet (prompt, chosen, rejected).
    Masks the prompt from the loss.
    """
    # Template
    prompt_full = f"PROMPT:\n{prompt}\n\nRESPONSE:\n"
    
    # Encode prompt
    prompt_tokens = tokenizer(
        prompt_full,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    prompt_len = len(prompt_tokens["input_ids"])
    
    # Helper to encode response and concatenate
    def encode_response(resp: str):
        # We append EOS manually
        full_text = prompt_full + resp + tokenizer.eos_token
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Create labels: mask prompt with -100
        labels = list(enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        # Also mask padding
        for i in range(len(enc["input_ids"])):
            if enc["input_ids"][i] == tokenizer.pad_token_id:
                labels[i] = -100
        
        return enc["input_ids"], enc["attention_mask"], labels

    c_ids, c_mask, c_labels = encode_response(chosen)
    r_ids, r_mask, r_labels = encode_response(rejected)
    
    return {
        "chosen_input_ids": torch.tensor(c_ids),
        "chosen_attention_mask": torch.tensor(c_mask),
        "chosen_labels": torch.tensor(c_labels),
        "rejected_input_ids": torch.tensor(r_ids),
        "rejected_attention_mask": torch.tensor(r_mask),
        "rejected_labels": torch.tensor(r_labels),
    }


def tokenize_sft_item(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Dict[str, Any]:
    """Tokenizes a single prompt-response for SFT."""
    prompt_full = f"PROMPT:\n{prompt}\n\nRESPONSE:\n"
    full_text = prompt_full + response + tokenizer.eos_token
    
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    
    # Calculate prompt length to mask it
    prompt_enc = tokenizer(prompt_full, add_special_tokens=True)
    prompt_len = len(prompt_enc["input_ids"])
    
    labels = list(enc["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    
    # Mask padding
    for i in range(len(enc["input_ids"])):
        if enc["input_ids"][i] == tokenizer.pad_token_id:
            labels[i] = -100
            
    return {
        "input_ids": torch.tensor(enc["input_ids"]),
        "attention_mask": torch.tensor(enc["attention_mask"]),
        "labels": torch.tensor(labels),
    }


__all__ = ["as_text", "response_text", "tokenize_dpo_pair", "tokenize_sft_item"]
