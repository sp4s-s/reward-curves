"""Policy model initialization."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpocurv.utils.device import DeviceProfile, get_device_profile


def load_policy(
    model_name: str,
    bf16: bool = True,
    use_flash_attn: bool = True,
    device: str = "cuda",
    profile: DeviceProfile | None = None,
    gradient_checkpointing: bool = True,
):
    """
    Loads the causal LM policy.
    Chooses precision/attention from the active GPU when possible.
    """
    profile = profile or get_device_profile(device, bf16, use_flash_attn)
    kwargs = {
        "torch_dtype": profile.dtype,
        "trust_remote_code": True,
    }
    
    if profile.attn_implementation:
        kwargs["attn_implementation"] = profile.attn_implementation
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.to(profile.device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer
