"""Policy model initialization."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpocurv.utils.device import DeviceProfile, get_device_profile


class DPWrapper(torch.nn.Module):
    """Wrapper to support DataParallel while returning .logits for compatibility."""
    def __init__(self, model):
        super().__init__()
        self.module = model
        self.dp = torch.nn.DataParallel(model)
        self.config = model.config

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = False
        out = self.dp(*args, **kwargs)
        class Output:
            pass
        ret = Output()
        ret.logits = out[0]
        return ret
        
    def parameters(self, recurse=True):
        return self.module.parameters(recurse)
        
    def requires_grad_(self, requires_grad=True):
        self.module.requires_grad_(requires_grad)
        return self
        
    def save_pretrained(self, *args, **kwargs):
        return self.module.save_pretrained(*args, **kwargs)


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
        
    if torch.cuda.device_count() > 1:
        model = DPWrapper(model)
        
    return model, tokenizer
