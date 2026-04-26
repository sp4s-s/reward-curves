"""Gold Reward Model loader."""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from curvature_dpo.utils.device import get_device_profile


class GoldRewardModel:
    """
    Wrapper for the external Gold RM.
    Used for evaluation and overoptimization gap analysis.
    """
    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = "cuda",
        bf16: bool = True
    ):
        profile = get_device_profile(device, prefer_bf16=bf16, prefer_flash_attn=False)
        self.device = profile.device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=profile.dtype,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """
        Computes the reward for a single (prompt, response) pair.
        The OpenAssistant RM usually expects input as <prompt> + <eos> + <response>.
        """
        inputs = self.tokenizer(
            prompt,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        logits = self.model(**inputs).logits
        return logits.item()

    @torch.no_grad()
    def score_batch(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """Efficient batch scoring."""
        inputs = self.tokenizer(
            prompts,
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        logits = self.model(**inputs).logits
        return logits.squeeze(-1)
