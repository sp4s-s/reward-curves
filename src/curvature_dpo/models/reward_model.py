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
            dtype=profile.dtype,
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
        """Efficient batch scoring with chunking to cap peak memory."""
        return self.score_batch_chunked(prompts, responses)

    @torch.no_grad()
    def score_batch_chunked(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        outputs: list[torch.Tensor] = []
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            inputs = self.tokenizer(
                prompts[start:end],
                responses[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            logits = self.model(**inputs).logits.squeeze(-1).detach().cpu()
            outputs.append(logits)
        if not outputs:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(outputs, dim=0)
