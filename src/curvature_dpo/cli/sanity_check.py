"""One-time reward model sanity check script."""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from curvature_dpo.models.reward_model import GoldRewardModel
from curvature_dpo.models.policy import load_policy
from curvature_dpo.eval.score import compute_implicit_rewards
from curvature_dpo.utils.logging import get_logger

logger = get_logger("curvature_dpo.sanity")

def run_sanity_check(cfg: Any, device: str = "cuda"):
    logger.info("Initializing sanity check...")
    
    gold_rm = GoldRewardModel(device=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    policy = load_policy(cfg.model_name, device=device)
    ref_model = load_policy(cfg.model_name, device=device)
    
    # 1. Gold RM accuracy on test_prefs
    logger.info("Checking Gold RM accuracy on test_prefs...")
    test_ds = load_dataset(cfg.dataset_name, split="test_prefs").select(range(500))
    
    correct = 0
    for item in test_ds:
        s_chosen = gold_rm.score(item["prompt"], item["chosen"])
        s_rejected = gold_rm.score(item["prompt"], item["rejected"])
        if s_chosen > s_rejected:
            correct += 1
    
    acc = correct / len(test_ds)
    logger.info(f"Gold RM Test Accuracy: {acc:.2f} (Expected: 0.70-0.80)")
    
    # 2. Gold-vs-implicit agreement at step 0
    logger.info("Checking gold-vs-implicit agreement at step 0...")
    prompts = [item["prompt"] for item in test_ds]
    responses = [item["chosen"] for item in test_ds]
    
    r_gold = gold_rm.score_batch(prompts, responses)
    r_implicit = compute_implicit_rewards(policy, ref_model, tokenizer, prompts, responses, beta=cfg.beta, device=device)
    
    # Agreement: how correlated are they?
    correlation = np.corrcoef(r_gold.cpu().numpy(), r_implicit.cpu().numpy())[0, 1]
    logger.info(f"Gold-vs-Implicit Correlation at Step 0: {correlation:.4f}")
    
    # 3. Gold RM distribution on neutral corpus
    # (Placeholder: use first 500 prompts)
    logger.info(f"Gold RM Mean: {r_gold.mean().item():.4f}, Std: {r_gold.std().item():.4f}")
    
    if acc < 0.65:
        logger.error("CRITICAL: Gold RM accuracy is too low. Check bug in RM loading or dataset.")
    
    return {
        "gold_rm_test_acc": acc,
        "step0_correlation": correlation,
        "gold_rm_mean": r_gold.mean().item(),
        "gold_rm_std": r_gold.std().item()
    }

if __name__ == "__main__":
    # Dummy config for testing
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        model_name="EleutherAI/pythia-410m-deduped",
        dataset_name="HuggingFaceH4/ultrafeedback_binarized",
        beta=0.1
    )
    run_sanity_check(cfg)
