"""Evaluation helpers for calibration and perplexity."""
from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict, Any

def compute_calibration_curve(margins: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes pref-acc bucketed by margin magnitude.
    """
    abs_margins = np.abs(margins)
    bins = np.linspace(0, abs_margins.max() if len(abs_margins) > 0 else 1.0, n_bins + 1)
    
    bin_accs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (abs_margins >= bins[i]) & (abs_margins < bins[i+1])
        if mask.any():
            bin_accs.append(float(accuracies[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_counts.append(0)
            
    return {
        "bins": bins.tolist(),
        "bin_accs": bin_accs,
        "bin_counts": bin_counts
    }

@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: List[str], device: str = "cuda") -> float:
    """
    Computes perplexity on a list of texts (e.g. WikiText-103).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        if enc.input_ids.size(1) <= 1:
            continue
            
        outputs = model(**enc, labels=enc.input_ids)
        loss = outputs.loss.item()
        n_tokens = enc.input_ids.size(1)
        
        total_loss += loss * n_tokens
        total_tokens += n_tokens
        
    if total_tokens == 0:
        return float("nan")
        
    return float(np.exp(total_loss / total_tokens))
