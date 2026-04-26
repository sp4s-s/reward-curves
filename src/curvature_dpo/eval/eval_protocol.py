"""Checkpoint evaluation pipeline."""
from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from curvature_dpo.eval.curvature import estimate_curvature, compute_bootstrap_ci
from curvature_dpo.eval.landscape import compute_2d_landscape
from curvature_dpo.eval.overopt import compute_overoptimization_gap, compute_goodhart_slope, lexical_diversity
from curvature_dpo.eval.calibration import compute_calibration_curve, compute_perplexity
from curvature_dpo.eval.score import compute_implicit_rewards
from curvature_dpo.models.reward_model import GoldRewardModel
from curvature_dpo.utils.logging import get_logger
from curvature_dpo.utils.tracking import tracker
from curvature_dpo.types import ProbeItem

logger = get_logger("curvature_dpo.eval")


def run_checkpoint_eval(
    policy,
    reference_model,
    tokenizer,
    gold_rm: GoldRewardModel,
    eval_prefs_dataset: Any,
    eval_gen_dataset: Any,
    probe_set: List[ProbeItem],
    cfg: Any,
    step: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    logger.info("Checkpoint eval at step %d", step)
    results = {"step": step, "timestamp": time.time()}

    pref_metrics = evaluate_preference_accuracy(policy, reference_model, tokenizer, eval_prefs_dataset, cfg, device)
    results.update({f"eval_pref/{k}": v for k, v in pref_metrics.items()})

    curv_metrics, swap_table = evaluate_curvature_distribution(policy, reference_model, tokenizer, probe_set, cfg, device)
    results.update({f"eval_curv/{k}": v for k, v in curv_metrics.items()})

    overopt_metrics, gen_df = evaluate_overoptimization(policy, reference_model, tokenizer, gold_rm, eval_gen_dataset, cfg, device)
    results.update({f"eval_overopt/{k}": v for k, v in overopt_metrics.items()})

    art_dir = os.path.join(cfg.out_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    swap_path = os.path.join(art_dir, f"swap_table_step_{step}.parquet")
    gen_path = os.path.join(art_dir, f"generations_step_{step}.parquet")
    pd.DataFrame(swap_table).to_parquet(swap_path)
    gen_df.to_parquet(gen_path)

    tracker.log_artifact(f"swap_table_step_{step}", "eval_table", swap_path)
    tracker.log_artifact(f"generations_step_{step}", "eval_table", gen_path)

    if step % (cfg.save_every * 4) == 0 or step == cfg.total_steps:
        from curvature_dpo.training.runtime import create_train_loader
        eval_batch = next(iter(create_train_loader(eval_prefs_dataset, cfg)))

        def _dpo_loss_fn(model, batch):
            from curvature_dpo.training.functional import compute_logprobs, dpo_loss
            with torch.no_grad():
                chosen_lp = compute_logprobs(
                    model(batch["chosen_input_ids"].to(device), attention_mask=batch["chosen_attention_mask"].to(device)).logits,
                    batch["chosen_labels"].to(device),
                )
                rejected_lp = compute_logprobs(
                    model(batch["rejected_input_ids"].to(device), attention_mask=batch["rejected_attention_mask"].to(device)).logits,
                    batch["rejected_labels"].to(device),
                )
            loss, _, _, _ = dpo_loss(chosen_lp, rejected_lp, chosen_lp.detach(), rejected_lp.detach(), beta=cfg.beta)
            return loss

        landscape_grid = compute_2d_landscape(policy, eval_batch, _dpo_loss_fn, n_points=11, range_val=0.5, device=device)
        landscape_path = os.path.join(art_dir, f"landscape_step_{step}.npy")
        np.save(landscape_path, landscape_grid)
        tracker.log_artifact(f"landscape_step_{step}", "landscape", landscape_path)

    tracker.log(results, step=step)
    return results


def evaluate_preference_accuracy(policy, reference_model, tokenizer, dataset, cfg, device) -> Dict[str, float]:
    from curvature_dpo.training.functional import compute_logprobs

    all_margins = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=cfg.micro_batch_size):
            chosen_lp = compute_logprobs(
                policy(batch["chosen_input_ids"].to(device), attention_mask=batch["chosen_attention_mask"].to(device)).logits,
                batch["chosen_labels"].to(device),
            )
            rejected_lp = compute_logprobs(
                policy(batch["rejected_input_ids"].to(device), attention_mask=batch["rejected_attention_mask"].to(device)).logits,
                batch["rejected_labels"].to(device),
            )
            all_margins.append((chosen_lp - rejected_lp).cpu())

    margins = torch.cat(all_margins)
    pref_acc = float((margins > 0).float().mean().item())
    margin_mean = float(margins.mean().item())

    wt103_samples = ["The quick brown fox jumps over the lazy dog."]
    ppl = compute_perplexity(policy, tokenizer, wt103_samples, device=device)

    return {
        "pref_acc": pref_acc,
        "margin_mean": margin_mean,
        "perplexity": ppl,
    }


def evaluate_curvature_distribution(
    policy, reference_model, tokenizer, probe_set, cfg, device
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    flat_metrics: Dict[str, Any] = {}
    full_swap_table: List[Dict[str, Any]] = []

    for dist_name in ["Q_topk", "Q_unif", "Q_rand-id"]:
        all_samples = []
        pos_curvatures: Dict[str, list] = {"early": [], "mid": [], "late": []}
        len_curvatures: Dict[str, list] = {"short": [], "medium": [], "long": []}

        for i, probe in enumerate(tqdm(probe_set, desc=f"Curv {dist_name}", leave=False)):
            sample = estimate_curvature(
                policy, reference_model, tokenizer, probe,
                n_positions=cfg.curv_n_positions,
                n_swaps=cfg.curv_n_swaps,
                swap_distribution=dist_name,
                beta=cfg.beta,
                device=device,
            )
            sample.probe_idx = i
            all_samples.append(sample)

            resp_len = sample.extra["response_len"]
            for sw in sample.extra["swaps"]:
                full_swap_table.append({
                    "probe_idx": i,
                    "position": sw["position"],
                    "candidate_token": sw["swap_token"],
                    "r_theta_swap": sw["r_theta_swap"],
                    "r_theta_orig": sw["r_theta_orig"],
                    "dist": dist_name,
                    "resp_len": resp_len,
                })

                rel_pos = sw["position"] / max(resp_len, 1)
                tertile = "early" if rel_pos < 0.33 else ("mid" if rel_pos < 0.66 else "late")
                pos_curvatures[tertile].append(sw["diff_sq"])

                if resp_len < 50:
                    len_curvatures["short"].append(sw["diff_sq"])
                elif resp_len < 150:
                    len_curvatures["medium"].append(sw["diff_sq"])
                else:
                    len_curvatures["long"].append(sw["diff_sq"])

        aggregates = [s.aggregate for s in all_samples]
        low, high = compute_bootstrap_ci(aggregates)
        flat_metrics.update({
            f"{dist_name}/mean": float(np.mean(aggregates)),
            f"{dist_name}/median": float(np.median(aggregates)),
            f"{dist_name}/std": float(np.std(aggregates)),
            f"{dist_name}/bootstrap_low": low,
            f"{dist_name}/bootstrap_high": high,
            f"{dist_name}/early_curv": float(np.mean(pos_curvatures["early"])) if pos_curvatures["early"] else 0.0,
            f"{dist_name}/mid_curv": float(np.mean(pos_curvatures["mid"])) if pos_curvatures["mid"] else 0.0,
            f"{dist_name}/late_curv": float(np.mean(pos_curvatures["late"])) if pos_curvatures["late"] else 0.0,
            f"{dist_name}/len_short_curv": float(np.mean(len_curvatures["short"])) if len_curvatures["short"] else 0.0,
            f"{dist_name}/len_medium_curv": float(np.mean(len_curvatures["medium"])) if len_curvatures["medium"] else 0.0,
            f"{dist_name}/len_long_curv": float(np.mean(len_curvatures["long"])) if len_curvatures["long"] else 0.0,
        })

    return flat_metrics, full_swap_table


def evaluate_overoptimization(
    policy, reference_model, tokenizer, gold_rm, dataset, cfg, device
) -> tuple[Dict[str, float], pd.DataFrame]:
    from curvature_dpo.eval.generate import generate_completions
    from curvature_dpo.eval.score import compute_implicit_rewards

    prompts = [item["prompt"] for item in dataset]
    completions = generate_completions(policy, tokenizer, prompts, max_new_tokens=cfg.data.max_response_tokens, device=device)

    r_implicit = compute_implicit_rewards(policy, reference_model, tokenizer, prompts, completions, beta=cfg.beta, device=device)
    r_gold = gold_rm.score_batch(prompts, completions)

    lengths = [len(tokenizer.encode(c)) for c in completions]
    df = pd.DataFrame({
        "prompt": prompts,
        "completion": completions,
        "r_implicit": r_implicit.cpu().numpy(),
        "r_gold": r_gold.cpu().numpy(),
        "length": lengths,
    })

    delta = float((df["r_implicit"] - df["r_gold"]).mean())
    goodhart = compute_goodhart_slope(df["r_implicit"].values, df["r_gold"].values)
    div = lexical_diversity(completions)
    truncated = float((df["length"] >= cfg.data.max_response_tokens).mean())

    from scipy.stats import pearsonr, spearmanr
    pearson_r, _ = pearsonr(df["r_implicit"].values, df["r_gold"].values)
    spearman_r, _ = spearmanr(df["r_implicit"].values, df["r_gold"].values)

    return {
        "delta_raw": delta,
        "goodhart_slope": goodhart,
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "dist_1": div,
        "mean_length": float(df["length"].mean()),
        "p99_length": float(df["length"].quantile(0.99)),
        "truncated_frac": truncated,
    }, df
