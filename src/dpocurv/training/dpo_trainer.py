"""DPO Trainer implementation."""
from __future__ import annotations

import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dpocurv.training.losses import compute_logprobs, dpo_loss
from dpocurv.training.runtime import (
    autocast_context,
    count_tokens,
    create_train_loader,
    move_batch,
    optimizer_step_ready,
)
from dpocurv.training.metrics import (
    clone_trainable_params,
    dpo_batch_metrics,
    parameter_norm,
    update_norm,
)
from dpocurv.utils.checkpoint import save_checkpoint
from dpocurv.utils.dashboard import DashboardWriter
from dpocurv.utils.logging import get_logger, JsonlMetricWriter
from dpocurv.utils.telemetry import GpuTelemetry, profiler_context
from dpocurv.utils.tracking import tracker


def train_dpo(
    policy,
    reference_model,
    tokenizer,
    train_dataset,
    cfg,
    device="cuda"
):
    logger = get_logger("dpocurv.dpo")
    logger.info(f"Starting DPO training: {cfg.run_name}")
    
    train_loader = create_train_loader(train_dataset, cfg)
    optimizer = AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    total_optimizer_steps = int(cfg.total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    
    policy.train()
    reference_model.eval()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    dashboard = DashboardWriter(cfg, logger)
    
    with (
        GpuTelemetry(cfg, logger, device) as telemetry,
        JsonlMetricWriter(f"{cfg.out_dir}/train.jsonl") as writer,
        profiler_context(cfg, cfg.out_dir) as prof,
    ):
        pbar = tqdm(total=total_optimizer_steps, desc="DPO")
        
        while optimizer_step < total_optimizer_steps:
            for batch in train_loader:
                if optimizer_step >= total_optimizer_steps:
                    break
                
                tokens = count_tokens(batch)
                batch = move_batch(batch, device)
                
                with autocast_context(policy, device):
                    with torch.no_grad():
                        ref_chosen_logits = reference_model(
                            input_ids=batch["chosen_input_ids"],
                            attention_mask=batch["chosen_attention_mask"],
                        ).logits
                        ref_chosen_logps = compute_logprobs(ref_chosen_logits, batch["chosen_labels"])
                        del ref_chosen_logits
                        
                        ref_rejected_logits = reference_model(
                            input_ids=batch["rejected_input_ids"],
                            attention_mask=batch["rejected_attention_mask"],
                        ).logits
                        ref_rejected_logps = compute_logprobs(ref_rejected_logits, batch["rejected_labels"])
                        del ref_rejected_logits

                    policy_chosen_logits = policy(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                    ).logits
                    policy_chosen_logps = compute_logprobs(policy_chosen_logits, batch["chosen_labels"])
                    del policy_chosen_logits

                    policy_rejected_logits = policy(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                    ).logits
                    policy_rejected_logps = compute_logprobs(policy_rejected_logits, batch["rejected_labels"])
                    del policy_rejected_logits
                    
                    raw_loss, c_rew, r_rew, acc = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        beta=cfg.beta,
                    )
                    loss = raw_loss / cfg.grad_accum

                loss.backward()
                micro_step += 1
                
                if not optimizer_step_ready(micro_step, int(cfg.grad_accum)):
                    continue

                next_step = optimizer_step + 1
                will_log = next_step == 1 or next_step % cfg.log_every == 0
                before_params = clone_trainable_params(policy) if will_log and cfg.diagnostics.exact_update_norm else None
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step = next_step
                pbar.update(1)
                
                if will_log:
                    extra = dpo_batch_metrics(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        batch["chosen_labels"],
                        batch["rejected_labels"],
                        cfg.beta,
                    )
                    upd_norm = update_norm(before_params, policy)
                    metrics = {
                        "step": optimizer_step,
                        "epoch": float(micro_step / max(len(train_loader), 1)),
                        "micro_step": micro_step,
                        "loss": raw_loss.item(),
                        "l_dpo": raw_loss.item(),
                        "reward_acc": acc.item(),
                        "accuracy": acc.item(),
                        "error_rate": 1.0 - acc.item(),
                        "chosen_reward": c_rew.mean().item(),
                        "rejected_reward": r_rew.mean().item(),
                        "reward_margin": (c_rew - r_rew).mean().item(),
                        "grad_norm/pre_clip": float(pre_clip_grad_norm),
                        "grad_norm/post_clip": float(min(float(pre_clip_grad_norm), float(cfg.grad_clip))),
                        "param_norm": parameter_norm(policy),
                        "update_norm": upd_norm,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    metrics.update(extra)
                    metrics = telemetry.capture(metrics, optimizer_step, micro_step, tokens=tokens)
                    writer.write(metrics)
                    telemetry.write(metrics)
                    telemetry.print_summary(metrics)
                    tracker.log(metrics, step=optimizer_step)
                    dashboard.maybe_update(optimizer_step)
                    pbar.set_postfix({"acc": f"{metrics['reward_acc']:.2f}", "loss": f"{metrics['loss']:.4f}"})
                
                if optimizer_step > 0 and optimizer_step % cfg.save_every == 0:
                    save_checkpoint(policy, tokenizer, optimizer, scheduler, optimizer_step, cfg.out_dir)
                if hasattr(prof, "step"):
                    prof.step()
                
    save_checkpoint(policy, tokenizer, optimizer, scheduler, optimizer_step, cfg.out_dir)
    dashboard.maybe_update(optimizer_step, force=True)
    logger.info("DPO training complete.")
