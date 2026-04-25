"""DPO Trainer implementation."""
from __future__ import annotations

import torch
from tqdm import tqdm
import wandb
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dpocurv.training.losses import compute_logprobs, dpo_loss
from dpocurv.training.runtime import autocast_context, create_train_loader, move_batch, optimizer_step_ready
from dpocurv.utils.checkpoint import save_checkpoint
from dpocurv.utils.logging import get_logger, JsonlMetricWriter


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
    
    with JsonlMetricWriter(f"{cfg.out_dir}/metrics.jsonl") as writer:
        pbar = tqdm(total=total_optimizer_steps, desc="DPO")
        
        while optimizer_step < total_optimizer_steps:
            for batch in train_loader:
                if optimizer_step >= total_optimizer_steps:
                    break
                
                batch = move_batch(batch, device)
                
                with autocast_context(policy, device):
                    with torch.no_grad():
                        ref_chosen_logits = reference_model(
                            input_ids=batch["chosen_input_ids"],
                            attention_mask=batch["chosen_attention_mask"],
                        ).logits
                        ref_rejected_logits = reference_model(
                            input_ids=batch["rejected_input_ids"],
                            attention_mask=batch["rejected_attention_mask"],
                        ).logits
                        
                        ref_chosen_logps = compute_logprobs(ref_chosen_logits, batch["chosen_labels"])
                        ref_rejected_logps = compute_logprobs(ref_rejected_logits, batch["rejected_labels"])

                    policy_chosen_logits = policy(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                    ).logits
                    policy_rejected_logits = policy(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                    ).logits
                    
                    policy_chosen_logps = compute_logprobs(policy_chosen_logits, batch["chosen_labels"])
                    policy_rejected_logps = compute_logprobs(policy_rejected_logits, batch["rejected_labels"])
                    
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

                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                pbar.update(1)
                
                if optimizer_step == 1 or optimizer_step % cfg.log_every == 0:
                    metrics = {
                        "step": optimizer_step,
                        "micro_step": micro_step,
                        "loss": raw_loss.item(),
                        "reward_acc": acc.item(),
                        "chosen_reward": c_rew.mean().item(),
                        "rejected_reward": r_rew.mean().item(),
                        "reward_margin": (c_rew - r_rew).mean().item(),
                        "grad_norm": float(grad_norm),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    writer.write(metrics)
                    if wandb.run is not None:
                        wandb.log(metrics)
                    pbar.set_postfix({"acc": f"{metrics['reward_acc']:.2f}", "loss": f"{metrics['loss']:.4f}"})
                
                if optimizer_step > 0 and optimizer_step % cfg.save_every == 0:
                    save_checkpoint(policy, tokenizer, optimizer, scheduler, optimizer_step, cfg.out_dir)
                
    save_checkpoint(policy, tokenizer, optimizer, scheduler, optimizer_step, cfg.out_dir)
    logger.info("DPO training complete.")
