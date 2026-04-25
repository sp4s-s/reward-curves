"""SFT Trainer implementation."""
from __future__ import annotations

import torch
from tqdm import tqdm
import wandb
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dpocurv.training.runtime import autocast_context, create_train_loader, move_batch, optimizer_step_ready
from dpocurv.utils.checkpoint import save_checkpoint
from dpocurv.utils.logging import get_logger, JsonlMetricWriter


def train_sft(
    model,
    tokenizer,
    train_dataset,
    cfg,
    device="cuda"
):
    logger = get_logger("dpocurv.sft")
    logger.info(f"Starting SFT training: {cfg.run_name}")
    
    train_loader = create_train_loader(train_dataset, cfg)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    total_optimizer_steps = int(cfg.total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    
    with JsonlMetricWriter(f"{cfg.out_dir}/metrics.jsonl") as writer:
        pbar = tqdm(total=total_optimizer_steps, desc="SFT")
        
        while optimizer_step < total_optimizer_steps:
            for batch in train_loader:
                if optimizer_step >= total_optimizer_steps:
                    break
                
                batch = move_batch(batch, device)
                with autocast_context(model, device):
                    outputs = model(**batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / cfg.grad_accum
                loss.backward()
                micro_step += 1
                
                if not optimizer_step_ready(micro_step, int(cfg.grad_accum)):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
                        "grad_norm": float(grad_norm),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    writer.write(metrics)
                    if wandb.run is not None:
                        wandb.log(metrics)
                    pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
                
                if optimizer_step > 0 and optimizer_step % cfg.save_every == 0:
                    save_checkpoint(model, tokenizer, optimizer, scheduler, optimizer_step, cfg.out_dir)
                
    save_checkpoint(model, tokenizer, optimizer, scheduler, optimizer_step, cfg.out_dir)
    logger.info("SFT training complete.")
