"""SFT Trainer implementation."""
from __future__ import annotations

import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from curvature_dpo.training.runtime import (
    autocast_context,
    count_tokens,
    create_train_loader,
    move_batch,
    optimizer_step_ready,
)
from curvature_dpo.training.metrics import clone_trainable_params, parameter_norm, update_norm
from curvature_dpo.utils.checkpoint import CheckpointManager
from curvature_dpo.utils.dashboard import DashboardWriter
from curvature_dpo.utils.logging import get_logger, JsonlMetricWriter
from curvature_dpo.utils.telemetry import GpuTelemetry, profiler_context
from curvature_dpo.utils.tracking import tracker


def train_sft(
    model,
    tokenizer,
    train_dataset,
    cfg,
    device="cuda"
):
    logger = get_logger("curvature_dpo.sft")
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
    
    ckpt_mgr = CheckpointManager(
        cfg.out_dir,
        keep_last_n=int(getattr(cfg, "keep_last_n", 2)),
        keep_best=bool(getattr(cfg, "keep_best", True)),
        mode="min",  # best = lowest loss for SFT
    )
    
    dashboard = DashboardWriter(cfg, logger)
    
    with (
        GpuTelemetry(cfg, logger, device) as telemetry,
        JsonlMetricWriter(f"{cfg.out_dir}/train.jsonl") as writer,
        profiler_context(cfg, cfg.out_dir) as prof,
    ):
        pbar = tqdm(total=total_optimizer_steps, desc="SFT")
        
        while optimizer_step < total_optimizer_steps:
            for batch in train_loader:
                if optimizer_step >= total_optimizer_steps:
                    break
                
                tokens = count_tokens(batch)
                batch = move_batch(batch, device)
                with autocast_context(model, device):
                    outputs = model(**batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / cfg.grad_accum
                loss.backward()
                micro_step += 1
                
                if not optimizer_step_ready(micro_step, int(cfg.grad_accum)):
                    continue

                next_step = optimizer_step + 1
                will_log = next_step == 1 or next_step % cfg.log_every == 0
                before_params = clone_trainable_params(model) if will_log and cfg.diagnostics.exact_update_norm else None
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step = next_step
                pbar.update(1)

                if will_log:
                    upd_norm = update_norm(before_params, model)
                    metrics = {
                        "step": optimizer_step,
                        "epoch": float(micro_step / max(len(train_loader), 1)),
                        "micro_step": micro_step,
                        "loss": raw_loss.item(),
                        "grad_norm/pre_clip": float(pre_clip_grad_norm),
                        "grad_norm/post_clip": float(min(float(pre_clip_grad_norm), float(cfg.grad_clip))),
                        "param_norm": parameter_norm(model),
                        "update_norm": upd_norm,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    metrics = telemetry.capture(metrics, optimizer_step, micro_step, tokens=tokens)
                    writer.write(metrics)
                    telemetry.write(metrics)
                    telemetry.print_summary(metrics)
                    tracker.log(metrics, step=optimizer_step)
                    dashboard.maybe_update(optimizer_step)
                    pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
                
                if optimizer_step > 0 and optimizer_step % cfg.save_every == 0:
                    ckpt_mgr.save(
                        model, tokenizer, optimizer, scheduler,
                        optimizer_step, score=raw_loss.item()
                    )
                if hasattr(prof, "step"):
                    prof.step()
                
    ckpt_mgr.save(
        model, tokenizer, optimizer, scheduler,
        optimizer_step, score=raw_loss.item() if 'raw_loss' in locals() else None
    )
    dashboard.maybe_update(optimizer_step, force=True)
    logger.info("SFT training complete.")
