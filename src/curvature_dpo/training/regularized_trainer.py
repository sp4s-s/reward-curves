"""Curvature-Regularized DPO trainer."""
from __future__ import annotations

import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from curvature_dpo.training.functional import compute_logprobs, dpo_loss, curvature_loss
from curvature_dpo.training.runtime import (
    autocast_context,
    count_tokens,
    create_train_loader,
    move_batch,
    optimizer_step_ready,
)
from curvature_dpo.training.diagnostics import (
    clone_trainable_params,
    dpo_batch_metrics,
    gradient_cosine,
    parameter_norm,
    update_norm,
)
from curvature_dpo.eval.eval_protocol import run_checkpoint_eval
from curvature_dpo.models.reward_model import GoldRewardModel
from curvature_dpo.data.probe_set import build_probe_set
from curvature_dpo.utils.checkpoint import CheckpointManager, load_checkpoint
from curvature_dpo.utils.dashboard import DashboardWriter
from curvature_dpo.utils.logging import get_logger, JsonlMetricWriter
from curvature_dpo.utils.telemetry import GpuTelemetry, profiler_context
from curvature_dpo.utils.tracking import tracker


def train_curv_dpo(
    policy,
    reference_model,
    tokenizer,
    train_dataset,
    eval_prefs_dataset,
    eval_gen_dataset,
    probe_dataset,
    cfg,
    device="cuda",
    ref_device: str | None = None,
    resume_ckpt=None,
):
    logger = get_logger("curvature_dpo.curv_dpo")
    logger.info("Starting Curvature-Regularized DPO: %s (lambda=%.3f)", cfg.run_name, cfg.curv_lambda)
    ref_device = ref_device or device

    gold_rm = GoldRewardModel(device=device, bf16=cfg.model.bf16)
    probe_set = build_probe_set(probe_dataset, tokenizer)

    train_loader = create_train_loader(train_dataset, cfg)
    optimizer = AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    total_optimizer_steps = int(cfg.total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    optimizer_step = 0
    if resume_ckpt is not None:
        optimizer_step = load_checkpoint(optimizer, scheduler, resume_ckpt)
    micro_step = optimizer_step * int(cfg.grad_accum)

    ckpt_mgr = CheckpointManager(
        cfg.out_dir,
        keep_last_n=int(getattr(cfg, "keep_last_n", 2)),
        keep_best=bool(getattr(cfg, "keep_best", True)),
    )

    policy.train()
    reference_model.eval()
    optimizer.zero_grad(set_to_none=True)
    dashboard = DashboardWriter(cfg, logger)

    with (
        GpuTelemetry(cfg, logger, device) as telemetry,
        JsonlMetricWriter(f"{cfg.out_dir}/train.jsonl") as writer,
        profiler_context(cfg, cfg.out_dir) as prof,
    ):
        pbar = tqdm(total=total_optimizer_steps, desc="Curv-DPO", initial=optimizer_step)

        while optimizer_step < total_optimizer_steps:
            for batch in train_loader:
                if optimizer_step >= total_optimizer_steps:
                    break

                tokens = count_tokens(batch)
                batch = move_batch(batch, device)

                with autocast_context(policy, device):
                    with torch.no_grad():
                        ref_chosen_logits = reference_model(
                            input_ids=batch["chosen_input_ids"].to(ref_device),
                            attention_mask=batch["chosen_attention_mask"].to(ref_device),
                        ).logits
                        ref_chosen_logps = compute_logprobs(
                            ref_chosen_logits, batch["chosen_labels"].to(ref_device)
                        ).to(device)

                        ref_rejected_logits = reference_model(
                            input_ids=batch["rejected_input_ids"].to(ref_device),
                            attention_mask=batch["rejected_attention_mask"].to(ref_device),
                        ).logits
                        ref_rejected_logps = compute_logprobs(
                            ref_rejected_logits, batch["rejected_labels"].to(ref_device)
                        ).to(device)
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

                    l_dpo, c_rew, r_rew, acc = dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps, beta=cfg.beta,
                    )

                    l_curv = curvature_loss(
                        policy, reference_model,
                        batch["chosen_input_ids"], batch["chosen_labels"],
                        pi_logps_base=policy_chosen_logps,
                        ref_logps_base=ref_chosen_logps,
                        ref_logits=ref_chosen_logits.to(device),
                        beta=cfg.beta,
                        n_positions=cfg.curv_n_positions,
                        n_swaps=cfg.curv_n_swaps,
                        swap_topk=cfg.curv_swap_topk,
                        device=device, ref_device=ref_device,
                    )
                    del ref_chosen_logits

                    raw_loss = l_dpo + cfg.curv_lambda * l_curv
                    loss = raw_loss / cfg.grad_accum

                micro_step += 1
                will_update = optimizer_step_ready(micro_step, int(cfg.grad_accum))
                next_step = optimizer_step + 1 if will_update else optimizer_step
                will_log = will_update and (next_step == 1 or next_step % cfg.log_every == 0)

                # Compute gradient cosine before backward while graph is live.
                # autograd.grad does not accumulate into .grad, so backward is unaffected.
                grad_cos = None
                if will_log and cfg.curv_lambda > 0:
                    grad_cos = gradient_cosine(l_dpo, l_curv, policy)

                loss.backward()

                if not will_update:
                    continue

                before_params = clone_trainable_params(policy) if will_log else None
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step = next_step
                pbar.update(1)

                if will_log:
                    extra = dpo_batch_metrics(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps,
                        batch["chosen_labels"], batch["rejected_labels"], cfg.beta,
                    )
                    metrics = {
                        "step": optimizer_step,
                        "epoch": float(micro_step / len(train_loader)),
                        "loss": raw_loss.item(),
                        "l_dpo": l_dpo.item(),
                        "l_curv": l_curv.item(),
                        "grad_cosine/dpo_curv": grad_cos,
                        "grad_norm/pre_clip": float(pre_clip_grad_norm),
                        "param_norm": parameter_norm(policy),
                        "update_norm": update_norm(before_params, policy),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    metrics.update(extra)
                    metrics = telemetry.capture(metrics, optimizer_step, micro_step, tokens=tokens)
                    writer.write(metrics)
                    tracker.log(metrics, step=optimizer_step)
                    dashboard.maybe_update(optimizer_step)

                if optimizer_step > 0 and optimizer_step % cfg.save_every == 0:
                    eval_results = run_checkpoint_eval(
                        policy, reference_model, tokenizer, gold_rm,
                        eval_prefs_dataset, eval_gen_dataset, probe_set,
                        cfg, optimizer_step, device=device,
                    )
                    ckpt_path = ckpt_mgr.save(
                        policy, tokenizer, optimizer, scheduler,
                        optimizer_step, score=eval_results.get("eval_pref/pref_acc"),
                    )
                    tracker.log_artifact(
                        f"checkpoint_step_{optimizer_step}", "model", str(ckpt_path)
                    )

                if prof is not None:
                    prof.step()

    run_checkpoint_eval(
        policy, reference_model, tokenizer, gold_rm,
        eval_prefs_dataset, eval_gen_dataset, probe_set,
        cfg, optimizer_step, device=device,
    )
    final_ckpt = ckpt_mgr.save(policy, tokenizer, optimizer, scheduler, optimizer_step)
    tracker.log_artifact(f"checkpoint_final", "model", str(final_ckpt))
    logger.info("Curvature-Regularized DPO training complete.")
