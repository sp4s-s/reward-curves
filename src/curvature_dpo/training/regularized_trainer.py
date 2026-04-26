"""Curvature-Regularized DPO trainer — 2×T4 optimized."""
from __future__ import annotations

import os

import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from curvature_dpo.training.functional import (
    compute_logprobs, dpo_loss, curvature_loss, sample_swap_candidates,
)
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

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


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
    dual_gpu = device != ref_device

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

    # Dedicated CUDA stream for the frozen reference model (cuda:1).
    # Enqueuing ref work here lets it execute in parallel with policy work on cuda:0.
    ref_stream = torch.cuda.Stream(device=ref_device) if dual_gpu else None

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

                # ── Phase 1: enqueue both ref forwards on ref_stream (cuda:1) ──────
                # Python returns immediately after enqueueing; the GPU executes these
                # while phase 2 runs the policy forwards on cuda:0.
                if ref_stream is not None:
                    ref_stream.wait_stream(torch.cuda.current_stream(device))

                with torch.cuda.stream(ref_stream) if ref_stream else _nullctx():
                    with torch.no_grad():
                        rc_logits = reference_model(
                            input_ids=batch["chosen_input_ids"].to(ref_device),
                            attention_mask=batch["chosen_attention_mask"].to(ref_device),
                        ).logits
                        _ref_chosen_lp = compute_logprobs(rc_logits, batch["chosen_labels"].to(ref_device))
                        _swap_cands, _swap_pos = sample_swap_candidates(
                            rc_logits, batch["chosen_labels"].to(ref_device),
                            cfg.curv_n_positions, cfg.curv_n_swaps, cfg.curv_swap_topk,
                        )
                        del rc_logits

                        rr_logits = reference_model(
                            input_ids=batch["rejected_input_ids"].to(ref_device),
                            attention_mask=batch["rejected_attention_mask"].to(ref_device),
                        ).logits
                        _ref_rejected_lp = compute_logprobs(rr_logits, batch["rejected_labels"].to(ref_device))
                        del rr_logits

                # ── Phase 2: policy forwards on cuda:0 (parallel with phase 1) ────
                with autocast_context(policy, device):
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

                    # ── Sync: wait for ref_stream, then move small result tensors ──
                    if ref_stream is not None:
                        torch.cuda.current_stream(device).wait_stream(ref_stream)

                    # Blocking transfer of small tensors — ref log-probs are [B] floats,
                    # swap candidates are [B, n_pos, n_sw] ints. Both negligible size.
                    ref_chosen_logps = _ref_chosen_lp.to(device)
                    ref_rejected_logps = _ref_rejected_lp.to(device)
                    swap_candidates = _swap_cands.to(device)
                    swap_positions = _swap_pos.to(device)

                    l_dpo, c_rew, r_rew, acc = dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps, beta=cfg.beta,
                    )

                    l_curv = curvature_loss(
                        policy, reference_model,
                        batch["chosen_input_ids"], batch["chosen_labels"],
                        pi_logps_base=policy_chosen_logps,
                        ref_logps_base=ref_chosen_logps,
                        swap_candidates=swap_candidates,
                        swap_positions=swap_positions,
                        beta=cfg.beta,
                        device=device,
                        ref_device=ref_device,
                    )

                    raw_loss = l_dpo + cfg.curv_lambda * l_curv
                    loss = raw_loss / cfg.grad_accum

                micro_step += 1
                will_update = optimizer_step_ready(micro_step, int(cfg.grad_accum))
                next_step = optimizer_step + 1 if will_update else optimizer_step
                will_log = will_update and (next_step == 1 or next_step % cfg.log_every == 0)

                # Gradient cosine uses autograd.grad (retain_graph=True) on the live
                # graph — must be called before loss.backward().
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
                    tracker.log_artifact(f"checkpoint_step_{optimizer_step}", "model", str(ckpt_path))

                if prof is not None:
                    prof.step()

    run_checkpoint_eval(
        policy, reference_model, tokenizer, gold_rm,
        eval_prefs_dataset, eval_gen_dataset, probe_set,
        cfg, optimizer_step, device=device,
    )
    final_ckpt = ckpt_mgr.save(policy, tokenizer, optimizer, scheduler, optimizer_step)
    tracker.log_artifact("checkpoint_final", "model", str(final_ckpt))
    logger.info("Curvature-Regularized DPO training complete.")


class _nullctx:
    """No-op context manager used when ref_stream is None (single GPU)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
