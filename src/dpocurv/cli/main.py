"""Main entry point for orchestration."""
from __future__ import annotations

import hydra
from pathlib import Path
from omegaconf import DictConfig, open_dict

from dpocurv.data.ultrafeedback import as_text, response_text, tokenize_dpo_pair, tokenize_sft_item
from dpocurv.data.splits import get_splits
from dpocurv.models.policy import load_policy
from dpocurv.training.curv_dpo_trainer import train_curv_dpo
from dpocurv.training.dpo_trainer import train_dpo
from dpocurv.training.sft_trainer import train_sft
from dpocurv.utils.artifacts import finalize_run_artifacts, write_error_record, write_run_meta
from dpocurv.utils.device import configure_torch_for_device, get_device_profile
from dpocurv.utils.logging import get_logger
from dpocurv.utils.seed import set_seed
from dpocurv.utils.tracking import tracker


def _flatten_runtime_cfg(cfg: DictConfig, out_dir: Path, device: str) -> None:
    with open_dict(cfg):
        cfg.out_dir = str(out_dir)
        cfg.device = device
        cfg.run_name = cfg.experiment.name
        for key, value in cfg.training.items():
            cfg[key] = value
        cfg.beta = cfg.training.beta
        cfg.curv_lambda = cfg.curvature["lambda"]
        cfg.curv_n_positions = cfg.curvature.n_positions
        cfg.curv_n_swaps = cfg.curvature.n_swaps
        cfg.curv_swap_topk = cfg.curvature.swap_topk


def _validate_cfg(cfg: DictConfig) -> None:
    positive_ints = {
        "training.micro_batch_size": cfg.training.micro_batch_size,
        "training.grad_accum": cfg.training.grad_accum,
        "training.total_steps": cfg.training.total_steps,
        "training.log_every": cfg.training.log_every,
        "training.save_every": cfg.training.save_every,
        "data.sft_train_size": cfg.data.sft_train_size,
        "data.dpo_train_size": cfg.data.dpo_train_size,
        "data.probe_size": cfg.data.probe_size,
    }
    bad = [name for name, value in positive_ints.items() if int(value) <= 0]
    if bad:
        raise ValueError(f"Config values must be positive integers: {', '.join(bad)}")
    if not 0.0 < float(cfg.data.oracle_holdout_pct) < 1.0:
        raise ValueError("data.oracle_holdout_pct must be in (0, 1)")


def _tokenize_split(ds, tokenizer, cfg: DictConfig, stage: str):
    max_len = int(cfg.model.max_seq_len)
    if stage == "sft":
        tokenized = ds.map(
            lambda row: tokenize_sft_item(
                as_text(row.get("prompt")),
                response_text(row.get("chosen")),
                tokenizer,
                max_length=max_len,
            ),
            remove_columns=ds.column_names,
            desc="Tokenizing SFT split",
        )
        columns = ["input_ids", "attention_mask", "labels"]
    else:
        tokenized = ds.map(
            lambda row: tokenize_dpo_pair(
                as_text(row.get("prompt")),
                response_text(row.get("chosen")),
                response_text(row.get("rejected")),
                tokenizer,
                max_length=max_len,
            ),
            remove_columns=ds.column_names,
            desc="Tokenizing DPO split",
        )
        columns = [
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_labels",
        ]
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup paths and logger
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    logger = get_logger("dpocurv.main", log_file=out_dir / "run.log")
    _validate_cfg(cfg)
    profile = get_device_profile(
        cfg.device,
        prefer_bf16=cfg.model.bf16,
        prefer_flash_attn=cfg.model.use_flash_attn,
    )
    configure_torch_for_device(profile)
    _flatten_runtime_cfg(cfg, out_dir, profile.device)
    write_run_meta(cfg, out_dir)

    logger.info(f"Work dir: {cfg.out_dir}")
    logger.info(
        "Device: %s (%s), dtype=%s, attention=%s",
        profile.device,
        profile.name,
        profile.dtype_name,
        profile.attn_implementation or "sdpa/eager",
    )
    set_seed(cfg.seed)
    
    tracker.init(cfg, out_dir, logger)

    failed = False
    try:
        stage = cfg.experiment.stage
        model_source = cfg.model.name
        if stage in {"dpo", "dpo_curv"} and cfg.paths.sft_checkpoint:
            model_source = cfg.paths.sft_checkpoint
        elif stage in {"dpo", "dpo_curv"}:
            logger.warning(
                "No paths.sft_checkpoint provided; training policy from base model with a separate frozen base reference."
            )

        logger.info(f"Loading policy model: {model_source}")
        model, tokenizer = load_policy(
            model_source,
            bf16=cfg.model.bf16,
            use_flash_attn=cfg.model.use_flash_attn,
            device=profile.device,
            profile=profile,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )

        logger.info("Loading splits...")
        splits = get_splits(
            dataset_name=cfg.data.dataset,
            sft_size=cfg.data.sft_train_size,
            dpo_size=cfg.data.dpo_train_size,
            probe_size=cfg.data.probe_size,
            oracle_pct=cfg.data.oracle_holdout_pct,
            seed=cfg.seed
        )

        logger.info(f"Dispatching stage: {stage}")
        
        if stage == "sft":
            train_sft(
                model,
                tokenizer,
                _tokenize_split(splits["sft"], tokenizer, cfg, stage),
                cfg,
                device=cfg.device,
            )
        elif stage == "dpo":
            logger.info(f"Loading frozen reference model: {cfg.model.name}")
            reference_model, _ = load_policy(
                cfg.model.name,
                bf16=cfg.model.bf16,
                use_flash_attn=cfg.model.use_flash_attn,
                device=profile.device,
                profile=profile,
                gradient_checkpointing=False,
            )
            reference_model.requires_grad_(False)
            train_dpo(
                model,
                reference_model,
                tokenizer,
                _tokenize_split(splits["dpo"], tokenizer, cfg, stage),
                cfg,
                device=cfg.device,
            )
        elif stage == "dpo_curv":
            logger.info(f"Loading frozen reference model: {cfg.model.name}")
            reference_model, _ = load_policy(
                cfg.model.name,
                bf16=cfg.model.bf16,
                use_flash_attn=cfg.model.use_flash_attn,
                device=profile.device,
                profile=profile,
                gradient_checkpointing=False,
            )
            reference_model.requires_grad_(False)
            train_curv_dpo(
                model,
                reference_model,
                tokenizer,
                _tokenize_split(splits["dpo"], tokenizer, cfg, stage),
                cfg,
                device=cfg.device,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
        logger.info("Stage execution complete.")
    except Exception as exc:
        failed = True
        logger.exception("Run failed.")
        write_error_record(cfg.out_dir, exc)
        raise
    finally:
        try:
            finalize_run_artifacts(cfg, logger, failed=failed)
        except Exception:
            logger.exception("Failed to finalize artifacts.")
        tracker.finish()

if __name__ == "__main__":
    main()
