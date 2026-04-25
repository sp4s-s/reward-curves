# Usage Guide: Implicit-Reward Curvature Research

This document provides instructions for running the research pipeline and interpreting the results.

## 1. Setup

Ensure you have created and activated the environment:

```bash
conda env create -f environment.yaml
conda activate implicit_rl
pip install -e .
```

## 2. Configuration

We use **Hydra** for configuration. The base settings are in `configs/config.yaml`.
Experiment-specific settings are in `configs/experiment/*.yaml`.

### Key Parameters:
- `data.oracle_holdout_pct`: Percentage of data held out for the gold RM (default: 0.2).
- `data.probe_size`: Number of items in the fixed probe set (default: 128).
- `curvature.lambda`: Strength of the curvature regularizer (0.0 for baseline DPO).

## 3. Running Experiments

The system automatically saves results in the `runs/` directory with a timestamped and experiment-named folder.

### Step 1: SFT Baseline
Trains the reference policy `π_ref`.
```bash
python src/dpocurv/cli/main.py experiment=sft
```

### Step 2: DPO Baseline
Trains a policy with standard DPO loss.
```bash
python src/dpocurv/cli/main.py experiment=dpo_baseline paths.sft_checkpoint=/path/to/sft/step-000250
```

### Step 3: Curvature-Regularized DPO
Trains a policy with the curvature penalty.
```bash
python src/dpocurv/cli/main.py experiment=dpo_curv paths.sft_checkpoint=/path/to/sft/step-000250 curvature.lambda=0.1
```

Kaggle-friendly single command:
```bash
python src/dpocurv/cli/main.py experiment=dpo_curv wandb.enabled=false

!python src/dpocurv/cli/main.py experiment=dpo_curv profiler.enabled=true wandb.enabled=false
```

With W&B:
```bash
python src/dpocurv/cli/main.py experiment=dpo_curv wandb.enabled=true
```



## 4. Results & Logs

Every run generates an output directory: `runs/YYYY-MM-DD/HH-MM-SS_<experiment_name>/`.

### Directory Contents:
- `run.log`: Detailed console logs with timestamps.
- `metrics.jsonl`: Raw training metrics (loss, reward accuracy, curvature) for every `log_every` steps.
- `gpu_metrics.jsonl`: Live GPU memory/utilization/throughput telemetry for every logged step.
- `run_dashboard.html`: Self-contained visual dashboard for loss, reward, curvature, GPU memory, memory divergence, utilization, throughput, and recent metrics.
- `torch_traces/`: PyTorch profiler traces when `profiler.enabled=true`.
- `step-XXXXXX/`: Model checkpoints in HuggingFace format.
- `.hydra/`: Full record of the config and environment used for the run.

## 5. Curvature Estimation (Offline)

To analyze checkpoints from a completed run:
```bash
# This is typically called via a script in src/dpocurv/cli/eval_all.py (TBD)
```

## 6. Performance Optimization

The code is configured for single-GPU runs across T4, P100, V100, H100, and B200:
- **Device auto-detect**: `device=auto` selects CUDA when available and logs the detected GPU.
- **Precision fallback**: H100/B200/A100 use bf16 when supported; T4/V100 use fp16; P100 falls back to fp32.
- **Attention fallback**: FlashAttention-2 is used only on Ampere-or-newer GPUs when the package is installed; otherwise the run falls back to PyTorch attention.
- **Memory tuning**: lower `training.micro_batch_size` for smaller cards such as T4/P100.
- **Live telemetry**: notebook output prints loss, GPU memory, memory divergence, utilization, and tokens/sec at each `training.log_every`.
- **Visual dashboard**: `dashboard.enabled=true` keeps `run_dashboard.html` updated during training.
- **Deep traces**: enable profiler traces with `profiler.enabled=true profiler.active=3` for TensorBoard/Chrome trace inspection.
