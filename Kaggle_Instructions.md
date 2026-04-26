# Research Project: Implicit Curvature Regularization for DPO

This project has been refactored to align with academic research standards. All modules and diagnostics are now oriented toward measuring and regularizing the implicit reward curvature in Direct Preference Optimization.

## Kaggle Execution Protocol

### 1. Environment Setup
Execute this in a Kaggle cell to install the necessary research dependencies and authenticate with Weights & Biases.

```bash
# Install package in editable mode
pip install -e .

# Login to WandB (Ensure you have your API key ready)
import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb.login(key=user_secrets.get_secret("WANDB_API_KEY"))
```

### 2. Execution (Training & Diagnostic Phase)
Run the main experiment. This script now captures all per-step telemetry and per-checkpoint diagnostics, automatically uploading Parquet files and metrics to WandB.

```bash
python src/curvature_dpo/cli/run_experiment.py \
    experiment=curvature_regularized_dpo \
    training.total_steps=5000 \
    training.save_every=500 \
    wandb.enabled=true
```

### 3. Post-Processing (Analysis & Paper Artifacts)
Once the training is complete, run the analysis protocol to generate figures and pooled statistical tests. These will be uploaded as WandB Images and Artifacts.

```bash
# Replace <RUN_DIR> with the output directory from the previous step
python src/curvature_dpo/cli/analyze_results.py --exp_dir outputs/<RUN_DIR>
```

## Academic Module Mapping
For your reference, the project structure is now organized as follows:

| Old Name | New Research Name | Description |
| :--- | :--- | :--- |
| `dpocurv` | `curvature_dpo` | Core research package. |
| `main.py` | `run_experiment.py` | Orchestration entry point. |
| `curv_dpo_trainer.py` | `regularized_trainer.py` | Implementation of H1/H2 regularizers. |
| `pipeline.py` | `eval_protocol.py` | Standardized evaluation logic. |
| `metrics.py` | `diagnostics.py` | Statistical diagnostic suite. |
| `losses.py` | `functional.py` | Loss function implementations. |
| `generate_paper_artifacts.py` | `analyze_results.py` | Figure and Table generation. |

## Data Persistence
All research artifacts are stored in `outputs/` and mirrored to WandB:
- **`swap_table_step_N.parquet`**: Raw token-swap curvature samples.
- **`generations_step_N.parquet`**: Model rollouts and reward scores.
- **`landscape_step_N.npy`**: 2D loss surface grids.
- **`summary.json`**: Aggregated run metrics.
- **`paper/`**: Publication-ready PNG figures and CSV tables.
