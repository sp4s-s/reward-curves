"""Script to generate paper-ready figures and tables from experiment results."""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from pathlib import Path

from curvature_dpo.utils.analysis import (
    compute_correlations, 
    compute_pareto_frontier, 
    matched_pref_acc_comparison
)

def permutation_test(x: np.ndarray, y: np.ndarray, n_perms: int = 5000) -> float:
    """Computes p-value for Spearman correlation using permutation test."""
    obs_rho, _ = stats.spearmanr(x, y)
    count = 0
    y_perm = y.copy()
    for _ in range(n_perms):
        np.random.shuffle(y_perm)
        perm_rho, _ = stats.spearmanr(x, y_perm)
        if abs(perm_rho) >= abs(obs_rho):
            count += 1
    return count / n_perms

def pooled_rho_random_effects(rhos: List[float], ns: List[int]) -> Dict[str, float]:
    """Fisher z-transformation pooling for random effects."""
    zs = [0.5 * np.log((1 + r) / (1 - r)) for r in rhos]
    weights = [n - 3 for n in ns]
    pooled_z = np.average(zs, weights=weights)
    pooled_rho = (np.exp(2 * pooled_z) - 1) / (np.exp(2 * pooled_z) + 1)
    return {"pooled_rho": float(pooled_rho)}

def load_run_data(run_dir: Path) -> pd.DataFrame:
    """Loads all metrics for a single run."""
    # 1. Load training.diagnostics
    train_metrics = []
    train_path = run_dir / "train.jsonl"
    if train_path.exists():
        with open(train_path, "r") as f:
            for line in f:
                train_metrics.append(json.loads(line))
    
    # 2. Load eval metrics (these are logged to WandB, but for local analysis we can look at the jsonl or re-extract)
    # For now, let's assume we have a summary.json or similar
    return pd.DataFrame(train_metrics)

def generate_figure_1(dfs: List[pd.DataFrame], out_path: Path):
    """C_bar and Delta over time."""
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(dfs):
        sns.lineplot(data=df, x="step", y="eval_curv/Q_topk_mean", label=f"Run {i} Curv")
        sns.lineplot(data=df, x="step", y="eval_overopt/delta_raw", label=f"Run {i} Delta", linestyle="--")
    plt.title("Curvature and Overoptimization Gap Over Time")
    plt.savefig(out_path)
    plt.close()

def generate_figure_2(df: pd.DataFrame, out_path: Path):
    """Scatter of (C_bar, Delta) with regression."""
    plt.figure(figsize=(8, 8))
    sns.regplot(data=df, x="eval_curv/Q_topk_mean", y="eval_overopt/delta_raw")
    plt.title("Curvature vs Overoptimization Gap")
    plt.savefig(out_path)
    plt.close()

def generate_figure_2_3d(df: pd.DataFrame, out_path: Path):
    """3D Scatter of (step, C_bar, Delta)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["step"], df["eval_curv/Q_topk_mean"], df["eval_overopt/delta_raw"], c=df["step"], cmap='viridis')
    ax.set_xlabel('Step')
    ax.set_ylabel('Curvature')
    ax.set_zlabel('Delta')
    plt.title("3D Loss Landscape Trajectory")
    plt.savefig(out_path)
    plt.close()

def generate_figure_6(df: pd.DataFrame, out_path: Path):
    """Per-position curvature heatmap."""
    # Group by position tertiles
    tertiles = ["early_curv", "mid_curv", "late_curv"]
    data = df[[f"eval_curv/Q_topk_{t}" for t in tertiles]].mean().values.reshape(1, -1)
    plt.figure(figsize=(8, 4))
    sns.heatmap(data, annot=True, xticklabels=["Early", "Mid", "Late"], yticklabels=["Mean Curv"])
    plt.title("Per-Position Curvature Heatmap")
    plt.savefig(out_path)
    plt.close()

def generate_figure_landscape(grid_path: Path, out_path: Path):
    """Contour plot of the 2D loss landscape."""
    if not grid_path.exists():
        return
    grid = np.load(grid_path)
    plt.figure(figsize=(8, 7))
    plt.contourf(grid, levels=20, cmap="RdGy")
    plt.colorbar(label="Loss / Reward")
    plt.title("2D Loss Landscape (Filter-normalized directions)")
    plt.savefig(out_path)
    plt.close()

def generate_figure_3(df: pd.DataFrame, out_path: Path):
    """Pareto frontier of pref-acc vs Delta."""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="eval_pref/pref_acc", y="eval_overopt/delta_raw", hue="run_name")
    # Add Pareto line logic here
    plt.title("Pareto Frontier: Accuracy vs Overoptimization")
    plt.savefig(out_path)
    plt.close()

def generate_table_1(df: pd.DataFrame, out_path: Path):
    """Run inventory with final metrics."""
    summary = df.groupby("run_name").tail(1)[[
        "run_name", "eval_pref/pref_acc", "eval_overopt/delta_raw", 
        "eval_overopt/kl_est", "eval_curv/Q_topk_mean"
    ]]
    summary.to_csv(out_path, index=False)

def main(exp_dir: str):
    exp_path = Path(exp_dir)
    run_dirs = [d for d in exp_path.iterdir() if d.is_dir()]
    
    all_runs_data = []
    for rd in run_dirs:
        df = load_run_data(rd)
        df["run_name"] = rd.name
        all_runs_data.append(df)
        
    combined_df = pd.concat(all_runs_data)
    
    artifacts_dir = exp_path / "paper_artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    generate_figure_1(all_runs_data, artifacts_dir / "fig1_trajectories.png")
    generate_figure_2(combined_df, artifacts_dir / "fig2_scatter.png")
    generate_figure_2_3d(combined_df, artifacts_dir / "fig2_3d_landscape.png")
    generate_figure_3(combined_df, artifacts_dir / "fig3_pareto.png")
    generate_figure_6(combined_df, artifacts_dir / "fig6_position_heatmap.png")
    
    # Generate landscapes for any found .npy files
    for npy in (exp_path / run_dirs[0] / "artifacts").glob("landscape_step_*.npy"):
        step = npy.stem.split("_")[-1]
        generate_figure_landscape(npy, artifacts_dir / f"fig_landscape_step_{step}.png")
        
    generate_table_1(combined_df, artifacts_dir / "table1_inventory.csv")
    
    print(f"Artifacts generated in {artifacts_dir}")
    
    # Upload all artifacts to WandB
    from curvature_dpo.utils.tracking import tracker
    for fig in artifacts_dir.glob("*.png"):
        tracker.log({f"paper/{fig.stem}": tracker._wandb.Image(str(fig))})
    for table in artifacts_dir.glob("*.csv"):
        tracker.save_file(str(table))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.exp_dir)
