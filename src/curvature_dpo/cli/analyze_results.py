"""Generate figures and tables from completed experiment runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from curvature_dpo.utils.analysis import compute_correlations, compute_pareto_frontier, matched_pref_acc_comparison


# ── Data loading ─────────────────────────────────────────────────────────────

def load_run_metrics(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    train_path = run_dir / "train.jsonl"
    if train_path.exists():
        with open(train_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def permutation_test(x: np.ndarray, y: np.ndarray, n_perms: int = 5000) -> float:
    obs_rho, _ = stats.spearmanr(x, y)
    count = 0
    y_perm = y.copy()
    for _ in range(n_perms):
        np.random.shuffle(y_perm)
        perm_rho, _ = stats.spearmanr(x, y_perm)
        if abs(perm_rho) >= abs(obs_rho):
            count += 1
    return count / n_perms


def pooled_spearman_z(rhos: List[float], ns: List[int]) -> Dict[str, float]:
    """Fisher z-transform pooling for random effects across runs."""
    zs = [0.5 * np.log((1 + r) / (1 - r + 1e-8) + 1e-8) for r in rhos]
    weights = [max(n - 3, 1) for n in ns]
    pooled_z = np.average(zs, weights=weights)
    pooled_rho = (np.exp(2 * pooled_z) - 1) / (np.exp(2 * pooled_z) + 1)
    return {"pooled_rho": float(pooled_rho)}


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_reward_curvature_trajectory(dfs: List[pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(dfs):
        if "eval_curv/Q_topk/mean" in df.columns:
            sns.lineplot(data=df, x="step", y="eval_curv/Q_topk/mean", label=f"Run {i} C̄")
        if "eval_overopt/delta_raw" in df.columns:
            sns.lineplot(data=df, x="step", y="eval_overopt/delta_raw", label=f"Run {i} Δ", linestyle="--")
    plt.xlabel("Training step")
    plt.ylabel("Metric value")
    plt.title("Reward curvature and overoptimisation gap over training")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_curvature_overopt_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    sns.regplot(data=df, x="eval_curv/Q_topk/mean", y="eval_overopt/delta_raw", scatter_kws={"alpha": 0.5})
    plt.xlabel("C̄ (mean reward curvature)")
    plt.ylabel("Δ (overoptimisation gap)")
    plt.title("Reward curvature vs overoptimisation gap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_trajectory_3d(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df["step"],
        df.get("eval_curv/Q_topk/mean", np.zeros(len(df))),
        df.get("eval_overopt/delta_raw", np.zeros(len(df))),
        c=df["step"], cmap="viridis", s=20,
    )
    plt.colorbar(sc, ax=ax, label="Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("C̄ curvature")
    ax.set_zlabel("Δ overopt")
    plt.title("Training trajectory in (step, curvature, overopt) space")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pareto_frontier(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    if "run_name" in df.columns:
        sns.scatterplot(data=df, x="eval_pref/pref_acc", y="eval_overopt/delta_raw", hue="run_name", alpha=0.7)
    else:
        sns.scatterplot(data=df, x="eval_pref/pref_acc", y="eval_overopt/delta_raw", alpha=0.7)
    plt.xlabel("Preference accuracy")
    plt.ylabel("Δ (overoptimisation gap)")
    plt.title("Pareto frontier: preference accuracy vs overoptimisation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_position_curvature_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in ["eval_curv/Q_topk/early_curv", "eval_curv/Q_topk/mid_curv", "eval_curv/Q_topk/late_curv"] if c in df.columns]
    if not cols:
        return
    data = df[cols].mean().values.reshape(1, -1)
    labels = [c.split("/")[-1].replace("_curv", "") for c in cols]
    plt.figure(figsize=(8, 3))
    sns.heatmap(data, annot=True, fmt=".4f", xticklabels=labels, yticklabels=["Mean C̄"])
    plt.title("Per-position reward curvature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loss_landscape(grid_path: Path, out_path: Path) -> None:
    if not grid_path.exists():
        return
    grid = np.load(grid_path)
    plt.figure(figsize=(8, 7))
    plt.contourf(grid, levels=20, cmap="RdGy")
    plt.colorbar(label="Loss")
    plt.title(f"2D loss landscape — {grid_path.stem}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Tables ────────────────────────────────────────────────────────────────────

def export_metrics_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in [
        "run_name", "eval_pref/pref_acc", "eval_overopt/delta_raw",
        "eval_curv/Q_topk/mean",
    ] if c in df.columns]
    summary = df.dropna(subset=[c for c in cols if c != "run_name"])
    if "run_name" in summary.columns:
        summary = summary.groupby("run_name").tail(1)[cols]
    summary.to_csv(out_path, index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(exp_dir: str) -> None:
    exp_path = Path(exp_dir)
    run_dirs = sorted(d for d in exp_path.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found in {exp_path}")
        return

    all_dfs: List[pd.DataFrame] = []
    for rd in run_dirs:
        df = load_run_metrics(rd)
        if not df.empty:
            df["run_name"] = rd.name
            all_dfs.append(df)

    if not all_dfs:
        print("No training metrics found in any run directory.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out = exp_path / "figures"
    out.mkdir(exist_ok=True)

    plot_reward_curvature_trajectory(all_dfs, out / "curvature_trajectory.png")
    plot_curvature_overopt_scatter(combined, out / "curvature_overopt_scatter.png")
    plot_trajectory_3d(combined, out / "trajectory_3d.png")
    plot_pareto_frontier(combined, out / "pareto_frontier.png")
    plot_position_curvature_heatmap(combined, out / "position_curvature_heatmap.png")

    for npy in run_dirs[0].glob("artifacts/landscape_step_*.npy"):
        step = npy.stem.split("_")[-1]
        plot_loss_landscape(npy, out / f"landscape_step_{step}.png")

    export_metrics_table(combined, out / "metrics_table.csv")
    print(f"Figures written to {out}")

    # Upload to wandb if a run is active.
    try:
        import wandb
        if wandb.run is not None:
            for fig in sorted(out.glob("*.png")):
                wandb.log({f"figures/{fig.stem}": wandb.Image(str(fig))})
            wandb.save(str(out / "metrics_table.csv"), base_path=str(out))
            print("Uploaded figures and table to wandb.")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Path to the experiment output directory")
    args = parser.parse_args()
    main(args.exp_dir)
