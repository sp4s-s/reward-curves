"""Statistical analysis utilities for DPO curvature research."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional

def compute_correlations(
    df: pd.DataFrame, 
    x_col: str = "curv_mean", 
    y_col: str = "delta_raw",
    control_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes Pearson/Spearman correlations and partial correlations.
    """
    x = df[x_col].values
    y = df[y_col].values
    
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    results = {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }
    
    if control_cols:
        for ctrl in control_cols:
            results[f"partial_rho_given_{ctrl}"] = partial_corr(df, x_col, y_col, [ctrl], method="spearman")
            results[f"partial_r_given_{ctrl}"] = partial_corr(df, x_col, y_col, [ctrl], method="pearson")
            
    return results

def partial_corr(df: pd.DataFrame, x: str, y: str, covar: List[str], method: str = "pearson") -> float:
    """
    Computes partial correlation between x and y controlling for covar.
    Simplified implementation using OLS residuals.
    """
    from sklearn.linear_model import LinearRegression
    
    def get_residuals(target, controls):
        lr = LinearRegression()
        lr.fit(df[controls], df[target])
        return df[target] - lr.predict(df[controls])
    
    res_x = get_residuals(x, covar)
    res_y = get_residuals(y, covar)
    
    if method == "pearson":
        return stats.pearsonr(res_x, res_y)[0]
    else:
        return stats.spearmanr(res_x, res_y)[0]

def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, x_maximize: bool = True, y_minimize: bool = True) -> pd.DataFrame:
    """
    Identifies points on the Pareto frontier.
    x_col: usually pref_acc (maximize)
    y_col: usually delta_raw (minimize)
    """
    df = df.copy()
    points = df[[x_col, y_col]].values
    
    # Adjust for maximization/minimization
    if x_maximize:
        points[:, 0] = -points[:, 0]
    if not y_minimize:
        points[:, 1] = -points[:, 1]
        
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_pareto[i]:
            # Keep only points that are NOT dominated by p
            # Dominance: all dimensions better or equal, at least one strictly better
            is_pareto[is_pareto] = np.any(points[is_pareto] < p, axis=1) | np.all(points[is_pareto] == p, axis=1)
            is_pareto[i] = True # p is not dominated by itself
            
    df["is_pareto"] = is_pareto
    return df

def matched_pref_acc_comparison(baseline_df: pd.DataFrame, experimental_df: pd.DataFrame, n_buckets: int = 5) -> Dict[str, Any]:
    """
    Mann-Whitney U on Delta within pref-acc buckets.
    """
    min_acc = min(baseline_df["pref_acc"].min(), experimental_df["pref_acc"].min())
    max_acc = max(baseline_df["pref_acc"].max(), experimental_df["pref_acc"].max())
    
    buckets = np.linspace(min_acc, max_acc, n_buckets + 1)
    results = []
    
    for i in range(n_buckets):
        b_low, b_high = buckets[i], buckets[i+1]
        b_mask = (baseline_df["pref_acc"] >= b_low) & (baseline_df["pref_acc"] < b_high)
        e_mask = (experimental_df["pref_acc"] >= b_low) & (experimental_df["pref_acc"] < b_high)
        
        b_vals = baseline_df[b_mask]["delta_raw"].values
        e_vals = experimental_df[e_mask]["delta_raw"].values
        
        if len(b_vals) > 0 and len(e_vals) > 0:
            u_stat, p_val = stats.mannwhitneyu(b_vals, e_vals)
            results.append({
                "bucket": f"[{b_low:.2f}, {b_high:.2f})",
                "u_stat": u_stat,
                "p_val": p_val,
                "b_mean": np.mean(b_vals),
                "e_mean": np.mean(e_vals),
                "n_b": len(b_vals),
                "n_e": len(e_vals)
            })
            
    return {"bucket_results": results}

def compute_goodhart_slope(r_gold: np.ndarray, r_theta: np.ndarray) -> Dict[str, float]:
    """OLS slope of r_gold regressed on r_theta."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(r_theta, r_gold)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2)
    }
