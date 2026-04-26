"""Loss landscape visualization (2D interpolation)."""
from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
from typing import Any, List, Dict

@torch.no_grad()
def get_random_direction(model, device: str = "cuda") -> List[torch.Tensor]:
    """Generates a random direction with filter-wise normalization."""
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        # Filter-wise normalization
        if p.dim() <= 1:
            d.fill_(0) # Skip biases/layer-norms to keep it simple or handle differently
        else:
            for f in range(d.size(0)):
                d_f = d[f]
                p_f = p[f]
                d_f.mul_(p_f.norm() / (d_f.norm() + 1e-10))
        direction.append(d)
    return direction

@torch.no_grad()
def compute_2d_landscape(
    policy,
    batch: Dict[str, torch.Tensor],
    loss_fn: Any,
    n_points: int = 11,
    range_val: float = 1.0,
    device: str = "cuda"
) -> np.ndarray:
    """
    Computes loss values on a 2D grid around the current parameters.
    """
    policy.eval()
    orig_params = [p.clone() for p in policy.parameters()]
    dir_x = get_random_direction(policy, device)
    dir_y = get_random_direction(policy, device)
    
    alphas = np.linspace(-range_val, range_val, n_points)
    betas = np.linspace(-range_val, range_val, n_points)
    grid = np.zeros((n_points, n_points))
    
    for i, alpha in enumerate(tqdm(alphas, desc="Landscape Alpha", leave=False)):
        for j, beta in enumerate(betas):
            # Shift parameters: theta = theta_0 + alpha*dx + beta*dy
            for p, p0, dx, dy in zip(policy.parameters(), orig_params, dir_x, dir_y):
                p.copy_(p0 + alpha * dx + beta * dy)
            
            # Compute loss on a single batch (proxy for the whole landscape)
            # In a real run, we might want to average over several batches
            loss = loss_fn(policy, batch)
            grid[i, j] = loss.item()
            
    # Restore original parameters
    for p, p0 in zip(policy.parameters(), orig_params):
        p.copy_(p0)
        
    return grid
