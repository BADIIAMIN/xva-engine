from __future__ import annotations

import numpy as np

from .interpolation_schemes import (
    interp_zero_linear,
    interp_zero_logdf_linear,
    make_dense_grid,
)
from .metrics import rms, max_abs


def run_pillar_density_stress(
    rates_cube: np.ndarray,
    pillars: np.ndarray,
    scheme: str = "logdf",
    points_per_interval: int = 8,
) -> dict:
    """
    Compare curve reconstructed from:
      - full pillar set
      - coarse pillar subset (every other pillar)
    Both are interpolated to a dense grid and compared.

    scheme: "zero" (linear on zero) or "logdf" (linear on log DF)
    """
    pillars = np.asarray(pillars, dtype=float)
    if pillars.size < 4:
        raise ValueError("Need at least 4 pillars for a meaningful coarse subset test.")

    grid = make_dense_grid(pillars, points_per_interval=points_per_interval)

    coarse_idx = np.arange(0, pillars.size, 2)
    if coarse_idx[-1] != pillars.size - 1:
        coarse_idx = np.append(coarse_idx, pillars.size - 1)  # ensure last pillar included

    pillars_coarse = pillars[coarse_idx]
    z_coarse = rates_cube[..., coarse_idx]

    if scheme == "zero":
        z_full_grid = interp_zero_linear(pillars, rates_cube, grid)
        z_coarse_grid = interp_zero_linear(pillars_coarse, z_coarse, grid)
    elif scheme == "logdf":
        z_full_grid = interp_zero_logdf_linear(pillars, rates_cube, grid)
        z_coarse_grid = interp_zero_logdf_linear(pillars_coarse, z_coarse, grid)
    else:
        raise ValueError("scheme must be 'zero' or 'logdf'.")

    diff = z_coarse_grid - z_full_grid  # (Npaths,Ntimes,G)

    rms_pt = rms(diff, axis=-1)
    maxabs_pt = max_abs(diff, axis=-1)

    return {
        "grid": grid,
        "coarse_pillars": pillars_coarse,
        "diff": diff,
        "rms_pt": rms_pt,
        "maxabs_pt": maxabs_pt,
        "rms_time_med": np.median(rms_pt, axis=0),
        "rms_time_p95": np.quantile(rms_pt, 0.95, axis=0),
        "maxabs_time_p95": np.quantile(maxabs_pt, 0.95, axis=0),
    }
