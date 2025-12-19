from __future__ import annotations

import numpy as np

from .interpolation_schemes import (
    interp_zero_linear,
    interp_zero_logdf_linear,
    make_dense_grid,
)
from .metrics import rms, max_abs


def run_interpolation_sensitivity(
    rates_cube: np.ndarray,
    pillars: np.ndarray,
    points_per_interval: int = 8,
) -> dict:
    """
    Compare interpolated zero curves under:
      - linear on zero rates
      - linear on log DF (converted back to zero)
    rates_cube: (Npaths, Ntimes, Kpillars)
    pillars: (Kpillars,)
    """
    grid = make_dense_grid(pillars, points_per_interval=points_per_interval)

    # Interpolate along maturity dimension for each (path,time)
    z_lin = interp_zero_linear(pillars, rates_cube, grid)
    z_logdf = interp_zero_logdf_linear(pillars, rates_cube, grid)

    diff = z_lin - z_logdf  # (Npaths, Ntimes, G)

    # Metrics per (path,time)
    rms_pt = rms(diff, axis=-1)         # (Npaths, Ntimes)
    maxabs_pt = max_abs(diff, axis=-1)  # (Npaths, Ntimes)

    # Aggregate over paths
    rms_time_med = np.median(rms_pt, axis=0)         # (Ntimes,)
    rms_time_p95 = np.quantile(rms_pt, 0.95, axis=0) # (Ntimes,)
    maxabs_time_p95 = np.quantile(maxabs_pt, 0.95, axis=0)

    return {
        "grid": grid,
        "diff": diff,
        "rms_pt": rms_pt,
        "maxabs_pt": maxabs_pt,
        "rms_time_med": rms_time_med,
        "rms_time_p95": rms_time_p95,
        "maxabs_time_p95": maxabs_time_p95,
    }
