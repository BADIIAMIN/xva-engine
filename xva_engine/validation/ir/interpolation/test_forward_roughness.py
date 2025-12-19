from __future__ import annotations

import numpy as np

from .interpolation_schemes import (
    interp_zero_linear,
    interp_zero_logdf_linear,
    make_dense_grid,
)
from .metrics import implied_forward_from_zero, forward_roughness


def run_forward_roughness(
    rates_cube: np.ndarray,
    pillars: np.ndarray,
    points_per_interval: int = 8,
) -> dict:
    """
    Compute implied forward curves and a roughness proxy under two interpolation schemes.
    """
    grid = make_dense_grid(pillars, points_per_interval=points_per_interval)

    z_lin = interp_zero_linear(pillars, rates_cube, grid)
    z_logdf = interp_zero_logdf_linear(pillars, rates_cube, grid)

    f_lin = implied_forward_from_zero(z_lin, grid)
    f_logdf = implied_forward_from_zero(z_logdf, grid)

    rough_lin = forward_roughness(f_lin, grid)       # (Npaths, Ntimes)
    rough_logdf = forward_roughness(f_logdf, grid)   # (Npaths, Ntimes)

    return {
        "grid": grid,
        "f_lin": f_lin,
        "f_logdf": f_logdf,
        "rough_lin": rough_lin,
        "rough_logdf": rough_logdf,
    }
