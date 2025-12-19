from __future__ import annotations

import numpy as np


def _check_inputs(pillars: np.ndarray, zero_rates: np.ndarray) -> None:
    if pillars.ndim != 1:
        raise ValueError("pillars must be 1D array of maturities (year fractions).")
    if zero_rates.shape[-1] != pillars.shape[0]:
        raise ValueError("zero_rates last dimension must match pillars length.")
    if not np.all(np.diff(pillars) > 0):
        raise ValueError("pillars must be strictly increasing.")


def discount_factors_from_zero(z: np.ndarray, pillars: np.ndarray) -> np.ndarray:
    """
    Compute DF(0,T) from zero rates z(T) using continuous compounding: DF = exp(-z*T).
    Supports broadcasting on leading dimensions.
    """
    return np.exp(-z * pillars)


def zero_from_discount_factors(df: np.ndarray, pillars: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    """
    Convert DF(0,T) to zero rates z(T) = -log(DF)/T with safe handling at small T.
    """
    df = np.clip(df, eps, None)
    z = -np.log(df) / np.maximum(pillars, eps)
    # enforce z(0)=0 if pillar includes 0 (rare)
    z = np.where(pillars <= eps, 0.0, z)
    return z


def interp_zero_linear(pillars: np.ndarray, zero_rates: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Linear interpolation directly on zero rates.
    zero_rates shape: (..., K)
    returns shape: (..., G)
    """
    _check_inputs(pillars, zero_rates)
    grid = np.asarray(grid)
    if grid.ndim != 1:
        raise ValueError("grid must be 1D.")
    # np.interp supports 1D y only, so vectorize across leading dims
    flat = zero_rates.reshape(-1, zero_rates.shape[-1])
    out = np.vstack([np.interp(grid, pillars, row) for row in flat])
    return out.reshape(zero_rates.shape[:-1] + (grid.shape[0],))


def interp_zero_logdf_linear(pillars: np.ndarray, zero_rates: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Linear interpolation on log discount factors, then convert back to zero rates.
    This is often more stable than interpolating z(T) directly.
    """
    _check_inputs(pillars, zero_rates)
    df = discount_factors_from_zero(zero_rates, pillars)
    logdf = np.log(np.clip(df, 1e-16, None))

    flat = logdf.reshape(-1, logdf.shape[-1])
    out_logdf = np.vstack([np.interp(grid, pillars, row) for row in flat])
    out_logdf = out_logdf.reshape(logdf.shape[:-1] + (grid.shape[0],))

    out_df = np.exp(out_logdf)
    return zero_from_discount_factors(out_df, grid)


def make_dense_grid(pillars: np.ndarray, points_per_interval: int = 8) -> np.ndarray:
    """
    Create a dense maturity grid by inserting equally spaced points in each pillar interval.
    """
    pillars = np.asarray(pillars, dtype=float)
    if pillars.ndim != 1 or pillars.size < 2:
        raise ValueError("pillars must be 1D with at least 2 points.")
    if points_per_interval < 1:
        raise ValueError("points_per_interval must be >= 1")

    grid = []
    for a, b in zip(pillars[:-1], pillars[1:]):
        grid.append(np.linspace(a, b, points_per_interval + 1, endpoint=False))
    grid.append(np.array([pillars[-1]]))
    return np.unique(np.concatenate(grid))
