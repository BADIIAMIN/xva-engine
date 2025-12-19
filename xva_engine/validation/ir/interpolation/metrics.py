from __future__ import annotations

import numpy as np


def rms(x: np.ndarray, axis=None) -> np.ndarray:
    """
    Root Mean Square: sqrt(mean(x^2)).
    """
    x = np.asarray(x)
    return np.sqrt(np.mean(np.square(x), axis=axis))


def max_abs(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x)
    return np.max(np.abs(x), axis=axis)


def implied_forward_from_zero(z_grid: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """
    Compute f(T) = d/dT [T z(T)] on a grid using finite differences.
    z_grid shape: (..., G), T_grid shape: (G,)
    """
    T = T_grid
    tz = z_grid * T  # broadcast
    dtz = np.gradient(tz, T, axis=-1, edge_order=1)
    return dtz  # forward curve on same grid


def forward_roughness(f_grid: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """
    Roughness proxy: integral |d^2 f / dT^2| dT on grid.
    f_grid shape: (..., G)
    Returns shape: leading dims (...)
    """
    T = T_grid
    d2f = np.gradient(np.gradient(f_grid, T, axis=-1, edge_order=1), T, axis=-1, edge_order=1)
    # integrate absolute curvature
    abs_d2f = np.abs(d2f)
    # trapezoidal along T
    return np.trapz(abs_d2f, T, axis=-1)
