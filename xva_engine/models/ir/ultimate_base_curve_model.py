from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class UltimateBaseCurveIrParams:
    """Per-tenor parameters for Ultimate Base Curve simulation."""
    pillars_days: np.ndarray           # shape (K,)
    shift_bp: np.ndarray               # s_k in bp, shape (K,)
    sigma: np.ndarray                  # sigma_k, shape (K,)
    lam: np.ndarray                    # lambda_k, shape (K,)
    delta_floor: float = 1e-8          # floor for mean function (in rate units, not bp)


def _year_fractions_from_days(days: np.ndarray, day_count: float = 365.0) -> np.ndarray:
    return np.asarray(days, dtype=float) / float(day_count)


def ou_exact_step(x: np.ndarray, dt: float, lam: np.ndarray, sigma: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Exact discretization of independent OU components, vectorized across tenors.
    x_{t+dt} = exp(-lam dt) x_t + sqrt( (sigma^2/(2 lam)) (1-exp(-2 lam dt)) ) * z
    """
    expm = np.exp(-lam * dt)
    # handle lam ~ 0 robustly
    eps = 1e-14
    lam_safe = np.where(np.abs(lam) < eps, eps, lam)
    var = (sigma ** 2) * (1.0 - np.exp(-2.0 * lam_safe * dt)) / (2.0 * lam_safe)
    std = np.sqrt(np.maximum(var, 0.0))
    return expm * x + std * z


def driver_variance(t: np.ndarray, lam: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    v^2(t) for each tenor k at each time t.
    returns array shape (T, K) if t is (T,) and params are (K,)
    """
    t = np.asarray(t, dtype=float)
    lam = np.asarray(lam, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    eps = 1e-14
    lam_safe = np.where(np.abs(lam) < eps, eps, lam)
    # broadcast: (T,1) with (K,) -> (T,K)
    tt = t[:, None]
    v2 = (sigma[None, :] ** 2) * (1.0 - np.exp(-2.0 * lam_safe[None, :] * tt)) / (2.0 * lam_safe[None, :])
    return v2


def transform_shifted_exponential(
    x: np.ndarray,              # shape (K,)
    g: np.ndarray,              # shape (K,)
    s: np.ndarray,              # shape (K,)
    v2: np.ndarray,             # shape (K,)
) -> np.ndarray:
    """
    Y = (g+s)*exp(x - 0.5 v^2) - s
    """
    return (g + s) * np.exp(x - 0.5 * v2) - s


class UltimateBaseCurveModel:
    """
    Shifted Exponential Vasicek (per-tenor OU drivers + deterministic mean function).
    Produces continuously compounded zero rates at fixed maturity pillars.
    """

    def __init__(self, params: UltimateBaseCurveIrParams, corr: Optional[np.ndarray] = None):
        self.params = params
        self.K = int(len(params.pillars_days))

        if corr is None:
            self.corr = np.eye(self.K)
        else:
            corr = np.asarray(corr, dtype=float)
            if corr.shape != (self.K, self.K):
                raise ValueError(f"corr must be ({self.K},{self.K})")
            self.corr = corr

        # Cholesky for correlated normals
        self.chol = np.linalg.cholesky(self.corr + 1e-12 * np.eye(self.K))

        # convert shift from bp to rate units
        self.shift = np.asarray(params.shift_bp, dtype=float) * 1e-4
        self.lam = np.asarray(params.lam, dtype=float)
        self.sigma = np.asarray(params.sigma, dtype=float)

    def simulate_paths(
        self,
        time_grid: np.ndarray,          # year fractions shape (T,)
        mean_function: np.ndarray,      # g(t,k) shape (T,K)
        n_paths: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns simulated zero rates Y(t,k) as array shape (n_paths, T, K).
        """
        rng = np.random.default_rng(seed)
        time_grid = np.asarray(time_grid, dtype=float)
        T = len(time_grid)

        if mean_function.shape != (T, self.K):
            raise ValueError(f"mean_function must be (T,K)=({T},{self.K})")

        # precompute v^2(t,k) for transform
        v2_tk = driver_variance(time_grid, self.lam, self.sigma)  # (T,K)

        # state
        x = np.zeros((n_paths, self.K), dtype=float)
        y = np.zeros((n_paths, T, self.K), dtype=float)

        # initial step (t=0)
        y[:, 0, :] = transform_shifted_exponential(x, mean_function[0], self.shift, v2_tk[0])

        for i in range(1, T):
            dt = time_grid[i] - time_grid[i - 1]
            z = rng.standard_normal(size=(n_paths, self.K))
            zc = z @ self.chol.T  # correlate across tenors

            x = ou_exact_step(x, dt, self.lam, self.sigma, zc)
            y[:, i, :] = transform_shifted_exponential(x, mean_function[i], self.shift, v2_tk[i])

        return y
