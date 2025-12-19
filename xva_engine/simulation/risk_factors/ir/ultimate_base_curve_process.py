from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class UltimateBaseCurveParams:
    """
    Parameters for per-pillar shifted exponential Vasicek / OU drivers.

    pillars_days: curve pillars in days (K,)
    shift_bp: per-pillar shift in basis points (K,) or scalar
    sigma: per-pillar OU vol (K,) (in driver units, consistent with transform)
    lam: per-pillar mean reversion (K,) (1/year)
    delta_floor: floor applied to mean function g(t,k)
    day_count: used to convert pillars_days -> years
    """
    pillars_days: np.ndarray
    shift_bp: np.ndarray
    sigma: np.ndarray
    lam: np.ndarray
    delta_floor: float = 1e-8
    day_count: float = 365.0


def _as_1d(x: np.ndarray, K: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.full(K, float(x))
    if x.shape != (K,):
        raise ValueError(f"{name} must be shape (K,) or scalar; got {x.shape}")
    return x


def ou_exact_step(x: np.ndarray, dt: float, lam: np.ndarray, sigma: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Exact OU step (vectorized over pillars; x is (...,K), z is (...,K)).
    x_{t+dt} = e^{-lam dt} x_t + sqrt( sigma^2 * (1-e^{-2 lam dt})/(2 lam) ) * z
    """
    eps = 1e-14
    lam_safe = np.where(np.abs(lam) < eps, eps, lam)
    expm = np.exp(-lam * dt)
    var = (sigma ** 2) * (1.0 - np.exp(-2.0 * lam_safe * dt)) / (2.0 * lam_safe)
    std = np.sqrt(np.maximum(var, 0.0))
    return expm * x + std * z


def driver_variance(time_grid: np.ndarray, lam: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Var[X_k(t)] for each t and k. Returns (T,K).
    """
    t = np.asarray(time_grid, dtype=float)
    lam = np.asarray(lam, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    eps = 1e-14
    lam_safe = np.where(np.abs(lam) < eps, eps, lam)
    tt = t[:, None]  # (T,1)
    v2 = (sigma[None, :] ** 2) * (1.0 - np.exp(-2.0 * lam_safe[None, :] * tt)) / (2.0 * lam_safe[None, :])
    return v2


def transform_shifted_exponential(x: np.ndarray, g: np.ndarray, s: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Y = (g+s)*exp(x - 0.5 v^2) - s
    Ensures E[Y]=g when x is OU driver with Var=v^2 and mean 0.
    """
    return (g + s) * np.exp(x - 0.5 * v2) - s


class UltimateBaseCurveProcess:
    """
    Multi-pillar OU drivers with correlated Brownian increments + shifted exponential transform.
    Produces simulated per-pillar continuous zero rates Y(t,k).
    """

    def __init__(self, params: UltimateBaseCurveParams, corr: Optional[np.ndarray] = None):
        self.params = params
        self.K = int(len(np.asarray(params.pillars_days)))

        self.shift_bp = _as_1d(params.shift_bp, self.K, "shift_bp")
        self.sigma = _as_1d(params.sigma, self.K, "sigma")
        self.lam = _as_1d(params.lam, self.K, "lam")

        # convert shifts from bp -> rate units
        self.shift = self.shift_bp * 1e-4

        if corr is None:
            self.corr = np.eye(self.K)
        else:
            corr = np.asarray(corr, dtype=float)
            if corr.shape != (self.K, self.K):
                raise ValueError(f"corr must be shape (K,K)=({self.K},{self.K}); got {corr.shape}")
            self.corr = corr

        # stabilize SPD
        self.chol = np.linalg.cholesky(self.corr + 1e-12 * np.eye(self.K))

    def simulate(
        self,
        time_grid: np.ndarray,        # (T,)
        mean_function: np.ndarray,    # (T,K) g(t,k)
        n_paths: int,
        seed: Optional[int] = None,
        return_driver: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
          y: (n_paths, T, K) simulated zero rates
          x: (n_paths, T, K) driver paths if return_driver=True else None
        """
        rng = np.random.default_rng(seed)

        time_grid = np.asarray(time_grid, dtype=float)
        T = len(time_grid)
        if mean_function.shape != (T, self.K):
            raise ValueError(f"mean_function must be (T,K)=({T},{self.K}), got {mean_function.shape}")

        v2_tk = driver_variance(time_grid, self.lam, self.sigma)  # (T,K)

        x = np.zeros((n_paths, self.K), dtype=float)
        y = np.zeros((n_paths, T, self.K), dtype=float)
        x_store = np.zeros_like(y) if return_driver else None

        # t=0
        y[:, 0, :] = transform_shifted_exponential(x, mean_function[0], self.shift, v2_tk[0])
        if return_driver:
            x_store[:, 0, :] = x

        for i in range(1, T):
            dt = time_grid[i] - time_grid[i - 1]
            z = rng.standard_normal(size=(n_paths, self.K))
            zc = z @ self.chol.T

            x = ou_exact_step(x, dt, self.lam, self.sigma, zc)
            y[:, i, :] = transform_shifted_exponential(x, mean_function[i], self.shift, v2_tk[i])

            if return_driver:
                x_store[:, i, :] = x

        return y, x_store
