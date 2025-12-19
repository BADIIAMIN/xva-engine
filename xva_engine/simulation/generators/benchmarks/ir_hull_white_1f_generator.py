from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HullWhite1FParams:
    a: float
    sigma: float


def _interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x, y)


def simulate_hw1f_curve_paths(
    n_paths: int,
    time_grid_years: np.ndarray,
    pillars_days: np.ndarray,
    df0_curve_times: np.ndarray,     # years
    df0_curve_values: np.ndarray,    # DF(0,t)
    params: HullWhite1FParams,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.asarray(time_grid_years, dtype=float)
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_grid_years must be increasing")
    M = np.asarray(pillars_days, dtype=float) / 365.0
    P, Tn, K = n_paths, len(t), len(M)

    # Build required times: simulation times + times needed for t+M
    required_times = np.unique(np.concatenate([t, (t[:, None] + M[None, :]).ravel()]))
    required_times = required_times[required_times >= 0.0]

    df0_req = _interp_1d(df0_curve_times, df0_curve_values, required_times)
    ln_df = np.log(np.clip(df0_req, 1e-300, None))
    dln = np.gradient(ln_df, required_times, edge_order=1)
    f0_req = -dln  # approx inst forward f(0,t)

    # shift on simulation times
    f0_t = _interp_1d(required_times, f0_req, t)

    # simulate OU x(t): dx=-a x dt + sigma dW
    a, sigma = params.a, params.sigma
    x = np.zeros((P, Tn), dtype=float)
    for i in range(Tn - 1):
        dti = dt[i]
        if a > 1e-12:
            phi = np.exp(-a * dti)
            var = sigma * sigma * (1.0 - phi * phi) / (2.0 * a)
        else:
            phi = 1.0
            var = sigma * sigma * dti
        z = rng.standard_normal(P)
        x[:, i + 1] = phi * x[:, i] + np.sqrt(max(var, 0.0)) * z

    # interpolate x to required_times per path
    x_req = np.empty((P, len(required_times)), dtype=float)
    for p in range(P):
        x_req[p] = _interp_1d(t, x[p], required_times)

    # build DF path on required_times via Euler: DF_{j+1}=DF_j*exp(-(x+f0)*dt)
    df_req_path = np.ones((P, len(required_times)), dtype=float)
    dtreq = np.diff(required_times)
    for j in range(len(required_times) - 1):
        rj = x_req[:, j] + f0_req[j]
        df_req_path[:, j + 1] = df_req_path[:, j] * np.exp(-rj * dtreq[j])

    idx_t = np.searchsorted(required_times, t)
    idx_tM = np.searchsorted(required_times, (t[:, None] + M[None, :]))

    DF_t = df_req_path[:, idx_t]           # (P,T)
    DF_tM = df_req_path[:, idx_tM]         # (P,T,K)
    DF_rel = DF_tM / DF_t[:, :, None]      # DF(t,t+M)

    rates = -np.log(np.clip(DF_rel, 1e-300, None)) / M[None, None, :]
    return rates


# --- Add below your existing code (at bottom of file) ---

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IRHullWhite1FGeneratorConfig:
    n_paths: int = 3000
    seed: Optional[int] = 1234

    # These must be provided by caller OR inferred from a market-data environment
    time_grid_years: Optional[np.ndarray] = None          # shape (T,)
    pillars_days: Optional[np.ndarray] = None             # shape (K,)

    # Initial curve DF(0,t) used to build f0(t)
    df0_curve_times: Optional[np.ndarray] = None          # years, shape (M,)
    df0_curve_values: Optional[np.ndarray] = None         # DF(0,t), shape (M,)

    # HW1F parameters
    a: float = 0.03
    sigma: float = 0.01


@dataclass(frozen=True)
class IRRateCube:
    """
    Standard container to match your generator output conventions.
    """
    zero_rates: np.ndarray        # (Npaths, Ntimes, K)
    time_grid_years: np.ndarray   # (Ntimes,)
    pillars_days: np.ndarray      # (K,)

    @property
    def pillars(self) -> np.ndarray:
        return np.asarray(self.pillars_days, dtype=float) / 365.0


class IRHullWhite1FGenerator:
    """
    HW1F benchmark generator wrapper providing a consistent .generate() API.
    """
    def __init__(self, cfg: IRHullWhite1FGeneratorConfig):
        self.cfg = cfg

    def generate(self) -> IRRateCube:
        cfg = self.cfg
        if cfg.time_grid_years is None:
            raise ValueError("IRHullWhite1FGeneratorConfig.time_grid_years must be provided.")
        if cfg.pillars_days is None:
            raise ValueError("IRHullWhite1FGeneratorConfig.pillars_days must be provided.")
        if cfg.df0_curve_times is None or cfg.df0_curve_values is None:
            raise ValueError("df0_curve_times and df0_curve_values must be provided.")

        params = HullWhite1FParams(a=cfg.a, sigma=cfg.sigma)

        rates = simulate_hw1f_curve_paths(
            n_paths=cfg.n_paths,
            time_grid_years=np.asarray(cfg.time_grid_years, dtype=float),
            pillars_days=np.asarray(cfg.pillars_days, dtype=float),
            df0_curve_times=np.asarray(cfg.df0_curve_times, dtype=float),
            df0_curve_values=np.asarray(cfg.df0_curve_values, dtype=float),
            params=params,
            seed=cfg.seed,
        )

        return IRRateCube(
            zero_rates=rates,
            time_grid_years=np.asarray(cfg.time_grid_years, dtype=float),
            pillars_days=np.asarray(cfg.pillars_days, dtype=float),
        )
