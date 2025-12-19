from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MeanFunctionConfig:
    """
    Configuration for deterministic mean function g_k(t)=max(f_k(t), delta_floor).
    """
    delta_floor: float = 1e-8
    day_count: float = 365.0


def build_forward_forward_mean_function(
    time_grid: np.ndarray,                # (T,) year fractions
    pillars_days: np.ndarray,             # (K,) in days
    df0: Callable[[float], float],        # discount factor DF(0,t), t in years
    cfg: MeanFunctionConfig = MeanFunctionConfig(),
) -> np.ndarray:
    """
    Computes g(t,k) using forward-forward from DF:
      f_k(t) = -(1/M_k) ln( DF(0,t+M_k)/DF(0,t) )
      g_k(t) = max(f_k(t), delta_floor)

    Returns array (T,K).
    """
    t = np.asarray(time_grid, dtype=float)
    M = np.asarray(pillars_days, dtype=float) / cfg.day_count  # (K,)

    T = len(t)
    K = len(M)
    g = np.zeros((T, K), dtype=float)

    for i, ti in enumerate(t):
        df_t = float(df0(ti))
        # compute DF(0, ti + M_k) for all k
        df_tM = np.array([float(df0(ti + Mk)) for Mk in M], dtype=float)
        f = -(1.0 / M) * np.log(df_tM / df_t)
        g[i, :] = np.maximum(f, cfg.delta_floor)

    return g
