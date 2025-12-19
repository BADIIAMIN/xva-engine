from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MeanFunctionSpec:
    """Mean function g_k(t)=max(f_k(t), delta)."""
    delta_floor: float = 1e-8
    day_count: float = 365.0


def build_forward_forward_mean_function(
    time_grid: np.ndarray,            # year fractions (T,)
    pillars_days: np.ndarray,         # (K,) in days
    df0: callable,                    # function t -> DF(0,t) with t in year fractions
    spec: MeanFunctionSpec = MeanFunctionSpec(),
) -> np.ndarray:
    """
    Returns g(t,k) shape (T,K) using f_k(t) = -(1/M_k) ln( DF(0,t+M_k)/DF(0,t) )
    where M_k is pillar maturity expressed in years.
    """
    t = np.asarray(time_grid, dtype=float)
    M = np.asarray(pillars_days, dtype=float) / spec.day_count  # (K,)

    g = np.zeros((len(t), len(M)), dtype=float)
    for i, ti in enumerate(t):
        df_t = df0(ti)
        df_t_M = np.array([df0(ti + Mk) for Mk in M], dtype=float)
        f = -(1.0 / M) * np.log(df_t_M / df_t)
        g[i, :] = np.maximum(f, spec.delta_floor)
    return g
