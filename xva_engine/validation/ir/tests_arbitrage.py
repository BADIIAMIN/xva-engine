from __future__ import annotations

import numpy as np


def _pillars_years(pillars_days: np.ndarray, day_count: float = 365.0) -> np.ndarray:
    return np.asarray(pillars_days, dtype=float) / day_count


def implied_discount_factors(
    rates: np.ndarray, pillars_days: np.ndarray, day_count: float = 365.0
) -> np.ndarray:
    """
    rates: (P, T, K) simulated continuous zero rates Y(t, M_k)
    returns DF: (P, T, K) with DF(t, t+M_k) = exp(-Y * M_k_years)
    """
    M = _pillars_years(pillars_days, day_count)  # (K,)
    return np.exp(-rates * M[None, None, :])


def test_df_monotonicity(df: np.ndarray, tol: float = 0.0) -> dict:
    """
    Static arbitrage screen: DF must be non-increasing in maturity.
    df: (P, T, K)

    tol: allow df_{k+1} <= df_k + tol
    """
    # violation where DF increases with maturity
    diff = df[:, :, 1:] - df[:, :, :-1]  # (P, T, K-1)
    viol = diff > tol
    viol_rate = viol.mean()

    # severity: max increase
    max_increase = diff[viol].max() if np.any(viol) else 0.0

    # time x pillar frequency
    freq_time_pillar = viol.mean(axis=0)  # (T, K-1)

    return {
        "violation_rate": float(viol_rate),
        "max_increase": float(max_increase),
        "freq_time_pillar": freq_time_pillar,
    }


def implied_forwards(
    rates: np.ndarray, pillars_days: np.ndarray, i: int, j: int, day_count: float = 365.0
) -> np.ndarray:
    """
    Forward between pillars i<j:
      F_{i,j}(t) = (Y_j*M_j - Y_i*M_i)/(M_j-M_i)

    returns: (P, T)
    """
    M = _pillars_years(pillars_days, day_count)
    Mi, Mj = M[i], M[j]
    Yi = rates[:, :, i]
    Yj = rates[:, :, j]
    return (Yj * Mj - Yi * Mi) / (Mj - Mi)


def kink_index(rates: np.ndarray) -> np.ndarray:
    """
    Simple curve 'kink' metric: max absolute second difference across pillars.
    rates: (P, T, K)
    returns: (P, T)
    """
    # second finite difference along pillar axis
    d2 = rates[:, :, 2:] - 2.0 * rates[:, :, 1:-1] + rates[:, :, :-2]  # (P,T,K-2)
    return np.max(np.abs(d2), axis=2)


def _interp_rate_at_maturity(
    rates_t: np.ndarray, M: np.ndarray, target: float
) -> np.ndarray:
    """
    rates_t: (P, K) rates at a fixed time t across pillars
    M: (K,) maturity in years
    target: maturity in years
    returns: (P,) interpolated rate at target maturity
    """
    # numpy interp works 1D; do per path
    out = np.empty(rates_t.shape[0], dtype=float)
    for p in range(rates_t.shape[0]):
        out[p] = np.interp(target, M, rates_t[p])
    return out


def df_wedge_one_step(
    rates: np.ndarray,
    time_grid: np.ndarray,
    pillars_days: np.ndarray,
    base_pillar_index: int,
    step_index: int,
    day_count: float = 365.0,
) -> dict:
    """
    Dynamic multiplicative identity check (one-step):
      wedge = log DF(t, t+T) - log DF(t,t+u) - log DF(t+u, t+T)

    where:
      u = time_grid[step_index+1] - time_grid[step_index]
      T = maturity at base_pillar_index (M_T)
      DF(t,t+T) from rate at time t for maturity T
      DF(t,t+u) from interpolated rate at maturity u from curve at time t
      DF(t+u,t+T) from interpolated rate at maturity (T-u) from curve at time t+u

    This is the cleanest implementable pathwise consistency diagnostic for your model.
    """
    rates = np.asarray(rates, dtype=float)
    t = np.asarray(time_grid, dtype=float)
    M = _pillars_years(pillars_days, day_count)

    P, Tn, K = rates.shape
    if step_index >= Tn - 1:
        raise ValueError("step_index must be < len(time_grid)-1")
    if base_pillar_index < 0 or base_pillar_index >= K:
        raise ValueError("invalid base_pillar_index")

    u = t[step_index + 1] - t[step_index]
    TT = M[base_pillar_index]
    if u <= 0:
        raise ValueError("time_grid must be increasing")
    if u >= TT:
        raise ValueError("choose a maturity pillar longer than one time step for wedge test")

    # time t
    rates_t = rates[:, step_index, :]      # (P,K)
    # time t+u
    rates_tu = rates[:, step_index + 1, :] # (P,K)

    # DF(t,t+TT) from pillar
    y_T = rates[:, step_index, base_pillar_index]  # (P,)
    df_long = np.exp(-y_T * TT)

    # DF(t,t+u): interpolate rate at maturity u
    y_u = _interp_rate_at_maturity(rates_t, M, u)
    df_short = np.exp(-y_u * u)

    # DF(t+u,t+TT): interpolate rate at maturity (TT-u) from curve at time t+u
    y_rem = _interp_rate_at_maturity(rates_tu, M, TT - u)
    df_rem = np.exp(-y_rem * (TT - u))

    wedge = np.log(df_long) - np.log(df_short) - np.log(df_rem)  # (P,)

    return {
        "u": float(u),
        "T": float(TT),
        "wedge": wedge,
        "wedge_mean": float(np.mean(wedge)),
        "wedge_p95": float(np.quantile(wedge, 0.95)),
        "wedge_p05": float(np.quantile(wedge, 0.05)),
        "frac_abs_gt_1bp": float(np.mean(np.abs(wedge) > 1e-4)),  # 1bp in log DF units approx
    }
