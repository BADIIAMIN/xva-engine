import numpy as np


def pillars_years(pillars_days: np.ndarray, day_count: float = 365.0) -> np.ndarray:
    return np.asarray(pillars_days, dtype=float) / day_count


def discount_factors_from_zero_rates(rates: np.ndarray, pillars_days: np.ndarray, day_count: float = 365.0) -> np.ndarray:
    """
    rates: (P,T,K) continuous zero rates
    DF(t,t+M_k)=exp(-rate*M_k_years)
    """
    M = pillars_years(pillars_days, day_count)
    return np.exp(-rates * M[None, None, :])


def df_monotonicity_violations(df: np.ndarray, tol: float = 0.0) -> dict:
    diff = df[:, :, 1:] - df[:, :, :-1]
    viol = diff > tol
    return {
        "violation_rate": float(viol.mean()),
        "max_increase": float(diff[viol].max()) if np.any(viol) else 0.0,
        "freq_time_pillar": viol.mean(axis=0),  # (T,K-1)
        "violations": viol,
    }


def implied_forward(r_i: np.ndarray, r_j: np.ndarray, M_i: float, M_j: float) -> np.ndarray:
    return (r_j * M_j - r_i * M_i) / (M_j - M_i)


def kink_index(rates: np.ndarray) -> np.ndarray:
    """
    max abs second diff along pillars, returns (P,T)
    """
    d2 = rates[:, :, 2:] - 2.0 * rates[:, :, 1:-1] + rates[:, :, :-2]
    return np.max(np.abs(d2), axis=2)


def interp_rate_at_maturity(rates_t: np.ndarray, M: np.ndarray, target: float) -> np.ndarray:
    """
    rates_t: (P,K) rates across pillars at fixed time
    returns: (P,) interpolated rate at maturity target (years)
    """
    out = np.empty(rates_t.shape[0], dtype=float)
    for p in range(rates_t.shape[0]):
        out[p] = np.interp(target, M, rates_t[p])
    return out


def df_wedge_one_step(
    rates: np.ndarray,
    time_grid_years: np.ndarray,
    pillars_days: np.ndarray,
    base_pillar_index: int,
    step_index: int,
    day_count: float = 365.0,
) -> dict:
    """
    wedge = log DF(t,T) - log DF(t,u) - log DF(t+u,T-u)
    using interpolated rates at maturities u and T-u.
    """
    t = np.asarray(time_grid_years, dtype=float)
    M = pillars_years(pillars_days, day_count)

    P, Tn, K = rates.shape
    u = t[step_index + 1] - t[step_index]
    TT = M[base_pillar_index]
    if u <= 0 or u >= TT:
        raise ValueError("Choose step_index such that 0<u<T")

    rates_t = rates[:, step_index, :]      # (P,K)
    rates_tu = rates[:, step_index + 1, :] # (P,K)

    # DF(t,t+T) from pillar
    y_T = rates[:, step_index, base_pillar_index]
    df_long = np.exp(-y_T * TT)

    # DF(t,t+u) from interpolated rate at maturity u
    y_u = interp_rate_at_maturity(rates_t, M, u)
    df_short = np.exp(-y_u * u)

    # DF(t+u,t+T) from interpolated rate at maturity (T-u)
    y_rem = interp_rate_at_maturity(rates_tu, M, TT - u)
    df_rem = np.exp(-y_rem * (TT - u))

    wedge = np.log(df_long) - np.log(df_short) - np.log(df_rem)

    return {
        "u": float(u),
        "T": float(TT),
        "wedge": wedge,
        "wedge_mean": float(np.mean(wedge)),
        "wedge_p05": float(np.quantile(wedge, 0.05)),
        "wedge_p95": float(np.quantile(wedge, 0.95)),
        "frac_abs_gt_1bp": float(np.mean(np.abs(wedge) > 1e-4)),
    }
