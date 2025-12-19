from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HistoricalCalibConfig:
    """
    Minimal historical calibration config.
    - lam: fixed mean reversion (global) for now (can be per pillar later)
    - shift_bp: fixed shift (global or per pillar) for now
    - return_horizon_days: sampling horizon for log-returns (e.g. 5 business days)
    """
    lam: float = 0.08
    shift_bp: float = 100.0
    return_horizon_days: int = 5
    day_count: float = 365.0


def estimate_corr_and_sigma_from_history(
    rates_hist: np.ndarray,   # (Nobs, K) historical zero rates in rate units (not bp)
    cfg: HistoricalCalibConfig = HistoricalCalibConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate correlation across pillars + per-pillar sigma from shifted log-returns:
      r_t = ln((Y_{t+h}+s)/(Y_t+s))

    Returns:
      corr: (K,K)
      sigma: (K,)
    """
    Y = np.asarray(rates_hist, dtype=float)
    if Y.ndim != 2:
        raise ValueError("rates_hist must be (Nobs, K)")
    N, K = Y.shape

    s = cfg.shift_bp * 1e-4
    h = int(cfg.return_horizon_days)
    if N <= h:
        raise ValueError("Not enough history for given return horizon")

    # shifted log returns
    num = Y[h:, :] + s
    den = Y[:-h, :] + s
    if np.any(num <= 0) or np.any(den <= 0):
        raise ValueError("Shift too small: shifted rates must be positive for log returns")

    r = np.log(num / den)  # (N-h, K)

    # corr of returns
    corr = np.corrcoef(r, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    # enforce diag=1
    np.fill_diagonal(corr, 1.0)

    # estimate sigma_k from return variance:
    # For small dt, Var[r] ≈ Var[X(t+h)-X(t)].
    # We use dt = h/252 (approx), then map to OU sigma crudely:
    dt = h / 252.0
    var_r = np.var(r, axis=0, ddof=1)
    lam = cfg.lam
    lam_safe = lam if abs(lam) > 1e-14 else 1e-14
    # Var[ΔX] for OU over dt: sigma^2*(1-exp(-2 lam dt))/(2 lam)
    # -> sigma = sqrt( var_r * 2 lam / (1-exp(-2 lam dt)) )
    denom = (1.0 - np.exp(-2.0 * lam_safe * dt))
    denom = max(denom, 1e-12)
    sigma = np.sqrt(np.maximum(var_r, 0.0) * 2.0 * lam_safe / denom)

    return corr, sigma
