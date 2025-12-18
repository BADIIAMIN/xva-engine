from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CreditCurveMeta:
    curve_type: str
    curve_id: str
    observation_date: str
    currency_id: str
    interp_type: str
    extrap_type: str
    payment_freq: str
    day_count: str
    basis: float
    recovery: float


class CreditSpreadCurve:
    """
    Engine-facing credit spread curve: interpolates par spreads by maturity.

    Notes
    -----
    For now we store spreads directly (par spreads). Later you can extend to:
    - hazard rate / survival curve bootstrapping
    - CDS pricing conventions (accrual, ISDA, etc.)
    """

    def __init__(self, meta: CreditCurveMeta, maturity_months: np.ndarray, par_spreads: np.ndarray):
        self.meta = meta
        self.maturity_months = np.asarray(maturity_months, dtype=float)
        self.par_spreads = np.asarray(par_spreads, dtype=float)

        if self.maturity_months.ndim != 1 or self.par_spreads.ndim != 1:
            raise ValueError("maturity_months and par_spreads must be 1D arrays.")
        if self.maturity_months.size != self.par_spreads.size:
            raise ValueError("maturity_months and par_spreads must have same length.")
        if self.maturity_months.size < 2:
            raise ValueError("Need at least 2 pillars to interpolate a curve.")

        idx = np.argsort(self.maturity_months)
        self.maturity_months = self.maturity_months[idx]
        self.par_spreads = self.par_spreads[idx]

    def spread(self, maturity_months: float) -> float:
        """Linear interpolation of par spreads."""
        x = float(maturity_months)
        xs, ys = self.maturity_months, self.par_spreads

        if x <= xs[0]:
            return float(ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0]))
        if x >= xs[-1]:
            return float(ys[-2] + (ys[-1] - ys[-2]) * (x - xs[-2]) / (xs[-1] - xs[-2]))

        return float(np.interp(x, xs, ys))
