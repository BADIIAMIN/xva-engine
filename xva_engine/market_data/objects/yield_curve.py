from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class YieldCurveMeta:
    curve_type: str
    curve_id: str
    observation_date: str   # keep as string for now (e.g. "02/01/2024")
    currency_id: str
    interp_type: str
    extrap_type: str
    day_count_conv: str
    compounding_freq: str


class YieldCurve:
    """
    Engine-facing yield curve object: zero rate interpolation + discount factor.
    """

    def __init__(self, meta: YieldCurveMeta, maturity_days: np.ndarray, zero_rates: np.ndarray):
        self.meta = meta
        self.maturity_days = np.asarray(maturity_days, dtype=float)
        self.zero_rates = np.asarray(zero_rates, dtype=float)

        if self.maturity_days.ndim != 1 or self.zero_rates.ndim != 1:
            raise ValueError("maturity_days and zero_rates must be 1D arrays.")
        if self.maturity_days.size != self.zero_rates.size:
            raise ValueError("maturity_days and zero_rates must have same length.")
        if self.maturity_days.size < 2:
            raise ValueError("Need at least 2 curve pillars.")

        idx = np.argsort(self.maturity_days)
        self.maturity_days = self.maturity_days[idx]
        self.zero_rates = self.zero_rates[idx]

    def zero_rate(self, maturity_days: float) -> float:
        """Linear interpolation of zero rates."""
        x = float(maturity_days)
        xs, ys = self.maturity_days, self.zero_rates

        if x <= xs[0]:
            return float(ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0]))
        if x >= xs[-1]:
            return float(ys[-2] + (ys[-1] - ys[-2]) * (x - xs[-2]) / (xs[-1] - xs[-2]))

        return float(np.interp(x, xs, ys))

    def df(self, maturity_days: float) -> float:
        """Discount factor using continuous compounding (simple default)."""
        t = float(maturity_days) / 365.0
        r = self.zero_rate(maturity_days)
        return float(np.exp(-r * t))
