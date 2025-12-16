from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np


InterpType = Literal["lin", "log-lin", "cubic"]
Compounding = Literal["CONTINUOUS", "ANNUAL", "SEMIANNUAL", "QUARTERLY", "DAILY"]


@dataclass(frozen=True)
class YieldCurveMeta:
    curve_type: str
    curve_id: str
    observation_date: str  # keep string for now; you can convert to datetime later
    currency: str
    interpolation: str
    extrapolation: str
    day_count: str
    compounding: str


class YieldCurve:
    """
    Simple yield curve object (zero-rates vs maturity in days).

    This is the "passerelle" object consumed by models/pricers:
    - query zero rate by maturity
    - query discount factor (continuous compounding for now)

    Notes
    -----
    Start simple. Later you can:
    - support multiple compounding conventions
    - build from instruments (OIS swaps, deposits, FRAs, etc.)
    - add bumping, sensitivities, curve hierarchies, multi-curve, etc.
    """

    def __init__(
        self,
        meta: YieldCurveMeta,
        maturities_days: np.ndarray,
        zero_rates: np.ndarray,
        *,
        allow_extrapolation: bool = True,
    ):
        maturities_days = np.asarray(maturities_days, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)

        if maturities_days.ndim != 1 or zero_rates.ndim != 1:
            raise ValueError("maturities_days and zero_rates must be 1D arrays.")
        if maturities_days.size != zero_rates.size:
            raise ValueError("maturities_days and zero_rates must have same length.")
        if maturities_days.size < 2:
            raise ValueError("Need at least 2 pillars to interpolate a curve.")

        # sort by maturity
        idx = np.argsort(maturities_days)
        self.meta = meta
        self.maturities_days = maturities_days[idx]
        self.zero_rates = zero_rates[idx]
        self.allow_extrapolation = allow_extrapolation

    def _interp_lin(self, x: float) -> float:
        xs = self.maturities_days
        ys = self.zero_rates

        if not self.allow_extrapolation:
            if x < xs[0] or x > xs[-1]:
                raise ValueError(f"Requested maturity {x} outside curve range [{xs[0]}, {xs[-1]}].")

        # numpy.interp extrapolates flat outside bounds, so do explicit linear extrap if allowed
        if x <= xs[0]:
            # linear extrap
            return ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0])
        if x >= xs[-1]:
            return ys[-2] + (ys[-1] - ys[-2]) * (x - xs[-2]) / (xs[-1] - xs[-2])

        return float(np.interp(x, xs, ys))

    def zero_rate(self, maturity_days: float) -> float:
        """
        Return interpolated zero rate at maturity (in days).
        """
        # For now: only linear. You can extend to log-linear/cubic later.
        return self._interp_lin(float(maturity_days))

    def df(self, maturity_days: float) -> float:
        """
        Discount factor using continuous compounding (default).
        """
        t = float(maturity_days) / 365.0
        r = self.zero_rate(maturity_days)
        return float(np.exp(-r * t))
