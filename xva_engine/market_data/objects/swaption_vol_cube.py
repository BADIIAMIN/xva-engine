from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class SwaptionVolMeta:
    cube_type: str
    cube_id: str
    observation_date: str
    currency_id: str
    expiry_interp: str
    expiry_extrap: str
    tenor_interp: str
    tenor_extrap: str
    strike_interp: str
    strike_extrap: str
    payment_freq: str
    day_count: str
    basis: str
    calendar: str
    unit: str
    roll_rule: str
    stub_rule: str
    is_strike_smile: str  # keep as string (TRUE/FALSE) from file


class SwaptionVolSlice:
    """
    A single slice for fixed (expiry_months, strike), storing vol vs tenor_days.
    """

    def __init__(self, tenor_days: np.ndarray, vols: np.ndarray):
        tenor_days = np.asarray(tenor_days, dtype=float)
        vols = np.asarray(vols, dtype=float)

        if tenor_days.ndim != 1 or vols.ndim != 1:
            raise ValueError("tenor_days and vols must be 1D arrays.")
        if tenor_days.size != vols.size:
            raise ValueError("tenor_days and vols must have same length.")
        if tenor_days.size < 2:
            raise ValueError("Need at least 2 tenors for interpolation.")

        idx = np.argsort(tenor_days)
        self.tenor_days = tenor_days[idx]
        self.vols = vols[idx]

    def vol(self, tenor_days: float) -> float:
        x = float(tenor_days)
        xs, ys = self.tenor_days, self.vols

        if x <= xs[0]:
            return float(ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0]))
        if x >= xs[-1]:
            return float(ys[-2] + (ys[-1] - ys[-2]) * (x - xs[-2]) / (xs[-1] - xs[-2]))

        return float(np.interp(x, xs, ys))


class SwaptionVolCube:
    """
    Engine-facing container for a swaption volatility cube.

    Stores slices keyed by (expiry_months, strike) where each slice is vol vs tenor_days.

    Notes
    -----
    This is a clean "passerelle" object: you can later upgrade to full 3D interpolation:
      vol(expiry, strike, tenor)
    For now, we provide:
      - exact slice retrieval
      - 1D interpolation in tenor within each slice
    """

    def __init__(self, meta: SwaptionVolMeta):
        self.meta = meta
        self.slices: Dict[Tuple[float, float], SwaptionVolSlice] = {}

    def add_slice(self, expiry_months: float, strike: float, tenor_days: np.ndarray, vols: np.ndarray) -> None:
        self.slices[(float(expiry_months), float(strike))] = SwaptionVolSlice(tenor_days, vols)

    def get_slice(self, expiry_months: float, strike: float) -> SwaptionVolSlice:
        key = (float(expiry_months), float(strike))
        if key not in self.slices:
            raise KeyError(f"No slice found for expiry={expiry_months}, strike={strike}")
        return self.slices[key]
