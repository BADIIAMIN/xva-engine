from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import date, timedelta


@dataclass(frozen=True)
class DummyCreditCurveSpec:
    curve_id: str
    currency_id: str
    interp_type: str = "log-lin"
    extrap_type: str = "flat"
    payment_freq: str = "QUARTERLY"
    day_count: str = "ACT365"
    basis: float = 365.0
    recovery: float = 0.4


class CreditSpreadDummySimulator:
    """
    Generates dummy credit par spread curves in WIDE format:
      [meta(10), maturities(months)..., spreads...]
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def default_maturities_months() -> List[int]:
        # similar to your dataset (6, 12, 24, 36, ..., 3600)
        return [6, 12, 24, 36, 48, 84, 120, 180, 360, 720, 1080, 1800, 3600]

    def _shape(self, m: np.ndarray, level: float, slope: float, curv: float) -> np.ndarray:
        # increasing-ish spreads with maturity; ensure positivity
        t = m / 12.0
        s = level + slope * (1.0 - np.exp(-t)) + curv * (t / (1.0 + t))
        return np.maximum(s, 1e-6)

    def simulate_row(self, spec: DummyCreditCurveSpec, obs: date, mats: List[int]) -> List[object]:
        mats_arr = np.asarray(mats, dtype=float)

        # spreads in decimal (e.g. 0.01 = 100 bps)
        level = float(self.rng.uniform(0.002, 0.02))
        slope = float(self.rng.uniform(0.0, 0.03))
        curv = float(self.rng.uniform(-0.002, 0.01))

        spreads = self._shape(mats_arr, level, slope, curv)
        spreads += 5e-4 * self.rng.standard_normal(size=spreads.shape)  # ~5 bps noise
        spreads = np.maximum(spreads, 1e-6)

        base = [
            "ParCreditSpread",
            spec.curve_id,
            obs.strftime("%d/%m/%Y"),
            spec.currency_id,
            spec.interp_type,
            spec.extrap_type,
            spec.payment_freq,
            spec.day_count,
            float(spec.basis),
            float(spec.recovery),
        ]

        return base + [float(x) for x in mats_arr.tolist()] + [float(x) for x in spreads.tolist()]

    def simulate_dataset(
        self,
        specs: List[DummyCreditCurveSpec],
        start: date,
        n_days: int,
        step_days: int = 1,
        maturities_months: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        mats = maturities_months or self.default_maturities_months()
        rows = []
        for sp in specs:
            for k in range(0, n_days, step_days):
                rows.append(self.simulate_row(sp, start + timedelta(days=k), mats))
        return pd.DataFrame(rows)

    @staticmethod
    def save_wide_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False, header=False)
