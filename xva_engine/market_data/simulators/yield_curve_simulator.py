from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from datetime import date, timedelta


@dataclass(frozen=True)
class DummyYieldCurveSpec:
    curve_id: str
    currency_id: str
    interp_type: str = "log-lin"
    extrap_type: str = "near"
    day_count_conv: str = "ACT365FIXED"
    compounding_freq: str = "CONTINUOUS"


class YieldCurveDummySimulator:
    """
    Generates dummy yield curves in WIDE format compatible with YieldCurveParser.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def default_maturities_days() -> List[int]:
        return [1, 7, 14, 30, 60, 91, 182, 365, 730, 1825, 3652, 7305, 10957]

    def _curve_shape(self, mats_days: np.ndarray, level: float, slope: float, curv: float) -> np.ndarray:
        t = mats_days / 365.0
        return level + slope * (1.0 - np.exp(-t)) + curv * (t / (1.0 + t))

    def simulate_row(self, spec: DummyYieldCurveSpec, obs: date, mats: List[int]) -> List[object]:
        mats_arr = np.asarray(mats, dtype=float)

        level = float(self.rng.uniform(-0.002, 0.05))
        slope = float(self.rng.uniform(-0.01, 0.04))
        curv = float(self.rng.uniform(-0.01, 0.02))

        y = self._curve_shape(mats_arr, level, slope, curv)
        y += 2e-4 * self.rng.standard_normal(size=y.shape)  # ~2bps noise

        base = [
            "Yield", spec.curve_id, obs.strftime("%d/%m/%Y"), spec.currency_id,
            spec.interp_type, spec.extrap_type, spec.day_count_conv, spec.compounding_freq
        ]
        return base + [int(x) for x in mats_arr.tolist()] + [float(x) for x in y.tolist()]

    def simulate_dataset(
        self,
        specs: List[DummyYieldCurveSpec],
        start: date,
        n_days: int,
        step_days: int = 1,
        maturities_days: List[int] | None = None,
    ) -> pd.DataFrame:
        mats = maturities_days or self.default_maturities_days()
        rows = []
        for sp in specs:
            for k in range(0, n_days, step_days):
                rows.append(self.simulate_row(sp, start + timedelta(days=k), mats))
        return pd.DataFrame(rows)

    @staticmethod
    def save_wide_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False, header=False)
