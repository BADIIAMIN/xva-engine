from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import date, timedelta
import random


@dataclass(frozen=True)
class DummyCurveSpec:
    curve_id: str
    currency: str
    interpolation: str = "lin"
    extrapolation: str = "round"
    day_count: str = "ACT365FIXED"
    compounding: str = "CONTINUOUS"


class YieldCurveDummySimulator:
    """
    Generate synthetic yield curve data in the SAME row-based CSV format
    that the parser expects.

    Output row format:
      [Yield, curve_id, dd/mm/yyyy, ccy, interp, extrap, daycount, compounding,
       maturities..., yields...]
    """

    def __init__(self, *, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def standard_maturities_days() -> List[int]:
        return [1, 7, 14, 30, 60, 91, 182, 365, 730, 1095, 1825, 3652, 7305, 10957]

    def _term_structure(self, maturities_days: np.ndarray, level: float, slope: float, curvature: float) -> np.ndarray:
        t = maturities_days / 365.0
        return level + slope * (1.0 - np.exp(-t)) + curvature * (t / (1.0 + t))

    def generate_row(
        self,
        spec: DummyCurveSpec,
        obs_date: date,
        maturities_days: Optional[List[int]] = None,
        *,
        noise_bps: float = 3.0,
        inverted_prob: float = 0.1,
    ) -> List[object]:
        mats = np.array(maturities_days or self.standard_maturities_days(), dtype=float)

        # Random but realistic-ish shape
        base_level = float(self.rng.uniform(0.005, 0.05))   # 0.5% .. 5%
        slope = float(self.rng.uniform(0.0, 0.04))
        curvature = float(self.rng.uniform(-0.01, 0.02))

        if self.rng.random() < inverted_prob:
            slope = -abs(slope)

        y = self._term_structure(mats, base_level, slope, curvature)

        # add small noise in bps
        y += (noise_bps * 1e-4) * self.rng.standard_normal(size=y.shape)

        row = [
            "Yield",
            spec.curve_id,
            obs_date.strftime("%d/%m/%Y"),
            spec.currency,
            spec.interpolation,
            spec.extrapolation,
            spec.day_count,
            spec.compounding,
        ]
        row.extend([int(x) for x in mats.tolist()])
        row.extend([float(x) for x in y.tolist()])
        return row

    def generate_dataset(
        self,
        specs: List[DummyCurveSpec],
        start_date: date,
        n_days: int,
        *,
        step_days: int = 1,
    ) -> pd.DataFrame:
        rows = []
        for spec in specs:
            for k in range(0, n_days, step_days):
                rows.append(self.generate_row(spec, start_date + timedelta(days=k)))
        return pd.DataFrame(rows)

    def to_csv(self, df: pd.DataFrame, path: str) -> None:
        # No header: matches your current sample style
        df.to_csv(path, index=False, header=False)
