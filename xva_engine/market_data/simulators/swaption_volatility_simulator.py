from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import date, timedelta


@dataclass(frozen=True)
class DummySwaptionVolSpec:
    cube_id: str
    currency_id: str
    expiry_interp: str = "lin"
    expiry_extrap: str = "near"
    tenor_interp: str = "cubic"
    tenor_extrap: str = "near"
    strike_interp: str = "lin"
    strike_extrap: str = "near"
    payment_freq: str = "QUARTERLY"
    day_count: str = "ACT360"
    basis: str = "365"
    calendar: str = "NONE"
    unit: str = "DAYS"
    roll_rule: str = "AFTER"
    stub_rule: str = "PREV"
    is_strike_smile: str = "TRUE"


class SwaptionVolDummySimulator:
    """
    Generates dummy swaption volatility cube rows:
      [meta(18), expiry_months, strike, tenor_1, vol_1, tenor_2, vol_2, ...]
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def default_expiry_months() -> List[int]:
        return [1, 3, 6, 12, 24, 60]

    @staticmethod
    def default_strikes() -> List[float]:
        # “smile” grid (similar to your negative strike shifts)
        return [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01]

    @staticmethod
    def default_tenor_days() -> List[int]:
        return [30, 90, 365, 730, 1825, 3650, 5475, 7300]

    def _atm_term(self, tenor_days: np.ndarray, level: float, slope: float) -> np.ndarray:
        t = tenor_days / 365.0
        return np.maximum(level + slope * (1.0 - np.exp(-t)), 1e-6)

    def _smile(self, strike: float, atm: np.ndarray, smile: float) -> np.ndarray:
        # simple symmetric-ish smile around 0
        return np.maximum(atm + smile * (strike ** 2), 1e-6)

    def simulate_rows_for_date(
        self,
        spec: DummySwaptionVolSpec,
        obs: date,
        expiry_months: Optional[List[int]] = None,
        strikes: Optional[List[float]] = None,
        tenors: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        exp = expiry_months or self.default_expiry_months()
        kgrid = strikes or self.default_strikes()
        tnr = np.asarray(tenors or self.default_tenor_days(), dtype=float)

        rows = []
        for e in exp:
            # base level depends on expiry
            level = float(self.rng.uniform(0.08, 0.35)) * (1.0 / np.sqrt(max(e, 1)))
            slope = float(self.rng.uniform(-0.03, 0.05))
            atm = self._atm_term(tnr, level, slope)

            smile = float(self.rng.uniform(2.0, 8.0))  # curvature strength

            for k in kgrid:
                vols = self._smile(k, atm, smile)
                vols += 0.002 * self.rng.standard_normal(size=vols.shape)  # noise
                vols = np.maximum(vols, 1e-6)

                meta = [
                    "SwaptionVolCube",
                    spec.cube_id,
                    obs.strftime("%d/%m/%Y"),
                    spec.currency_id,
                    spec.expiry_interp,
                    spec.expiry_extrap,
                    spec.tenor_interp,
                    spec.tenor_extrap,
                    spec.strike_interp,
                    spec.strike_extrap,
                    spec.payment_freq,
                    spec.day_count,
                    spec.basis,
                    spec.calendar,
                    spec.unit,
                    spec.roll_rule,
                    spec.stub_rule,
                    spec.is_strike_smile,
                ]

                row = meta + [float(e), float(k)]
                for tt, vv in zip(tnr.tolist(), vols.tolist()):
                    row += [float(tt), float(vv)]
                rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def save_wide_csv(df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False, header=False)
