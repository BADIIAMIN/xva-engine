from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ..objects.swaption_vol_cube import SwaptionVolMeta, SwaptionVolCube


@dataclass(frozen=True)
class ParsedSwaptionVolRow:
    meta: SwaptionVolMeta
    expiry_months: float
    strike: float
    tenor_days: np.ndarray
    vols: np.ndarray


class SwaptionVolatilityParser:
    """
    Parser for swaption volatility cube CSVs.

    Observed input format (wide):
      [meta (18 fields), expiry_months, strike, tenor_1, vol_1, tenor_2, vol_2, ...]

    From your files:
      - expiry_months is often 6, 12, ...
      - strike is a grid (e.g. -0.0175, -0.0125, ...)
      - tenor_days are (30, 90, 365, 730, 1825, 3650, ...)
      - vols are decimals (e.g. 0.3176)
    """

    FIXED_COLS = 18  # based on your CSV inspection

    def parse(self, path: str) -> List[ParsedSwaptionVolRow]:
        out: List[ParsedSwaptionVolRow] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(self._parse_one_line(line))
        return out

    def _parse_one_line(self, line: str) -> ParsedSwaptionVolRow:
        parts = [p.strip() for p in line.split(",")]
        while parts and parts[-1] == "":
            parts.pop()

        if len(parts) < self.FIXED_COLS + 6:
            raise ValueError(f"Row too short to be a swaption vol row: {line[:120]}...")

        meta = SwaptionVolMeta(
            cube_type=parts[0],
            cube_id=parts[1],
            observation_date=parts[2],
            currency_id=parts[3],
            expiry_interp=parts[4],
            expiry_extrap=parts[5],
            tenor_interp=parts[6],
            tenor_extrap=parts[7],
            strike_interp=parts[8],
            strike_extrap=parts[9],
            payment_freq=parts[10],
            day_count=parts[11],
            basis=parts[12],
            calendar=parts[13],
            unit=parts[14],
            roll_rule=parts[15],
            stub_rule=parts[16],
            is_strike_smile=parts[17],
        )

        dyn = parts[self.FIXED_COLS :]

        # Convert as many as possible to floats (skip "..." if any)
        nums: List[float] = []
        for t in dyn:
            if t in ("", "..."):
                continue
            try:
                nums.append(float(t))
            except ValueError:
                continue

        # Expected: expiry, strike, then tenor/vol pairs
        if len(nums) < 4:
            raise ValueError(f"Not enough numeric fields after meta for {meta.cube_id} {meta.observation_date}")

        expiry_months = float(nums[0])
        strike = float(nums[1])
        pairs = nums[2:]

        if len(pairs) % 2 != 0:
            raise ValueError(
                f"Tenor/vol pairs length not even for expiry={expiry_months} strike={strike} "
                f"({meta.cube_id} {meta.observation_date}): {len(pairs)}"
            )

        tenor = np.array(pairs[0::2], dtype=float)
        vols = np.array(pairs[1::2], dtype=float)

        if tenor.size < 2:
            raise ValueError("Need at least 2 tenor pillars.")

        return ParsedSwaptionVolRow(meta, expiry_months, strike, tenor, vols)

    # ---------- outputs (like yield curves) ----------
    @staticmethod
    def to_long_df(rows: List[ParsedSwaptionVolRow]) -> pd.DataFrame:
        data = []
        for r in rows:
            for t, v in zip(r.tenor_days, r.vols):
                data.append({
                    "cube_type": r.meta.cube_type,
                    "cube_id": r.meta.cube_id,
                    "observation_date": r.meta.observation_date,
                    "currency_id": r.meta.currency_id,
                    "expiry_interp": r.meta.expiry_interp,
                    "expiry_extrap": r.meta.expiry_extrap,
                    "tenor_interp": r.meta.tenor_interp,
                    "tenor_extrap": r.meta.tenor_extrap,
                    "strike_interp": r.meta.strike_interp,
                    "strike_extrap": r.meta.strike_extrap,
                    "payment_freq": r.meta.payment_freq,
                    "day_count": r.meta.day_count,
                    "basis": r.meta.basis,
                    "calendar": r.meta.calendar,
                    "unit": r.meta.unit,
                    "roll_rule": r.meta.roll_rule,
                    "stub_rule": r.meta.stub_rule,
                    "is_strike_smile": r.meta.is_strike_smile,
                    "expiry_months": float(r.expiry_months),
                    "expiry_years": float(r.expiry_months) / 12.0,
                    "strike": float(r.strike),
                    "tenor_days": float(t),
                    "tenor_years": float(t) / 365.0,
                    "vol": float(v),
                    "vol_bps": float(v) * 10000.0,
                })
        return pd.DataFrame(data)

    @staticmethod
    def to_wide_df(rows: List[ParsedSwaptionVolRow]) -> pd.DataFrame:
        out_rows = []
        for r in rows:
            base = [
                r.meta.cube_type,
                r.meta.cube_id,
                r.meta.observation_date,
                r.meta.currency_id,
                r.meta.expiry_interp,
                r.meta.expiry_extrap,
                r.meta.tenor_interp,
                r.meta.tenor_extrap,
                r.meta.strike_interp,
                r.meta.strike_extrap,
                r.meta.payment_freq,
                r.meta.day_count,
                r.meta.basis,
                r.meta.calendar,
                r.meta.unit,
                r.meta.roll_rule,
                r.meta.stub_rule,
                r.meta.is_strike_smile,
                float(r.expiry_months),
                float(r.strike),
            ]
            pairs = []
            for t, v in zip(r.tenor_days.tolist(), r.vols.tolist()):
                pairs.extend([float(t), float(v)])
            out_rows.append(base + pairs)
        return pd.DataFrame(out_rows)

    @staticmethod
    def to_cube(rows: List[ParsedSwaptionVolRow]) -> SwaptionVolCube:
        if not rows:
            raise ValueError("No rows to build a SwaptionVolCube.")

        # All rows share same meta for a given (cube_id, obs_date) in your dataset.
        meta = rows[0].meta
        cube = SwaptionVolCube(meta)
        for r in rows:
            cube.add_slice(r.expiry_months, r.strike, r.tenor_days, r.vols)
        return cube
