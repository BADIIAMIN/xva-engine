from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from ..objects.credit_curve import CreditCurveMeta, CreditSpreadCurve


@dataclass(frozen=True)
class ParsedCreditCurveRow:
    meta: CreditCurveMeta
    maturity_months: np.ndarray
    par_spreads: np.ndarray


class CreditSpreadParser:
    """
    Parser for credit spread curves.

    Supports:
      - Wide format: one row per curve snapshot:
          [meta(10 cols), maturities..., spreads...]
        where maturities appear to be in MONTHS in your dataset (6,12,24,...)
      - Long format (optional future): one row per pillar

    Fixed meta columns (observed in your files):
      0 curve_type
      1 curve_id
      2 observation_date
      3 currency_id
      4 interp_type
      5 extrap_type
      6 payment_freq
      7 day_count
      8 basis
      9 recovery
    """

    FIXED_COLS = 10

    def parse(self, path: str) -> List[ParsedCreditCurveRow]:
        """
        Auto-detect: if it contains pillar columns, parse wide;
        if it already looks long, you can extend later.

        For now, both of your inputs are wide.
        """
        rows: List[ParsedCreditCurveRow] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(self._parse_one_wide_line(line))
        return rows

    def _parse_one_wide_line(self, line: str) -> ParsedCreditCurveRow:
        parts = [p.strip() for p in line.split(",")]

        # trim trailing empties
        while parts and parts[-1] == "":
            parts.pop()

        if len(parts) < self.FIXED_COLS + 4:
            raise ValueError(f"Row too short to be a credit curve: {line[:120]}...")

        meta = CreditCurveMeta(
            curve_type=parts[0],
            curve_id=parts[1],
            observation_date=parts[2],
            currency_id=parts[3],
            interp_type=parts[4],
            extrap_type=parts[5],
            payment_freq=parts[6],
            day_count=parts[7],
            basis=float(parts[8]),
            recovery=float(parts[9]),
        )

        dyn = parts[self.FIXED_COLS :]

        # Keep only tokens that can become floats (skip "..." just in case)
        clean: List[str] = []
        for t in dyn:
            if t == "" or t == "...":
                continue
            clean.append(t)

        # Convert to floats where possible
        nums: List[float] = []
        for t in clean:
            try:
                nums.append(float(t))
            except ValueError:
                # skip non-numeric stray tokens
                continue

        if len(nums) < 4:
            raise ValueError(f"Not enough numeric dynamic columns in row: {meta.curve_id} {meta.observation_date}")

        # Split dynamic numeric block into maturities + spreads:
        # This works because your files are [maturities..., spreads...]
        if len(nums) % 2 != 0:
            raise ValueError(
                f"Dynamic numeric block length not even for {meta.curve_id} {meta.observation_date}: {len(nums)}"
            )

        n = len(nums) // 2
        maturities = np.array(nums[:n], dtype=float)
        spreads = np.array(nums[n:], dtype=float)

        # Basic sanity: maturities should be increasing-ish
        if maturities.size >= 2 and not np.all(np.diff(np.sort(maturities)) >= 0):
            maturities = np.sort(maturities)

        if maturities.size != spreads.size:
            raise ValueError("Maturity/spread size mismatch after split.")

        return ParsedCreditCurveRow(meta=meta, maturity_months=maturities, par_spreads=spreads)

    # ---------- conversions ----------
    @staticmethod
    def to_long_df(rows: List[ParsedCreditCurveRow]) -> pd.DataFrame:
        data = []
        for r in rows:
            for m, s in zip(r.maturity_months, r.par_spreads):
                data.append({
                    "curve_type": r.meta.curve_type,
                    "curve_id": r.meta.curve_id,
                    "observation_date": r.meta.observation_date,
                    "currency_id": r.meta.currency_id,
                    "interp_type": r.meta.interp_type,
                    "extrap_type": r.meta.extrap_type,
                    "payment_freq": r.meta.payment_freq,
                    "day_count": r.meta.day_count,
                    "basis": r.meta.basis,
                    "recovery": r.meta.recovery,
                    "maturity_months": float(m),
                    "maturity_years": float(m) / 12.0,
                    "par_spread": float(s),
                    "par_spread_bps": float(s) * 10000.0,
                })
        return pd.DataFrame(data)

    @staticmethod
    def to_wide_df(rows: List[ParsedCreditCurveRow]) -> pd.DataFrame:
        out_rows = []
        for r in rows:
            base = [
                r.meta.curve_type, r.meta.curve_id, r.meta.observation_date, r.meta.currency_id,
                r.meta.interp_type, r.meta.extrap_type, r.meta.payment_freq, r.meta.day_count,
                float(r.meta.basis), float(r.meta.recovery),
            ]
            mats = [float(x) for x in r.maturity_months.tolist()]
            spr = [float(x) for x in r.par_spreads.tolist()]
            out_rows.append(base + mats + spr)
        return pd.DataFrame(out_rows)

    @staticmethod
    def to_objects(rows: List[ParsedCreditCurveRow]) -> List[CreditSpreadCurve]:
        return [CreditSpreadCurve(r.meta, r.maturity_months, r.par_spreads) for r in rows]
