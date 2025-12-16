from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..yield_curve import YieldCurve, YieldCurveMeta


@dataclass(frozen=True)
class ParsedYieldCurveRow:
    meta: YieldCurveMeta
    maturities_days: np.ndarray
    zero_rates: np.ndarray


class YieldCurveCsvParser:
    """
    Parse yield curve rows from CSV.

    Expected row format (no header required):
        curve_type, curve_id, observation_date, currency,
        interpolation, extrapolation, day_count, compounding,
        <maturity days...>, <yields...>

    The parser is robust to:
    - literal token "..." appearing in the row
    - trailing empty columns
    - files saved with or without header row
    """

    FIXED_COLS = 8

    def __init__(self, *, has_header: bool = False):
        self.has_header = has_header

    @staticmethod
    def _is_int_like(x: Any) -> bool:
        if x is None:
            return False
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, float) and np.isfinite(x) and abs(x - int(x)) < 1e-12:
            return True
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s == "...":
                return False
            return s.isdigit()
        return False

    @staticmethod
    def _to_int(x: Any) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            return int(round(x))
        s = str(x).strip()
        return int(s)

    @staticmethod
    def _to_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)) and np.isfinite(x):
            return float(x)
        s = str(x).strip()
        if s == "" or s == "...":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def read_rows(self, csv_path: str) -> List[ParsedYieldCurveRow]:
        """
        Read the file and return parsed yield curve rows (one per CSV row).
        """
        # Important: if has_header=False, pandas will assign default column names 0..N-1
        df = pd.read_csv(csv_path, header=0 if self.has_header else None)

        rows: List[ParsedYieldCurveRow] = []
        for _, r in df.iterrows():
            parsed = self._parse_single_row(r)
            if parsed is not None:
                rows.append(parsed)

        if not rows:
            raise ValueError(f"No valid yield curve rows parsed from {csv_path}")

        return rows

    def _parse_single_row(self, row: pd.Series) -> Optional[ParsedYieldCurveRow]:
        values = row.tolist()

        # drop trailing NaNs / empty strings
        while values and (values[-1] is None or (isinstance(values[-1], float) and np.isnan(values[-1])) or str(values[-1]).strip() == ""):
            values.pop()

        if len(values) < self.FIXED_COLS + 2:
            return None

        meta = YieldCurveMeta(
            curve_type=str(values[0]).strip(),
            curve_id=str(values[1]).strip(),
            observation_date=str(values[2]).strip(),
            currency=str(values[3]).strip(),
            interpolation=str(values[4]).strip(),
            extrapolation=str(values[5]).strip(),
            day_count=str(values[6]).strip(),
            compounding=str(values[7]).strip(),
        )

        dyn = values[self.FIXED_COLS :]

        # Remove any "..." tokens inside dyn
        dyn = [x for x in dyn if not (isinstance(x, str) and x.strip() == "...")]

        # Parse maturities as consecutive int-like tokens starting at dyn[0]
        maturities: List[int] = []
        i = 0
        while i < len(dyn) and self._is_int_like(dyn[i]):
            maturities.append(self._to_int(dyn[i]))
            i += 1

        if len(maturities) == 0:
            # no pillars
            return None

        # Remaining tokens are yields (take first len(maturities) numeric floats)
        yields_raw = dyn[i:]
        yields: List[float] = []
        for x in yields_raw:
            fx = self._to_float(x)
            if fx is None:
                continue
            yields.append(fx)
            if len(yields) == len(maturities):
                break

        # If yields shorter than maturities, pad with NaN
        if len(yields) < len(maturities):
            yields.extend([np.nan] * (len(maturities) - len(yields)))

        return ParsedYieldCurveRow(
            meta=meta,
            maturities_days=np.asarray(maturities, dtype=float),
            zero_rates=np.asarray(yields, dtype=float),
        )

    def to_objects(self, csv_path: str) -> List[YieldCurve]:
        """
        Parse all rows into YieldCurve objects.
        """
        parsed = self.read_rows(csv_path)
        curves: List[YieldCurve] = []
        for p in parsed:
            curves.append(YieldCurve(p.meta, p.maturities_days, p.zero_rates))
        return curves
