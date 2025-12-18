#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
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd
import numpy as np

from ..objects.yield_curve import YieldCurve, YieldCurveMeta


META_COLS_LONG = [
    "curve_type", "curve_id", "observation_date", "currency_id",
    "interp_type", "extrap_type", "day_count_conv", "compounding_freq",
]


@dataclass(frozen=True)
class ParsedCurveRow:
    meta: YieldCurveMeta
    maturity_days: np.ndarray
    zero_rates: np.ndarray


class YieldCurveParser:
    """
    Parser for yield curves that supports:
      - Long format CSV (columns: curve_type, curve_id, ..., maturity_days, yield, ...)
      - Wide format CSV (8 metadata fields then maturities then yields)
        including the '...'-glued token patterns such as '18...309'
    """

    # Detect tokens like "18...309" or "182...0.01" etc.
    _ELLIPSIS_GLUE = re.compile(r"^(\d+)\.\.\.(.+)$")

    def parse(self, path: str) -> List[ParsedCurveRow]:
        """
        Auto-detect long vs wide and return a list of ParsedCurveRow
        (one per curve snapshot row).
        """
        # Try reading as a "normal" CSV with header
        try:
            df = pd.read_csv(path)
            # Long format if it contains maturity_days and yield columns
            if "maturity_days" in df.columns and ("yield" in df.columns or "zero_rate" in df.columns):
                return self._parse_long_df(df)
            # If it looks like your wide file got mis-read into a single row of headers, fallback
        except Exception:
            pass

        # Wide format: read raw lines
        return self._parse_wide_lines(path)

    # ---------- LONG FORMAT ----------
    def _parse_long_df(self, df: pd.DataFrame) -> List[ParsedCurveRow]:
        ycol = "yield" if "yield" in df.columns else "zero_rate"

        required = set(META_COLS_LONG + ["maturity_days", ycol])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Long-format CSV missing columns: {missing}")

        out: List[ParsedCurveRow] = []
        for (curve_id, obs_date), g in df.groupby(["curve_id", "observation_date"]):
            meta_row = g.iloc[0]
            meta = YieldCurveMeta(
                curve_type=str(meta_row["curve_type"]),
                curve_id=str(meta_row["curve_id"]),
                observation_date=str(meta_row["observation_date"]),
                currency_id=str(meta_row["currency_id"]),
                interp_type=str(meta_row["interp_type"]),
                extrap_type=str(meta_row["extrap_type"]),
                day_count_conv=str(meta_row["day_count_conv"]),
                compounding_freq=str(meta_row["compounding_freq"]),
            )
            mats = g["maturity_days"].astype(float).to_numpy()
            y = g[ycol].astype(float).to_numpy()
            out.append(ParsedCurveRow(meta=meta, maturity_days=mats, zero_rates=y))
        return out

    # ---------- WIDE FORMAT ----------
    def _parse_wide_lines(self, path: str) -> List[ParsedCurveRow]:
        rows: List[ParsedCurveRow] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(self._parse_one_wide_line(line))
        return rows

    def _parse_one_wide_line(self, line: str) -> ParsedCurveRow:
        # Split by commas (wide files are comma-separated)
        parts = [p.strip() for p in line.split(",")]

        # Remove empties at the end
        while parts and parts[-1] == "":
            parts.pop()

        if len(parts) < 10:
            raise ValueError(f"Wide row too short: {line[:120]}...")

        # Metadata (first 8)
        meta = YieldCurveMeta(
            curve_type=parts[0],
            curve_id=parts[1],
            observation_date=parts[2],
            currency_id=parts[3],
            interp_type=parts[4],
            extrap_type=parts[5],
            day_count_conv=parts[6],
            compounding_freq=parts[7],
        )

        dyn = parts[8:]

        # Fix ellipsis glued tokens like "18...309" -> ["18", "309"]
        dyn_fixed: List[str] = []
        for tok in dyn:
            if tok == "...":
                continue
            m = self._ELLIPSIS_GLUE.match(tok)
            if m:
                dyn_fixed.append(m.group(1))
                rest = m.group(2)
                if rest and rest != "...":
                    dyn_fixed.append(rest)
            else:
                dyn_fixed.append(tok)

        # Now dyn_fixed starts with maturity days (ints) until first non-int token
        maturities: List[int] = []
        i = 0
        while i < len(dyn_fixed) and dyn_fixed[i].isdigit():
            maturities.append(int(dyn_fixed[i]))
            i += 1

        if not maturities:
            raise ValueError(f"Could not parse maturity pillars in wide row: {line[:120]}...")

        # The remaining tokens are yields (floats). Collect as many as maturities.
        yields: List[float] = []
        for tok in dyn_fixed[i:]:
            if tok in ("", "..."):
                continue
            try:
                yields.append(float(tok))
            except ValueError:
                continue
            if len(yields) == len(maturities):
                break

        if len(yields) != len(maturities):
            raise ValueError(
                f"Yield count mismatch for {meta.curve_id} {meta.observation_date}: "
                f"{len(maturities)} maturities vs {len(yields)} yields"
            )

        return ParsedCurveRow(
            meta=meta,
            maturity_days=np.array(maturities, dtype=float),
            zero_rates=np.array(yields, dtype=float),
        )

    # ---------- CONVERSIONS ----------
    @staticmethod
    def to_long_df(rows: List[ParsedCurveRow]) -> pd.DataFrame:
        data = []
        for r in rows:
            mats = r.maturity_days
            y = r.zero_rates
            for m, yy in zip(mats, y):
                data.append({
                    "curve_type": r.meta.curve_type,
                    "curve_id": r.meta.curve_id,
                    "observation_date": r.meta.observation_date,
                    "currency_id": r.meta.currency_id,
                    "interp_type": r.meta.interp_type,
                    "extrap_type": r.meta.extrap_type,
                    "day_count_conv": r.meta.day_count_conv,
                    "compounding_freq": r.meta.compounding_freq,
                    "maturity_days": float(m),
                    "yield": float(yy),
                    "maturity_years": float(m) / 365.0,
                    "yield_bps": float(yy) * 10000.0,
                })
        return pd.DataFrame(data)

    @staticmethod
    def to_wide_df(rows: List[ParsedCurveRow]) -> pd.DataFrame:
        """
        Output wide format: one row per curve snapshot.
        Uses a clean convention: maturity pillars then yields.
        """
        out_rows = []
        for r in rows:
            base = [
                r.meta.curve_type, r.meta.curve_id, r.meta.observation_date, r.meta.currency_id,
                r.meta.interp_type, r.meta.extrap_type, r.meta.day_count_conv, r.meta.compounding_freq,
            ]
            mats = [int(x) for x in r.maturity_days.tolist()]
            y = [float(x) for x in r.zero_rates.tolist()]
            out_rows.append(base + mats + y)
        return pd.DataFrame(out_rows)

    @staticmethod
    def to_objects(rows: List[ParsedCurveRow]) -> List[YieldCurve]:
        return [YieldCurve(r.meta, r.maturity_days, r.zero_rates) for r in rows]
