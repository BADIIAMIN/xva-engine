from __future__ import annotations

from typing import Any, Dict, List
from .base import MarketDataSource
from ..environment import MarketDataEnvironment
from ..parsers.yield_curve_parser import YieldCurveParser


class YieldCurveCsvSource(MarketDataSource):
    """
    Loads yield curves from CSV and exposes them through MarketDataEnvironment.

    Keys stored in env:
      curve:<curve_id>  -> YieldCurve object
      ts:<curve_id>     -> list[YieldCurve snapshots] (optional)
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.parser = YieldCurveParser()
        self.rows = self.parser.parse(csv_path)
        self.curves = self.parser.to_objects(self.rows)

    def get_snapshot(self, as_of: str) -> MarketDataEnvironment:
        data: Dict[str, Any] = {}
        selected = [c for c in self.curves if c.meta.observation_date == as_of]
        if not selected:
            raise ValueError(f"No curves for as_of={as_of} in {self.csv_path}")

        for c in selected:
            data[f"curve:{c.meta.curve_id}"] = c
        return MarketDataEnvironment(data)

    def get_time_series(self, identifier: str, start: str, end: str):
        # minimal: return all snapshots for curve_id (you can add proper date filtering later)
        return [c for c in self.curves if c.meta.curve_id == identifier]
