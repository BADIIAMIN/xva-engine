from typing import Any, Dict
from .base import MarketDataSource
from ..environment import MarketDataEnvironment


class DummySource(MarketDataSource):
    """
    Dummy market data source for testing and demos.

    The snapshot returns a simple environment with:
      - flat risk-free rate
      - equity spot and volatility

    Parameters
    ----------
    params : dict
        Arbitrary dictionary controlling the dummy data, e.g.:

        {
            "spot": 100.0,
            "rate": 0.02,
            "sigma": 0.20
        }
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def get_snapshot(self, as_of: str) -> MarketDataEnvironment:
        data = {
            "fx_spot:EQ.TEST": self.params.get("spot", 100.0),
            "curve:RISK_FREE:TEST": self.params.get("rate", 0.02),
            "vol:EQ.TEST": self.params.get("sigma", 0.20),
        }
        return MarketDataEnvironment(data)

    def get_time_series(self, identifier: str, start: str, end: str) -> Any:
        # For now, no historical data.
        return None
