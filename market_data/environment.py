from typing import Any, Dict


class MarketDataEnvironment:
    """
    Container for all market data needed by models and pricing:
    curves, vol surfaces, FX spots, spreads, etc.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get_curve(self, key: str) -> Any:
        """Return a yield curve / discount curve identified by key."""
        return self._data.get(f"curve:{key}")

    def get_vol_surface(self, key: str) -> Any:
        """Return a volatility surface identified by key."""
        return self._data.get(f"vol:{key}")

    def get_fx_spot(self, pair: str) -> float:
        """Return FX spot for a currency pair."""
        return self._data.get(f"fx_spot:{pair}")

    def get_time_series(self, key: str):
        """Return historical time series for a given identifier."""
        return self._data.get(f"ts:{key}")
