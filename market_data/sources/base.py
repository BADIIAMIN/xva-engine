from abc import ABC, abstractmethod
from typing import Any
from ..environment import MarketDataEnvironment


class MarketDataSource(ABC):
    """Abstract interface for all market data sources."""

    @abstractmethod
    def get_snapshot(self, as_of: str) -> MarketDataEnvironment:
        """Return a market data snapshot at a given date."""
        raise NotImplementedError

    @abstractmethod
    def get_time_series(self, identifier: str, start: str, end: str) -> Any:
        """Return historical time series for an identifier."""
        raise NotImplementedError
