from abc import ABC, abstractmethod
from ..market_data.environment import MarketDataEnvironment
from ..base import RiskFactorModel


class Calibrator(ABC):
    """Base class for all calibration strategies."""

    @abstractmethod
    def calibrate(self, model: RiskFactorModel, env: MarketDataEnvironment) -> None:
        """Update model.params in place to fit market/historical data."""
        raise NotImplementedError
