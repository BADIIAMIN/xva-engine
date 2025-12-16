from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from ..core.time_grid import TimeGrid
from ..market_data.environment import MarketDataEnvironment


class RiskFactorModel(ABC):
    """
    Abstract base class for all risk factor models.

    Each concrete subclass implements the dynamics of a specific
    risk factor under a chosen measure (risk–neutral or historical).

    Parameters
    ----------
    name : str
        Identifier for the model / risk factor (e.g. ``"EQ.SPX"``).
    params : dict
        Model parameters (spot, drift, volatility, mean reversion, etc.).

    Notes
    -----
    The contract of this class is intentionally minimal: calibration
    and path generation. Discretisation details (Euler, Milstein, ...)
    can either be implemented inside the model or delegated to the
    simulation driver.
    """

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def calibrate(self, env: MarketDataEnvironment, calibrator: "Calibrator") -> None:
        """
        Calibrate model parameters to market or historical data.

        Parameters
        ----------
        env : MarketDataEnvironment
            Market data snapshot and/or time series.
        calibrator : Calibrator
            Calibration strategy (historical vs implied).
        """
        raise NotImplementedError

    @abstractmethod
    def simulate_paths(
        self,
        time_grid: TimeGrid,
        n_scenarios: int,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Simulate model paths on a given time grid.

        Parameters
        ----------
        time_grid : TimeGrid
            Time grid for simulation.
        n_scenarios : int
            Number of Monte Carlo scenarios.
        rng : numpy.random.Generator
            Random number generator.
        **kwargs
            Optional implementation-specific arguments.

        Returns
        -------
        numpy.ndarray
            Simulated paths as array of shape ``(n_scenarios, n_times)`` or
            ``(n_scenarios, n_times, dim)``.
        """
        raise NotImplementedError
