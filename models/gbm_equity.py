import numpy as np
from typing import Any
from .base import RiskFactorModel
from ..core.time_grid import TimeGrid
from ..market_data.environment import MarketDataEnvironment


class GBMEquityModel(RiskFactorModel):
    r"""
    Geometric Brownian Motion (GBM) equity price model.

    Dynamics (under chosen measure)::

        dS_t = \mu S_t dt + \sigma S_t dW_t

    with Euler discretisation on the log-price.

    Parameters
    ----------
    name : str
        Risk factor identifier, e.g. ``"EQ.SPX"``.
    params : dict
        Must contain at least:
        ``"spot"`` : float
            Initial spot price.
        ``"mu"`` : float
            Drift (under P or Q).
        ``"sigma"`` : float
            Volatility.
    """

    def calibrate(self, env: MarketDataEnvironment, calibrator: "Calibrator") -> None:
        """
        Delegate parameter estimation to the provided calibrator.

        For the demo slice this method is a no-op; the parameters are
        assumed to be set directly in ``params``.
        """
        # In a full implementation, call calibrator.calibrate(self, env)
        return

    def simulate_paths(
        self,
        time_grid: TimeGrid,
        n_scenarios: int,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Simulate GBM equity paths using an Euler scheme on log S.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_scenarios, n_times)``.
        """
        times = time_grid.as_array()
        n_times = times.shape[0]

        spot = float(self.params["spot"])
        mu = float(self.params["mu"])
        sigma = float(self.params["sigma"])

        paths = np.zeros((n_scenarios, n_times), dtype=float)
        paths[:, 0] = spot

        for i in range(1, n_times):
            dt = times[i] - times[i - 1]
            # Normal increments
            z = rng.standard_normal(size=n_scenarios)
            # Log-Euler step: S_{t+dt} = S_t * exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) z)
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i - 1] * np.exp(drift + diffusion)

        return paths
