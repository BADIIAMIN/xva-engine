import numpy as np
from typing import Optional
from .base import PricingEngine
from ...instruments.base import Instrument
from ...instruments.vanilla import EuropeanOption
from ..context import PricingContext
from ...core.cube import RiskFactorCube


class PathwiseMCEngine(PricingEngine):
    """
    Simple pathwise Monte Carlo engine for European options.

    Notes
    -----
    This engine assumes that the `RiskFactorCube` contains the underlying
    price as one of the factors, and that the time grid includes the
    option maturity (or very close to it). Discounting is performed using
    a flat risk-free rate from the `PricingContext`.
    """

    def __init__(
        self,
        underlying_factor_name: str,
        risk_free_curve_key: str,
        maturity_tolerance: float = 1e-6,
    ):
        self.underlying_factor_name = underlying_factor_name
        self.risk_free_curve_key = risk_free_curve_key
        self.maturity_tolerance = maturity_tolerance

    def _find_factor_index(self, cube: RiskFactorCube) -> int:
        try:
            return cube.factors.index(self.underlying_factor_name)
        except ValueError as exc:
            raise KeyError(f"Factor {self.underlying_factor_name} not in cube.factors") from exc

    def price(self, inst: Instrument, ctx: PricingContext) -> float:
        """
        Single-valuation pricing is not implemented here; use `price_paths`.
        """
        raise NotImplementedError("Use price_paths() with a RiskFactorCube for MC valuation.")

    def price_paths(
        self,
        inst: Instrument,
        cube: RiskFactorCube,
        ctx: PricingContext,
    ) -> np.ndarray:
        """
        Price a EuropeanOption along all scenarios and times in the cube.

        Parameters
        ----------
        inst : Instrument
            Must be a `EuropeanOption`.
        cube : RiskFactorCube
            Simulated underlying paths.
        ctx : PricingContext
            Provides risk-free rate via market environment.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_scenarios, n_times)`` with discounted
            option value at each time. For this demo, the value is
            non-zero only at or after maturity.
        """
        if not isinstance(inst, EuropeanOption):
            raise TypeError("PathwiseMCEngine currently supports only EuropeanOption.")

        factor_idx = self._find_factor_index(cube)
        underlying_paths = cube.data[:, :, factor_idx]  # (n_scenarios, n_times)

        times = cube.time_grid.as_array()
        n_scenarios, n_times = underlying_paths.shape

        # Locate maturity index
        maturity = inst.maturity
        maturity_idx: Optional[int] = None
        for i, t in enumerate(times):
            if abs(t - maturity) <= self.maturity_tolerance:
                maturity_idx = i
                break
        if maturity_idx is None:
            raise ValueError(
                f"Maturity {maturity} not found in time grid (tolerance={self.maturity_tolerance})."
            )

        # Get flat risk-free rate
        r = ctx.market_env.get_curve(self.risk_free_curve_key)

        # Payoff at maturity
        s_T = underlying_paths[:, maturity_idx]
        if inst.option_type.lower() == "call":
            payoff = np.maximum(s_T - inst.strike, 0.0)
        else:
            payoff = np.maximum(inst.strike - s_T, 0.0)

        # Discount factor from 0 to maturity
        df = np.exp(-r * maturity)

        # Discounted payoff
        discounted_payoff = df * payoff  # shape: (n_scenarios,)

        # For simplicity: set value constant = discounted payoff after maturity,
        # and zero before maturity.
        values = np.zeros((n_scenarios, n_times))
        for i, t in enumerate(times):
            if t >= maturity - self.maturity_tolerance:
                values[:, i] = discounted_payoff

        return values
