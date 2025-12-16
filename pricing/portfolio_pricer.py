from typing import List
import numpy as np
from .context import PricingContext
from .engines.base import PricingEngine
from ..instruments.portfolio import Portfolio
from ..core.cube import RiskFactorCube, ExposureCube
from ..instruments.base import Instrument


class PortfolioPricer:
    """
    Prices a portfolio on a risk factor cube using a given pricing engine.
    """

    def __init__(self, engine: PricingEngine):
        self.engine = engine

    def price_on_cube(
        self,
        portfolio: Portfolio,
        cube: RiskFactorCube,
        ctx: PricingContext
    ) -> ExposureCube:
        """
        For now: naive loop over trades and delegate to engine.price_paths().
        """
        n_scenarios, n_times, _ = cube.data.shape
        n_trades = len(portfolio.trades)
        data = np.zeros((n_scenarios, n_times, n_trades))

        for k, trade in enumerate(portfolio.trades):
            # In a real implementation, you'd map cube -> scenario-specific market env.
            prices = self.engine.price_paths(trade, cube, ctx)
            data[:, :, k] = prices

        scenarios = cube.scenarios
        trades_ids = [t.id for t in portfolio.trades]
        return ExposureCube(data=data, scenarios=scenarios, time_grid=cube.time_grid, trades=trades_ids)
