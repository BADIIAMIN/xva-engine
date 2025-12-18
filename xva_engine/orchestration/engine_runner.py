from ..config.schema import RunConfig
from ..config.loader import load_config
from ..market_data.sources.base import MarketDataSource
from ..simulation.driver import SimulationDriver
from ..pricing.portfolio_pricer import PortfolioPricer
from ..collateral.engine import CollateralEngine
from ..aggregation.exposure import ExposureMetrics
from ..aggregation.xva import XVAEngine
from ..validation.backtesting.base import Backtest
from ..reporting.exporters import ReportExporter


class EngineRunner:
    """
    High-level orchestrator for a full XVA / CCR / PFE run.
    """

    def __init__(self, market_data_source: MarketDataSource):
        self.market_data_source = market_data_source

    def run_from_file(self, config_path: str):
        cfg = load_config(config_path)
        self.run(cfg)

    def run(self, cfg: RunConfig):
        """
        Orchestrate:
          1. Market data snapshot
          2. Simulation
          3. Pricing
          4. Collateral
          5. Aggregation & XVA
          6. Backtesting
          7. Reporting
        """
        # TODO: wire all layers using cfg
        pass
