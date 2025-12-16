from typing import List
from ...pricing.engines.base import PricingEngine
from ...instruments.portfolio import Portfolio
from ...pricing.context import PricingContext


class PricerBenchmark:
    """Compare two pricing engines on a given portfolio."""

    def compare(
        self,
        engine1: PricingEngine,
        engine2: PricingEngine,
        portfolio: Portfolio,
        ctx: PricingContext,
    ):
        # TODO: implement benchmark logic (differences, stats, sanity checks)
        pass
