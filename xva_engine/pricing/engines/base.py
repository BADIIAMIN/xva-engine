from abc import ABC, abstractmethod
import numpy as np
from ...instruments.base import Instrument
from ..context import PricingContext
from ...core.cube import RiskFactorCube


class PricingEngine(ABC):
    """Base pricing engine interface."""

    @abstractmethod
    def price(self, inst: Instrument, ctx: PricingContext) -> float:
        """Single-valuation pricing at t0."""
        raise NotImplementedError

    def price_paths(
        self,
        inst: Instrument,
        cube: RiskFactorCube,
        ctx: PricingContext
    ) -> np.ndarray:
        """
        Optional: price along paths.
        Default: raise unless overridden.
        """
        raise NotImplementedError
