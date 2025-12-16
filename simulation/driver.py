from typing import List
import numpy as np
from ..core.time_grid import TimeGrid
from ..core.cube import RiskFactorCube
from ..models.base import RiskFactorModel
from ..models.correlation import CorrelationModel
from ..config.schema import SimulationConfig


class SimulationDriver:
    """
    Orchestrates simulation of all risk factor models with correlation
    on a given time grid.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(
        self,
        models: List[RiskFactorModel],
        corr_model: CorrelationModel,
        time_grid: TimeGrid,
        seed: int = 42,
    ) -> RiskFactorCube:
        """Generate a RiskFactorCube according to the config and models."""
        rng = np.random.default_rng(seed)
        n_scenarios = self.config.n_scenarios
        n_times = len(time_grid.times)
        n_factors = len(models)

        # Placeholder structure
        data = np.zeros((n_scenarios, n_times, n_factors))

        # TODO: implement proper correlated stepping and SDE schemes
        # For now, just call each model independently
        for j, model in enumerate(models):
            paths = model.simulate_paths(time_grid, n_scenarios, rng)
            data[:, :, j] = paths

        factors = [m.name for m in models]
        scenarios = list(range(n_scenarios))
        return RiskFactorCube(data=data, scenarios=scenarios, time_grid=time_grid, factors=factors)
