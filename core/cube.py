from dataclasses import dataclass
from typing import List, Any
import numpy as np
from .time_grid import TimeGrid


@dataclass
class RiskFactorCube:
    """
    Scenario cube for risk factor paths.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_scenarios, n_times, n_factors)`` containing
        the simulated risk factor values.
    scenarios : list
        Scenario identifiers. Typically integers ``0 .. n_scenarios-1``.
    time_grid : TimeGrid
        Time grid corresponding to the second dimension of ``data``.
    factors : list of str
        Names or identifiers of each risk factor (e.g. ``"EQ.SPX"``,
        ``"IR.EUR.3M"``).

    Notes
    -----
    The `RiskFactorCube` is the main output of the simulation layer
    and the main input to the pricing layer.
    """
    data: np.ndarray
    scenarios: List[Any]
    time_grid: TimeGrid
    factors: List[str]


@dataclass
class ExposureCube:
    """
    Scenario cube for trade or portfolio values.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_scenarios, n_times, n_trades)`` containing
        the discounted value of each trade along each scenario and time.
    scenarios : list
        Scenario identifiers.
    time_grid : TimeGrid
        Time grid used for the valuation.
    trades : list of str
        Trade identifiers, aligned with the last dimension of ``data``.

    Notes
    -----
    The `ExposureCube` is produced by the pricing layer and consumed
    by collateral, aggregation and XVA modules.
    """
    data: np.ndarray
    scenarios: List[Any]
    time_grid: TimeGrid
    trades: List[str]
