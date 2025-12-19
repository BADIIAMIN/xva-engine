from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IRScenarioCube:
    """
    Container for simulated interest-rate scenarios.

    rates[p, t, k] = simulated continuous zero rate at simulation time t
                     for remaining maturity corresponding to pillar k.
    """
    rates: np.ndarray               # (P, T, K)
    time_grid_years: np.ndarray     # (T,)
    pillars_days: np.ndarray        # (K,)
    curve_id: str = "IR_BASE"
