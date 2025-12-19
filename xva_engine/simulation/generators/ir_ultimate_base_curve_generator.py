from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from xva_engine.simulation.risk_factors.ir.ultimate_base_curve_process import (
    UltimateBaseCurveParams,
    UltimateBaseCurveProcess,
)
from xva_engine.simulation.risk_factors.ir.mean_function import (
    MeanFunctionConfig,
    build_forward_forward_mean_function,
)
from xva_engine.simulation.risk_factors.ir.calibration_historical import (
    HistoricalCalibConfig,
    estimate_corr_and_sigma_from_history,
)


@dataclass(frozen=True)
class IrUltimateBaseCurveRunConfig:
    n_paths: int
    n_steps:int
    horizon_years: float
    seed: int = 1234
    return_driver: bool = False


class IrUltimateBaseCurveScenarioGenerator:
    """
    Generates IR zero-rate trajectories for the Ultimate Base Curve model.

    Notes on architecture:
    - This belongs under xva_engine/simulation/generators (trajectory generation).
    - It consumes market data (discount curve + history) from your market_data layer.
    - It returns a numpy cube (paths, times, pillars). You can wrap into your Cube later.
    """

    def __init__(
        self,
        pillars_days: np.ndarray,
        mean_cfg: MeanFunctionConfig = MeanFunctionConfig(),
        hist_cfg: HistoricalCalibConfig = HistoricalCalibConfig(),
    ):
        self.pillars_days = np.asarray(pillars_days, dtype=float)
        self.mean_cfg = mean_cfg
        self.hist_cfg = hist_cfg

    def calibrate_historical(
        self,
        rates_hist: np.ndarray,           # (Nobs, K) in rate units
        lam: Optional[float] = None,
        shift_bp: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (corr, sigma, lam_vec) for the process.
        """
        cfg = self.hist_cfg
        if lam is not None:
            cfg = HistoricalCalibConfig(
                lam=lam,
                shift_bp=cfg.shift_bp,
                return_horizon_days=cfg.return_horizon_days,
                day_count=cfg.day_count,
            )
        if shift_bp is not None:
            cfg = HistoricalCalibConfig(
                lam=cfg.lam,
                shift_bp=shift_bp,
                return_horizon_days=cfg.return_horizon_days,
                day_count=cfg.day_count,
            )

        corr, sigma = estimate_corr_and_sigma_from_history(rates_hist, cfg)
        K = rates_hist.shape[1]
        lam_vec = np.full(K, cfg.lam, dtype=float)
        return corr, sigma, lam_vec

    def generate(
        self,
        time_grid: np.ndarray,             # (T,) year fractions
        df0: Callable[[float], float],     # DF(0,t) callable
        corr: np.ndarray,
        sigma: np.ndarray,
        lam: np.ndarray,
        shift_bp: np.ndarray,
        run: IrUltimateBaseCurveRunConfig,
    ) -> dict[str, np.ndarray]:
        """
        Returns dict:
          - 'rates': (n_paths, T, K)
          - optionally 'driver': (n_paths, T, K)
        """
        params = UltimateBaseCurveParams(
            pillars_days=self.pillars_days,
            shift_bp=shift_bp,
            sigma=sigma,
            lam=lam,
            delta_floor=self.mean_cfg.delta_floor,
            day_count=self.mean_cfg.day_count,
        )
        process = UltimateBaseCurveProcess(params=params, corr=corr)

        g = build_forward_forward_mean_function(
            time_grid=np.asarray(time_grid, dtype=float),
            pillars_days=self.pillars_days,
            df0=df0,
            cfg=self.mean_cfg,
        )

        y, x = process.simulate(
            time_grid=time_grid,
            mean_function=g,
            n_paths=run.n_paths,
            seed=run.seed,
            return_driver=run.return_driver,
        )

        out = {"rates": y}
        if run.return_driver and x is not None:
            out["driver"] = x
        return out

    def _get_zero_rates_cube(obj) -> np.ndarray:
        if isinstance(obj, dict):
            for k in ("zero_rates", "rates", "rates_cube", "cube"):
                if k in obj and isinstance(obj[k], np.ndarray):
                    return obj[k]
        for name in ("zero_rates", "rates", "rates_cube", "cube"):
            if hasattr(obj, name):
                arr = getattr(obj, name)
                if isinstance(arr, np.ndarray):
                    return arr
        raise AttributeError("Could not find zero rates cube ...")

    def _get_pillars_years(obj) -> np.ndarray:
        if isinstance(obj, dict):
            if "pillars_years" in obj: return np.asarray(obj["pillars_years"], float)
            if "pillars_days" in obj:  return np.asarray(obj["pillars_days"], float) / 365.0
        for name in ("pillars_years", "pillars"):
            if hasattr(obj, name): return np.asarray(getattr(obj, name), float)
        if hasattr(obj, "pillars_days"):
            return np.asarray(getattr(obj, "pillars_days"), float) / 365.0
        raise AttributeError("Could not find pillars ...")


