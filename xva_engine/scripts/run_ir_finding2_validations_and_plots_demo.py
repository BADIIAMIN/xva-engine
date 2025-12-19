from __future__ import annotations

import numpy as np

from xva_engine.simulation.generators.ir_ultimate_base_curve_generator import (
    IrUltimateBaseCurveScenarioGenerator,
    IrUltimateBaseCurveRunConfig
)

from xva_engine.simulation.generators.benchmarks.ir_hull_white_1f_generator import (
    HullWhite1FParams,
    simulate_hw1f_curve_paths,
)

from xva_engine.validation.ir.interpolation.test_interpolation_sensitivity import (
    run_interpolation_sensitivity,
)
from xva_engine.validation.ir.interpolation.test_forward_roughness import (
    run_forward_roughness,
)
from xva_engine.validation.ir.interpolation.test_pillar_density import (
    run_pillar_density_stress,
)
from xva_engine.validation.ir.interpolation.reporting.plots_interpolation import (
    save_finding2_plots,
)


# ------------------------------------------------------------
# Helpers: robust access to generator outputs (consistent with "rest of codes")
# ------------------------------------------------------------
def _get_zero_rates_cube(obj) -> np.ndarray:
    """
    Try common attribute names used across the project.
    Expected shape: (Npaths, Ntimes, Kpillars).
    """
    for name in ("zero_rates", "rates", "rates_cube", "cube"):
        if hasattr(obj, name):
            arr = getattr(obj, name)
            if isinstance(arr, np.ndarray):
                return arr
    raise AttributeError(
        "Could not find zero rates cube on generator output. "
        "Expected attribute: zero_rates (or rates/rates_cube/cube)."
    )


def _get_pillars_years(obj) -> np.ndarray:
    """
    Try common pillar representations.
    Returns maturity pillars in years as 1D array.
    """
    if hasattr(obj, "pillars"):
        p = getattr(obj, "pillars")
        if isinstance(p, np.ndarray):
            return p.astype(float)

    # Some outputs may store days instead
    if hasattr(obj, "pillars_days"):
        d = getattr(obj, "pillars_days")
        if isinstance(d, np.ndarray):
            return (d.astype(float) / 365.0)

    raise AttributeError(
        "Could not find pillars on generator output. "
        "Expected attribute: pillars (years) or pillars_days (days)."
    )


def _get_time_grid_years(obj, cfg: IrUltimateBaseCurveRunConfig, cube: np.ndarray) -> np.ndarray:
    """
    Prefer output time grid if present. Otherwise derive from config if it exposes
    horizon and n_steps. As a last resort, assume uniform steps of 1/12y.
    """
    for name in ("time_grid_years", "time_grid", "times_years", "t_grid_years"):
        if hasattr(obj, name):
            tg = getattr(obj, name)
            if isinstance(tg, np.ndarray) and tg.ndim == 1:
                return tg.astype(float)

    # Try infer from config
    # Common patterns: cfg.time_grid_years, cfg.n_steps, cfg.dt_years, cfg.horizon_years
    if hasattr(cfg, "time_grid_years") and isinstance(cfg.time_grid_years, np.ndarray):
        return cfg.time_grid_years.astype(float)

    n_times = cube.shape[1]

    if hasattr(cfg, "dt_years") and isinstance(cfg.dt_years, (int, float)):
        dt = float(cfg.dt_years)
        return np.linspace(0.0, dt * (n_times - 1), n_times)

    if hasattr(cfg, "horizon_years") and hasattr(cfg, "n_steps"):
        H = float(cfg.horizon_years)
        n_steps = int(cfg.n_steps)
        # n_times should usually be n_steps + 1
        return np.linspace(0.0, H, n_times)

    # Fallback: monthly steps
    return np.linspace(0.0, (n_times - 1) / 12.0, n_times)


def _build_df0_from_initial_zero_curve(
    z0_pillars: np.ndarray,
    pillars_years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a simple DF(0,T) curve consistent with initial pillar zeros:
      DF(0,T) = exp(-z0(T) * T)

    Returns:
      df0_times_years, df0_values
    """
    df0_times = pillars_years.copy()
    df0_values = np.exp(-z0_pillars * df0_times)
    return df0_times, df0_values


# ------------------------------------------------------------
# Main demo
# ------------------------------------------------------------
def main() -> None:
    # -----------------------------
    # 1) Generate current-model cube
    # -----------------------------
    # -----------------------------
    # 1) Generate current-model cube
    # -----------------------------
    cfg = IrUltimateBaseCurveRunConfig(
        n_paths=3000,
        n_steps=120,  # adjust if your config requires it
        horizon_years=10.0,  # adjust if your config requires it
        seed=1234,
    )

    # Pillars in DAYS (edit to match your curve pillars if you have them elsewhere)
    pillars_days = np.array([30, 60, 90, 180, 365, 730, 1095, 1460, 1825, 2555, 3650], dtype=float)

    gen = IrUltimateBaseCurveScenarioGenerator(pillars_days=pillars_days)

    # Time grid from cfg
    time_grid_years = np.linspace(0.0, cfg.horizon_years, cfg.n_steps + 1)

    # Simple DF(0,t) for demo (replace with your market DF builder if you have one)
    flat_r = 0.02
    df0 = lambda t: float(np.exp(-flat_r * t))

    K = len(pillars_days)

    # Demo model inputs (replace with historical calibration if available)
    corr = np.eye(K)
    sigma = 0.01 * np.ones(K)  # vol per pillar (in driver units used by your process)
    lam = 0.10 * np.ones(K)  # mean reversion per pillar
    shift_bp = np.zeros(K)  # or e.g. 50.0*np.ones(K) if your process expects a positive shift

    current = gen.generate(
        time_grid=time_grid_years,
        df0=df0,
        corr=corr,
        sigma=sigma,
        lam=lam,
        shift_bp=shift_bp,
        run=cfg,
    )

    # cfg = IrUltimateBaseCurveRunConfig()
    # gen = IrUltimateBaseCurveScenarioGenerator(cfg)
    # current = gen.generate()



    current_cube = _get_zero_rates_cube(current)         # (P, T, K)
    pillars_years = _get_pillars_years(current)           # (K,)
    time_grid_years = _get_time_grid_years(current, cfg, current_cube)  # (T,)

    print("Current model rates cube:", current_cube.shape)
    print("Time grid len:", time_grid_years.shape[0], " Pillars:", pillars_years.shape[0])

    # -----------------------------
    # 2) Build DF(0,T) for HW1F benchmark
    # -----------------------------
    # Use the initial curve at t=0. For robustness: average across paths at t=0.
    z0 = np.mean(current_cube[:, 0, :], axis=0)  # (K,)
    df0_times, df0_values = _build_df0_from_initial_zero_curve(z0, pillars_years)

    # HW1F params (benchmark)
    # You can later plug in calibrated (a, sigma) if available.
    hw_params = HullWhite1FParams(a=0.03, sigma=0.01)

    # Convert pillars to days for the HW function
    pillars_days = pillars_years * 365.0

    # -----------------------------
    # 3) Simulate HW1F benchmark cube on the SAME grids
    # -----------------------------
    hw_cube = simulate_hw1f_curve_paths(
        n_paths=current_cube.shape[0],
        time_grid_years=time_grid_years,
        pillars_days=pillars_days,
        df0_curve_times=df0_times,
        df0_curve_values=df0_values,
        params=hw_params,
        seed=1234,
    )
    print("HW1F benchmark rates cube:", hw_cube.shape)

    # -----------------------------
    # 4) Finding 2 tests (Current)
    # -----------------------------
    sens_cur = run_interpolation_sensitivity(current_cube, pillars_years)
    rough_cur = run_forward_roughness(current_cube, pillars_years)
    dens_cur = run_pillar_density_stress(current_cube, pillars_years, scheme="logdf")

    # -----------------------------
    # 5) Finding 2 tests (HW1F)
    # -----------------------------
    sens_hw = run_interpolation_sensitivity(hw_cube, pillars_years)
    rough_hw = run_forward_roughness(hw_cube, pillars_years)
    dens_hw = run_pillar_density_stress(hw_cube, pillars_years, scheme="logdf")

    # -----------------------------
    # 6) Plots
    # -----------------------------
    save_finding2_plots("outputs/finding2", sens_cur, rough_cur, dens_cur, prefix="current")
    save_finding2_plots("outputs/finding2", sens_hw, rough_hw, dens_hw, prefix="hw1f")

    # -----------------------------
    # 7) Console summary (quick materiality feel)
    # -----------------------------
    t_last = sens_cur["rms_time_med"].shape[0] - 1
    print("\n=== Finding 2 Summary (median RMS at final time index) ===")
    print(f"Current interp-scheme RMS median(t_last): {sens_cur['rms_time_med'][t_last]:.6g}")
    print(f"HW1F    interp-scheme RMS median(t_last): {sens_hw['rms_time_med'][t_last]:.6g}")
    print(f"Current pillar-density RMS median(t_last): {dens_cur['rms_time_med'][t_last]:.6g}")
    print(f"HW1F    pillar-density RMS median(t_last): {dens_hw['rms_time_med'][t_last]:.6g}")

    print("\nPlots written to: outputs/finding2/")


if __name__ == "__main__":
    main()
