import numpy as np

from xva_engine.simulation.generators.ir_ultimate_base_curve_generator import (
    IrUltimateBaseCurveScenarioGenerator,
    IrUltimateBaseCurveRunConfig,
)

# Example discount curve callable (replace with your actual curve object)
def df0(t: float) -> float:
    # dummy: flat 3% cont rate
    r = 0.03
    return np.exp(-r * t)

if __name__ == "__main__":
    pillars_days = np.array([30, 90, 180, 365, 730, 1825, 3650], dtype=float)
    time_grid = np.linspace(0.0, 5.0, 121)  # 5y monthly

    gen = IrUltimateBaseCurveScenarioGenerator(pillars_days=pillars_days)

    # Dummy historical rates (Nobs,K) in rate units
    rng = np.random.default_rng(0)
    rates_hist = 0.03 + 0.002 * rng.standard_normal(size=(800, len(pillars_days)))

    corr, sigma, lam = gen.calibrate_historical(rates_hist=rates_hist, lam=0.08, shift_bp=100.0)

    out = gen.generate(
        time_grid=time_grid,
        df0=df0,
        corr=corr,
        sigma=sigma,
        lam=lam,
        shift_bp=np.full(len(pillars_days), 100.0),
        run=IrUltimateBaseCurveRunConfig(n_paths=2000, seed=42, return_driver=False),
    )

    print(out["rates"].shape)  # (paths, times, pillars)
