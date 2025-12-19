import numpy as np

from xva_engine.simulation.generators.ir_ultimate_base_curve_generator import (
    IrUltimateBaseCurveScenarioGenerator,
    IrUltimateBaseCurveRunConfig,
)

def build_df0_from_zero_curve(pillars_years, zero_rates):
    # continuous comp
    def df0(t):
        if t <= 0.0:
            return 1.0
        z = float(np.interp(t, pillars_years, zero_rates))
        return float(np.exp(-z * t))
    return df0


if __name__ == "__main__":
    pillars_days = np.array([30, 90, 180, 365, 730, 1825, 3650, 7300], dtype=float)
    pillars_years = pillars_days / 365.0

    # simple upward sloping initial curve
    zero0 = 0.02 + 0.005 * np.log1p(pillars_years)
    df0 = build_df0_from_zero_curve(pillars_years, zero0)

    time_grid = np.linspace(0.0, 10.0, 121)  # monthly to 10y

    gen = IrUltimateBaseCurveScenarioGenerator(pillars_days=pillars_days)

    # dummy historical rates for calibration (Nobs,K)
    rng = np.random.default_rng(0)
    hist = (zero0[None, :] + 0.002 * rng.standard_normal(size=(1200, len(pillars_days))))

    corr, sigma, lam = gen.calibrate_historical(rates_hist=hist, lam=0.08, shift_bp=100.0)

    out = gen.generate(
        time_grid=time_grid,
        df0=df0,
        corr=corr,
        sigma=sigma,
        lam=lam,
        shift_bp=np.full(len(pillars_days), 100.0),
        run=IrUltimateBaseCurveRunConfig(n_paths=3000, seed=42, return_driver=False),
    )

    print("Current model rates cube:", out["rates"].shape)
