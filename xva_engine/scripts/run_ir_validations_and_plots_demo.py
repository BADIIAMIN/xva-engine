import os
import numpy as np

from xva_engine.simulation.generators.ir_ultimate_base_curve_generator import (
    IrUltimateBaseCurveScenarioGenerator,
    IrUltimateBaseCurveRunConfig,
)
from xva_engine.simulation.generators.benchmarks.ir_hull_white_1f_generator import (
    HullWhite1FParams,
    simulate_hw1f_curve_paths,
)

from xva_engine.validation.ir.arbitrage_free.test_df_monotonicity import run_df_monotonicity_test
from xva_engine.validation.ir.arbitrage_free.test_forward_reconstruction import run_forward_sanity_test
from xva_engine.validation.ir.arbitrage_free.test_discount_factor_wedge import run_df_wedge_test
from xva_engine.validation.ir.arbitrage_free.metrics import kink_index, pillars_years
from xva_engine.validation.ir.reporting.plots_arbitrage import (
    plot_df_monotonicity_heatmap,
    plot_kink_index_bands,
    plot_wedge_histogram,
    plot_wedge_vs_maturity,
)

def build_df0_from_zero_curve(pillars_years, zero_rates):
    def df0(t):
        if t <= 0.0:
            return 1.0
        z = float(np.interp(t, pillars_years, zero_rates))
        return float(np.exp(-z * t))
    return df0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    out_dir = "outputs/finding1"
    ensure_dir(out_dir)

    pillars_days = np.array([30, 90, 180, 365, 730, 1825, 3650, 7300], dtype=float)
    M_years = pillars_years(pillars_days)

    # initial zero curve
    zero0 = 0.02 + 0.005 * np.log1p(M_years)

    time_grid = np.linspace(0.0, 10.0, 121)  # monthly to 10y
    df0 = build_df0_from_zero_curve(M_years, zero0)

    # --- Current model scenarios ---
    gen = IrUltimateBaseCurveScenarioGenerator(pillars_days=pillars_days)
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
        run=IrUltimateBaseCurveRunConfig(n_paths=2000, seed=42, return_driver=False),
    )
    rates_cur = out["rates"]

    # --- HW1F benchmark scenarios ---
    df_times = np.linspace(0.0, 40.0, 4001)
    zt = np.interp(df_times, M_years, zero0, left=zero0[0], right=zero0[-1])
    df_vals = np.exp(-zt * df_times)
    df_vals[0] = 1.0

    rates_hw = simulate_hw1f_curve_paths(
        n_paths=2000,
        time_grid_years=time_grid,
        pillars_days=pillars_days,
        df0_curve_times=df_times,
        df0_curve_values=df_vals,
        params=HullWhite1FParams(a=0.05, sigma=0.01),
        seed=123,
    )

    # --- Run Finding 1 validations (both models) ---
    res_mon_cur = run_df_monotonicity_test(rates_cur, pillars_days)
    res_mon_hw  = run_df_monotonicity_test(rates_hw, pillars_days)

    kink_cur = kink_index(rates_cur)
    kink_hw  = kink_index(rates_hw)

    # forward sanity on two pillars (e.g. 1Y vs 10Y)
    i_1y = int(np.argmin(np.abs(M_years - 1.0)))
    j_10y = int(np.argmin(np.abs(M_years - 10.0))) if np.max(M_years) >= 10.0 else (len(M_years)-1)

    res_fwd_cur = run_forward_sanity_test(rates_cur, pillars_days, i_1y, j_10y)
    res_fwd_hw  = run_forward_sanity_test(rates_hw, pillars_days, i_1y, j_10y)

    # wedge on a long pillar index at first step
    base_k = len(pillars_days) - 1  # longest maturity
    step_index = 0
    res_wedge_cur = run_df_wedge_test(rates_cur, time_grid, pillars_days, base_k, step_index)
    res_wedge_hw  = run_df_wedge_test(rates_hw, time_grid, pillars_days, base_k, step_index)

    # wedge vs maturity (compute p95(|wedge|) for each pillar at same step)
    wedge_stat_cur = np.zeros(len(pillars_days))
    wedge_stat_hw  = np.zeros(len(pillars_days))
    for k in range(len(pillars_days)):
        if M_years[k] <= (time_grid[1] - time_grid[0]) * 1.5:
            wedge_stat_cur[k] = np.nan
            wedge_stat_hw[k]  = np.nan
            continue
        wc = run_df_wedge_test(rates_cur, time_grid, pillars_days, k, step_index)["wedge"]
        wh = run_df_wedge_test(rates_hw, time_grid, pillars_days, k, step_index)["wedge"]
        wedge_stat_cur[k] = np.quantile(np.abs(wc), 0.95)
        wedge_stat_hw[k]  = np.quantile(np.abs(wh), 0.95)

    # --- Save plots ---
    fig = plot_df_monotonicity_heatmap(res_mon_cur["freq_time_pillar"], "Current model: DF monotonicity violation frequency")
    fig.savefig(os.path.join(out_dir, "current_df_monotonicity_heatmap.png"), dpi=200)

    fig = plot_df_monotonicity_heatmap(res_mon_hw["freq_time_pillar"], "HW1F: DF monotonicity violation frequency")
    fig.savefig(os.path.join(out_dir, "hw_df_monotonicity_heatmap.png"), dpi=200)

    fig = plot_kink_index_bands(kink_cur, time_grid, "Current model: kink index bands")
    fig.savefig(os.path.join(out_dir, "current_kink_bands.png"), dpi=200)

    fig = plot_kink_index_bands(kink_hw, time_grid, "HW1F: kink index bands")
    fig.savefig(os.path.join(out_dir, "hw_kink_bands.png"), dpi=200)

    fig = plot_wedge_histogram(res_wedge_cur["wedge"], f"Current model: wedge histogram (T={res_wedge_cur['T']:.2f}y, u={res_wedge_cur['u']:.3f}y)")
    fig.savefig(os.path.join(out_dir, "current_wedge_hist.png"), dpi=200)

    fig = plot_wedge_histogram(res_wedge_hw["wedge"], f"HW1F: wedge histogram (T={res_wedge_hw['T']:.2f}y, u={res_wedge_hw['u']:.3f}y)")
    fig.savefig(os.path.join(out_dir, "hw_wedge_hist.png"), dpi=200)

    fig = plot_wedge_vs_maturity(M_years, wedge_stat_cur, "Current model: p95(|wedge|) vs maturity")
    fig.savefig(os.path.join(out_dir, "current_wedge_vs_maturity.png"), dpi=200)

    fig = plot_wedge_vs_maturity(M_years, wedge_stat_hw, "HW1F: p95(|wedge|) vs maturity")
    fig.savefig(os.path.join(out_dir, "hw_wedge_vs_maturity.png"), dpi=200)

    # --- Print summary (useful in logs) ---
    print("\n=== Finding 1 Summary ===")
    print("Current DF monotonicity violation rate:", res_mon_cur["violation_rate"], "max increase:", res_mon_cur["max_increase"])
    print("HW1F    DF monotonicity violation rate:", res_mon_hw["violation_rate"],  "max increase:", res_mon_hw["max_increase"])
    print("Current forward sanity (1Y->long):", res_fwd_cur)
    print("HW1F    forward sanity (1Y->long):", res_fwd_hw)
    print("Current wedge stats:", {k: res_wedge_cur[k] for k in ["wedge_mean", "wedge_p05", "wedge_p95", "frac_abs_gt_1bp"]})
    print("HW1F    wedge stats:", {k: res_wedge_hw[k]  for k in ["wedge_mean", "wedge_p05", "wedge_p95", "frac_abs_gt_1bp"]})
    print(f"\nPlots saved under: {out_dir}\n")
