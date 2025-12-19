import numpy as np

from xva_engine.simulation.generators.benchmarks.ir_hull_white_1f_generator import (
    HullWhite1FParams,
    simulate_hw1f_curve_paths,
)

if __name__ == "__main__":
    pillars_days = np.array([30, 90, 180, 365, 730, 1825, 3650, 7300], dtype=float)
    pillars_years = pillars_days / 365.0

    # initial zero curve
    zero0 = 0.02 + 0.005 * np.log1p(pillars_years)

    # build DF(0,t) grid for HW input
    df_times = np.linspace(0.0, 40.0, 4001)
    # interpolate zero on df_times
    zt = np.interp(df_times, pillars_years, zero0, left=zero0[0], right=zero0[-1])
    df_vals = np.exp(-zt * df_times)
    df_vals[0] = 1.0

    time_grid = np.linspace(0.0, 10.0, 121)

    rates_hw = simulate_hw1f_curve_paths(
        n_paths=3000,
        time_grid_years=time_grid,
        pillars_days=pillars_days,
        df0_curve_times=df_times,
        df0_curve_values=df_vals,
        params=HullWhite1FParams(a=0.05, sigma=0.01),
        seed=123,
    )

    print("HW1F benchmark rates cube:", rates_hw.shape)
