from .metrics import df_wedge_one_step


def run_df_wedge_test(rates, time_grid_years, pillars_days, base_pillar_index, step_index):
    return df_wedge_one_step(
        rates=rates,
        time_grid_years=time_grid_years,
        pillars_days=pillars_days,
        base_pillar_index=base_pillar_index,
        step_index=step_index,
    )
