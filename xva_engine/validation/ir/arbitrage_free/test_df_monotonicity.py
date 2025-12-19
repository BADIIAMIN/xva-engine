from .metrics import discount_factors_from_zero_rates, df_monotonicity_violations


def run_df_monotonicity_test(rates, pillars_days, tol=0.0):
    df = discount_factors_from_zero_rates(rates, pillars_days)
    return df_monotonicity_violations(df, tol=tol)
