import numpy as np
from .metrics import pillars_years, implied_forward


def run_forward_sanity_test(rates, pillars_days, i, j):
    M = pillars_years(pillars_days)
    r_i = rates[:, :, i]
    r_j = rates[:, :, j]
    fwd = implied_forward(r_i, r_j, M[i], M[j])
    return {
        "mean": float(np.mean(fwd)),
        "std": float(np.std(fwd)),
        "p05": float(np.quantile(fwd, 0.05)),
        "p95": float(np.quantile(fwd, 0.95)),
    }
