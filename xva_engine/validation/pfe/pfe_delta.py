from __future__ import annotations
import numpy as np


def pfe_profile(exposures: np.ndarray, q: float) -> np.ndarray:
    """
    exposures: (P, T) exposure paths
    returns: (T,) PFE profile at quantile q
    """
    return np.quantile(exposures, q, axis=0)


def pfe_delta(expo_a: np.ndarray, expo_b: np.ndarray, q: float) -> dict:
    """
    Compare two exposure cubes: a - b
    """
    pfe_a = pfe_profile(expo_a, q)
    pfe_b = pfe_profile(expo_b, q)
    delta = pfe_a - pfe_b

    rel = np.divide(delta, np.maximum(np.abs(pfe_b), 1e-12))

    return {
        "pfe_a": pfe_a,
        "pfe_b": pfe_b,
        "delta": delta,
        "rel_delta": rel,
        "max_abs_delta": float(np.max(np.abs(delta))),
        "max_rel_delta": float(np.max(np.abs(rel))),
    }
