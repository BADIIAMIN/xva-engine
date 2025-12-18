import numpy as np
from ..core.cube import ExposureCube


class ExposureMetrics:
    """Compute standard exposure metrics from an exposure cube."""

    @staticmethod
    def compute_EE(cube: ExposureCube) -> np.ndarray:
        """Expected Exposure as a function of time."""
        # average over scenarios, positive part
        return np.mean(np.maximum(cube.data, 0.0), axis=0)

    @staticmethod
    def compute_EPE_ENE(cube: ExposureCube):
        ee = ExposureMetrics.compute_EE(cube)
        epe = np.mean(ee)
        ene = np.mean(np.minimum(cube.data, 0.0))
        return epe, ene

    @staticmethod
    def compute_PFE(cube: ExposureCube, alpha: float) -> np.ndarray:
        """Potential Future Exposure at quantile alpha as a function of time."""
        # quantile over scenarios of positive exposure
        positive = np.maximum(cube.data, 0.0)
        return np.quantile(positive, alpha, axis=0)

    @staticmethod
    def compute_EEPE(cube: ExposureCube) -> float:
        """EEPE: time-average of EE."""
        ee = ExposureMetrics.compute_EE(cube)
        return float(np.mean(ee))
