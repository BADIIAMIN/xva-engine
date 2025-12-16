import numpy as np
from typing import Tuple


class StatsTests:
    """Wrapper for statistical tests (KS, AD, CvM, etc.)."""

    @staticmethod
    def ks_test(sample: np.ndarray, cdf) -> Tuple[float, float]:
        """Return (statistic, pvalue). Placeholder."""
        # TODO: call scipy.stats.ks_1samp or custom
        return 0.0, 1.0
