from dataclasses import dataclass
import numpy as np


@dataclass
class CorrelationModel:
    """Encapsulates correlation structure and sampling."""
    corr_matrix: np.ndarray

    def cholesky(self) -> np.ndarray:
        """Return Cholesky factor."""
        return np.linalg.cholesky(self.corr_matrix)

    def sample_normals(
        self,
        rng: np.random.Generator,
        n_scenarios: int,
        n_steps: int
    ) -> np.ndarray:
        """
        Sample correlated standard normals using Cholesky.
        Shape: [n_scenarios, n_steps, n_factors]
        """
        n_factors = self.corr_matrix.shape[0]
        L = self.cholesky()
        z = rng.standard_normal(size=(n_scenarios, n_steps, n_factors))
        return z @ L.T
