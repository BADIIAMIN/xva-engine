import numpy as np
from .csa import CSA
from ..core.cube import ExposureCube


class CollateralEngine:
    """
    Apply CSA logic to an uncollateralised exposure cube to obtain
    collateralised exposures.

    Notes
    -----
    The implementation here is a simplified example for demonstration:
    - Zero thresholds
    - Zero MTA
    - Margining at each time step
    - Collateral posted by the counterparty is equal to the negative
      exposure (fully collateralised against negative exposure).
    """

    def apply_csa(self, exposure: ExposureCube, csa: CSA) -> ExposureCube:
        """
        Apply a simplified CSA to the given exposure cube.

        Parameters
        ----------
        exposure : ExposureCube
            Uncollateralised trade/portfolio values.
        csa : CSA
            CSA terms. Only used for documentation in this toy implementation.

        Returns
        -------
        ExposureCube
            Collateralised exposure cube.
        """
        data = exposure.data.copy()
        # Collateral from the bank POV: if exposure < 0, we hold collateral from cpty
        # so effective exposure = max(exposure, 0).
        data_collateralised = np.maximum(data, 0.0)

        return ExposureCube(
            data=data_collateralised,
            scenarios=exposure.scenarios,
            time_grid=exposure.time_grid,
            trades=exposure.trades,
        )
