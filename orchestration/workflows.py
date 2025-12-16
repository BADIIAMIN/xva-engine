from ..collateral.csa import CSA
from ..collateral.engine import CollateralEngine


def run_single_equity_option_with_csa_demo(params: Dict[str, Any]) -> DemoResult:
    """
    Same as `run_single_equity_option_demo`, but applies a simple CSA
    and returns collateralised exposure EE and PFE as well.

    Returns
    -------
    DemoResult
        The `exp_cube` in this result will be the collateralised one.
        If you need both, adjust the container or return a tuple.
    """
    base_result = run_single_equity_option_demo(params)

    # Define a very simple CSA (all thresholds = 0 for full collateralisation)
    csa = CSA(
        threshold_bank=0.0,
        threshold_counterparty=0.0,
        mta_bank=0.0,
        mta_counterparty=0.0,
        rounding_bank=0.0,
        rounding_counterparty=0.0,
        mpor_days=10,
    )

    coll_engine = CollateralEngine()
    coll_cube = coll_engine.apply_csa(base_result.exp_cube, csa)

    # Recompute EE and PFE on collateralised cube
    ee_coll = ExposureMetrics.compute_EE(coll_cube)
    pfe_95_coll = ExposureMetrics.compute_PFE(coll_cube, alpha=0.95)

    # Overwrite exposure/metrics in result for simplicity
    base_result.exp_cube = coll_cube
    base_result.ee = ee_coll
    base_result.pfe_95 = pfe_95_coll

    return base_result
