from dataclasses import dataclass


@dataclass
class CSA:
    """CSA terms for collateralisation."""
    threshold_bank: float
    threshold_counterparty: float
    mta_bank: float
    mta_counterparty: float
    rounding_bank: float
    rounding_counterparty: float
    mpor_days: int
