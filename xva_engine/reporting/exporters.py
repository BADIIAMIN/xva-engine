from ..core.cube import ExposureCube, RiskFactorCube
from typing import Any


class ReportExporter:
    """Export cubes and metrics to files (CSV, Parquet, Excel, …)."""

    def export_exposure(self, cube: ExposureCube, output_cfg: Any):
        pass

    def export_risk_factors(self, cube: RiskFactorCube, output_cfg: Any):
        pass

    def export_xva(self, xva_results: Any, output_cfg: Any):
        pass
