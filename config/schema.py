from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class MarketDataConfig:
    source_type: str
    source_params: Dict[str, Any]
    as_of_date: str


@dataclass
class SimulationConfig:
    measure: str  # "RN" or "P"
    n_scenarios: int
    n_steps: int
    models: List[Dict[str, Any]]  # list of model configs
    correlation: Dict[str, Any]


@dataclass
class PricingConfig:
    engine_type: str  # "analytic", "lsmc", "external"
    external_engine: Optional[str] = None  # "quantlib", "ore", "murex"


@dataclass
class CollateralConfig:
    enabled: bool
    csa_params: Dict[str, Any]


@dataclass
class XVAConfig:
    enabled: bool
    xva_types: List[str]  # ["CVA", "DVA", "FVA", "MVA", "CollVA"]
    pd_lgd_source: Dict[str, Any]


@dataclass
class BacktestingConfig:
    enabled: bool
    tests: List[Dict[str, Any]]


@dataclass
class OutputConfig:
    output_dir: str
    formats: List[str]  # ["csv", "parquet", "xlsx"]


@dataclass
class RunConfig:
    market_data: MarketDataConfig
    simulation: SimulationConfig
    pricing: PricingConfig
    collateral: CollateralConfig
    xva: XVAConfig
    backtesting: BacktestingConfig
    output: OutputConfig
