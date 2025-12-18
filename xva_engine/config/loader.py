import yaml
from .schema import RunConfig, MarketDataConfig, SimulationConfig, PricingConfig, \
    CollateralConfig, XVAConfig, BacktestingConfig, OutputConfig


def load_config(path: str) -> RunConfig:
    """Load YAML/JSON config into RunConfig dataclass."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Here you’d map raw dicts into dataclasses properly.
    # For now, a minimal placeholder:
    return RunConfig(
        market_data=MarketDataConfig(**raw["market_data"]),
        simulation=SimulationConfig(**raw["simulation"]),
        pricing=PricingConfig(**raw["pricing"]),
        collateral=CollateralConfig(**raw["collateral"]),
        xva=XVAConfig(**raw["xva"]),
        backtesting=BacktestingConfig(**raw["backtesting"]),
        output=OutputConfig(**raw["output"]),
    )
