from dataclasses import dataclass
from ..market_data.environment import MarketDataEnvironment


@dataclass
class PricingContext:
    """Pricing context: market data + valuation settings."""
    market_env: MarketDataEnvironment
    valuation_date: str
    measure: str  # "RN" or "P"
