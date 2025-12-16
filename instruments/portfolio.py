from dataclasses import dataclass, field
from typing import List
from .base import Instrument


@dataclass
class Portfolio:
    """Simple container for a list of instruments."""
    trades: List[Instrument] = field(default_factory=list)

    def add(self, trade: Instrument) -> None:
        self.trades.append(trade)
