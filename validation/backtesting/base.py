from abc import ABC, abstractmethod


class BacktestResult:
    """Placeholder for backtest results (metrics, p-values, charts refs, etc.)."""
    pass


class Backtest(ABC):
    """Base interface for all backtests."""

    @abstractmethod
    def run(self) -> BacktestResult:
        raise NotImplementedError
