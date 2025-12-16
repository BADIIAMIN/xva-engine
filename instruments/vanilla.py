from dataclasses import dataclass
from .base import Instrument


@dataclass
class EuropeanOption(Instrument):
    """
    Plain-vanilla European option on a single underlying.

    Parameters
    ----------
    id : str
        Trade identifier.
    underlying : str
        Name of the underlying risk factor (e.g. ``"EQ.TEST"``).
    strike : float
        Strike price.
    maturity : float
        Option maturity (year fraction from valuation date).
    option_type : str
        Either ``"call"`` or ``"put"``.
    """

    underlying: str
    strike: float
    maturity: float
    option_type: str = "call"

    def get_cashflows(self):
        """
        For a European option the payoff is a single maturity cashflow
        depending on the underlying terminal value.

        Returns
        -------
        dict
            Placeholder representation of the payoff structure.
        """
        return {
            "type": "european_option",
            "underlying": self.underlying,
            "strike": self.strike,
            "maturity": self.maturity,
            "option_type": self.option_type,
        }
