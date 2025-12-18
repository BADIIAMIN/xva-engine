from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Instrument(ABC):
    """Base class for all instruments."""
    id: str

    @abstractmethod
    def get_cashflows(self):
        """Return a representation of cashflows (to be defined)."""
        raise NotImplementedError
