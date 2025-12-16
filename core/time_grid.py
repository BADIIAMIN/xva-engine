from dataclasses import dataclass
from typing import List


@dataclass
class TimeGrid:
    """
    Simulation time grid.

    Parameters
    ----------
    times : list of float
        Monotonically increasing time points (in year fractions)
        at which the simulation state is stored.

    Notes
    -----
    The `TimeGrid` is a core primitive shared by the simulation,
    pricing and aggregation layers. All scenario cubes index time
    via an instance of this class.
    """
    times: List[float]

    def as_array(self):
        """
        Return the time points as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_times,)`` with floating point times.
        """
        import numpy as np
        return np.array(self.times, dtype=float)

    def index_of(self, t: float) -> int:
        """
        Return the index of a given time in the grid.

        Parameters
        ----------
        t : float
            Time to look up.

        Returns
        -------
        int
            Index of the matching time in ``times``.

        Raises
        ------
        ValueError
            If the time is not present in the grid.
        """
        return self.times.index(t)
