from GA.selection.base_with_probabilities import BaseWithProbabilities
from abc import ABC
import numpy as np


class BaseExpRank(BaseWithProbabilities, ABC):
    """
    Base class for exponential rank selection.
    
    Args:
        c: The c parameter for the exponential rank selection.
    """
    def __init__(self, c: float):
        self.c = c

    def get_probabilities(self, last_generation_fitness: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """Get the probabilities of the solutions."""
        indices_sorted = np.argsort(last_generation_fitness)
        N = last_generation_fitness.shape[0]
        ranks = np.arange(N)[::-1]
        constant = (self.c - 1) / (self.c ** N - 1)
        probabilities = constant * np.power(
            self.c,
            N - ranks - 1
        )

        return probabilities[indices_sorted]
