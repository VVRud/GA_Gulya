import numpy as np
from .base_with_probabilities import BaseWithProbabilities
from abc import ABC


class BaseLinRank(BaseWithProbabilities, ABC):
    """
    Base class for linear rank selection.
    
    Args:
        beta: The beta parameter for the linear rank selection.
    """
    def __init__(self, beta: float):
        self.beta = beta

    def get_probabilities(self, last_generation_fitness: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Get the probabilities of the solutions.
        Args:
            last_generation_fitness: The fitness of the last generation.
        Returns:
            The probabilities of the solutions.
        """
        indices_sorted = np.argsort(last_generation_fitness)
        N = last_generation_fitness.shape[0]
        ranks = np.arange(N)[::-1]
        constant = (2 - self.beta) / N
        probabilities = constant + (
            2 * ranks * (self.beta - 1)
            / (N * (N - 1))
        )
        return probabilities[indices_sorted]
