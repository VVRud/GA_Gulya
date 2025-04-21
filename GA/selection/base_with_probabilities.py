from .base_selection import BaseSelection
import numpy as np
from abc import ABC, abstractmethod


class BaseWithProbabilities(BaseSelection, ABC):
    """Base class for selection methods that use probabilities."""
    
    def get_probabilities(self, last_generation_fitness: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Get the probabilities of the solutions.
        """
        probabilities = last_generation_fitness / np.sum(last_generation_fitness)
        return probabilities
