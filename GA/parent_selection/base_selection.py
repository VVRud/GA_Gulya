from abc import ABC, abstractmethod

import numpy as np
import pygad


class BaseSelection(ABC):
    """
    Base class for all selection methods.
    """

    @abstractmethod
    def select(
        self,
        last_generation_fitness: np.ndarray[np.float64],
        num_parents_mating: int,
        ga_instance: pygad.GA,
    ) -> tuple[np.ndarray[np.int64, np.int64], np.ndarray[np.int64]]:
        """
        Select the parents for the next generation.

        Args:
            last_generation_fitness: The fitness of the last generation.
            num_parents_mating: The number of parents to select.
            ga_instance: The GA instance.
        Returns:
            The selected parents.
        """
        pass
