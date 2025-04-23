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
        population: np.ndarray,
        population_fitness: np.ndarray,
        offsprings: np.ndarray,
        ga_instance: pygad.GA,
    ) -> np.ndarray:
        """
        Select next population from current population and offsprings.

        Args:
            population: The population of the previous generation.
            population_fitness: The fitness of the population.
            offsprings: The offsprings of the current generation.
            ga_instance: The instance of the genetic algorithm.

        Returns:
            The next population.
        """
        pass
