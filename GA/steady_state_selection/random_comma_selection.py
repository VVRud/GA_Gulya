import numpy as np
import pygad

from .base_selection import BaseSelection


class RandomCommaSelection(BaseSelection):
    """
    Select random individuals from the population and replace them with the offsprings.
    This is a (μ,λ) selection strategy where μ is randomly chosen from the population
    and λ is the number of offsprings.
    """

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
        if ga_instance.num_offspring == 0:
            return population

        np.random.shuffle(population)
        population[: ga_instance.num_offspring] = offsprings

        return population
