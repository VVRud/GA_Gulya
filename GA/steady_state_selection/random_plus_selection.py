import numpy as np
import pygad

from .base_selection import BaseSelection


class RandomPlusSelection(BaseSelection):
    """
    Remove random individuals from the concatenation of the population and
    the offsprings.
    This is a (μ+λ) selection strategy where random individuals from both
    parents and offsprings are selected for the next generation.
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

        combined_population = np.concatenate([population, offsprings])
        np.random.shuffle(combined_population)
        return combined_population[: ga_instance.sol_per_pop]
