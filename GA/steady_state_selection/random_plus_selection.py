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
        if offsprings.size == 0:
            return population

        # Create a combined pool of population and offsprings
        combined_population = np.concatenate([population, offsprings])

        # Randomly select individuals from the combined pool
        selected_indices = np.random.choice(
            np.arange(combined_population.shape[0]),
            size=ga_instance.sol_per_pop,
            replace=False,
        )

        return combined_population[selected_indices]
