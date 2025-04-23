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
        if offsprings.size == 0:
            return population

        num_parents_select = ga_instance.sol_per_pop - offsprings.shape[0]
        next_population = np.empty_like(population)

        # Randomly select individuals from the population
        selected_parents_indices = np.random.choice(
            np.arange(population.shape[0]),
            size=num_parents_select,
            replace=False,
        )

        next_population[:num_parents_select] = population[selected_parents_indices]
        next_population[num_parents_select:] = offsprings

        return next_population
