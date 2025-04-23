import numpy as np
import pygad

from .base_selection import BaseSelection


class WorstPlusSelection(BaseSelection):
    """
    Select the best individuals from the concatenation of the population and
    the offsprings.
    This is a (μ+λ) selection strategy where the best individuals from both
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

        # Calculate fitness for the offsprings
        offsprings_fitness = ga_instance.fitness_func(
            ga_instance, offsprings, np.arange(offsprings.shape[0])
        )

        # Combine fitness values
        combined_fitness = np.concatenate([population_fitness, offsprings_fitness])

        # Select the best individuals based on fitness
        selected_indices = np.argsort(combined_fitness)[::-1][: ga_instance.sol_per_pop]

        return combined_population[selected_indices]
