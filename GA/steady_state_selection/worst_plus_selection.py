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
        if ga_instance.num_offspring == 0:
            return population

        combined_population = np.concatenate([population, offsprings])
        offsprings_fitness = ga_instance.fitness_func(
            ga_instance, offsprings, np.arange(ga_instance.num_offspring)
        )
        combined_fitness = np.concatenate([population_fitness, offsprings_fitness])
        return combined_population[np.argsort(combined_fitness)][ga_instance.sol_per_pop:]
