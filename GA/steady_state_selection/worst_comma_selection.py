import numpy as np
import pygad

from .base_selection import BaseSelection


class WorstCommaSelection(BaseSelection):
    """
    Remove worst individuals from the population and replace them with the offsprings.
    This is a (μ,λ) selection strategy where μ is the number of parents to keep and
    λ is the number of offsprings.
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

        population = population[np.argsort(population_fitness)]
        population[:ga_instance.num_offspring] = offsprings

        return population
