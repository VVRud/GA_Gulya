import numpy as np
import pygad

from GA.selection.base_with_probabilities import BaseWithProbabilities


class SUSSelection(BaseWithProbabilities):
    """
    Stochastic Universal Sampling (SUS) is a selection method used in
    genetic algorithms. It works by selecting parents based on their
    fitness values, with higher fitness values having a higher probability
    of being selected.
    """

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
        probabilities = self.get_probabilities(last_generation_fitness)
        cumulative_probabilities = np.cumsum(probabilities)
        start_pointer = np.random.uniform(0, 1 / num_parents_mating)
        pointers = [
            start_pointer + i / num_parents_mating for i in range(num_parents_mating)
        ]
        parents_indices = np.searchsorted(cumulative_probabilities, pointers)
        parents = ga_instance.population[parents_indices]
        return parents, parents_indices
