import numpy as np
import pygad
from typing import List


class UniformMutation:
    def __init__(self, mutation_probability: float):
        self.mutation_probability = mutation_probability

    def mutate(self, solution: List[int], pygad_instance: pygad.GA) -> List[int]:
        """
        Mutate the solution by flipping bits based on mutation probability.
        
        The mutation operator works by generating a random mask of the same length as the solution,
        where each element is 1 with probability equal to the mutation probability. For each 1 in
        the mask, the corresponding bit in the solution is flipped (0->1 or 1->0).
        Args:
            solution: The solution to mutate.
            pygad_instance: The pygad instance.
        Returns:
            The mutated solution.
        """
        mask = np.random.uniform(0, 1, solution.shape) < self.mutation_probability
        return np.where(mask, 1 - solution, solution)
