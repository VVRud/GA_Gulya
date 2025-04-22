import numpy as np
import pygad


class UniformMutation:
    def __init__(self, mutation_probability: float):
        self.mutation_probability = mutation_probability

        # Pre-allocate a seed for random number generation to avoid overhead
        self._rng = np.random.default_rng()

    def mutate(
        self, solution: np.ndarray | list[int], pygad_instance: pygad.GA
    ) -> np.ndarray:
        """
        Mutate the solution by flipping bits based on mutation probability.

        The mutation operator works by generating a random mask of the same length
        as the solution, where each element is 1 with probability equal to
        the mutation probability. For each 1 in the mask, the corresponding bit
        in the solution is flipped (0->1 or 1->0).

        Args:
            solution: The solution to mutate.
            pygad_instance: The pygad instance.
        Returns:
            The mutated solution.
        """
        # Fast random number generation and vectorized operations
        mask = self._rng.random(solution.shape) < self.mutation_probability
        # XOR operation is faster than conditional selection for bit flipping
        return np.logical_xor(solution, mask).astype(solution.dtype)
