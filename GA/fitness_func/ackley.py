import numpy as np

from .base import BaseFitnessFunction


class AckleyFunction(BaseFitnessFunction):
    """
    Ackley function is a multi-modal function with a global maximum at x = 0.0.

    Args:
        n: The number of dimensions of the function.
    """

    @property
    def range(self) -> tuple[float, float]:
        """
        Calculate the range of the function.

        Returns:
            The range of the function.
        """
        return (-5.12, 5.12)

    @property
    def global_max_x(self) -> list[float]:
        """
        Calculate the global maximum of the function.

        Returns:
            The global maximum of the function.
        """
        return np.zeros(self.n, dtype=np.float64)

    def _fitness_func_impl(self, x: list[float]) -> float:
        """
        Calculate the fitness of an individual.

        Args:
            x: The individual to calculate the fitness of.

        Returns:
            The fitness of the individual.
        """
        return 20 * np.exp(
            -0.2 * np.sqrt(np.sum([x_i**2 for x_i in x]) / self.n)
        ) + np.exp(np.sum([np.cos(2 * np.pi * x_i) for x_i in x]) / self.n)

    def _fitness_func_many_impl(self, x: list[list[float]]) -> list[float]:
        """
        Calculate the fitness of many individuals.

        Args:
            x: The individuals to calculate the fitness of.

        Returns:
            The fitness of the individuals.
        """
        return 20 * np.exp(
            -0.2 * np.sqrt(np.sum(np.power(x, 2), axis=1) / self.n)
        ) + np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / self.n)
