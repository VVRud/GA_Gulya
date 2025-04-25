import numpy as np

from .base import BaseFitnessFunction


class Deb2Function(BaseFitnessFunction):
    """
    Deb2 function is a multi-modal function with a global maximum at x = 0.1.

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
        return (0.0, 1.023)

    @property
    def global_max_x(self) -> list[float]:
        """
        Calculate the global maximum of the function.

        Returns:
            The global maximum of the function.
        """
        return np.ones(self.n, dtype=np.float64) * 0.1

    def _fitness_func_impl(self, x: list[float]) -> float:
        """
        Calculate the fitness of an individual.

        Args:
            x: The individual to calculate the fitness of.

        Returns:
            The fitness of the individual.
        """
        return np.sum(
            self._exp_func(x[i]) * self._sin_part(x[i]) for i in range(self.n)
        )

    def _fitness_func_many_impl(self, x: list[list[float]]) -> list[float]:
        """
        Calculate the fitness of many individuals.

        Args:
            x: The individuals to calculate the fitness of.

        Returns:
            The fitness of the individuals.
        """
        return np.sum(self._exp_func(x) * self._sin_part(x), axis=1)

    def _exp_func(self, x: float) -> float:
        """
        Calculate the exponential function.

        Args:
            x: The value to calculate the exponential function of.

        Returns:
            The value of the exponential function.
        """
        return np.exp(-2 * np.log(2) * np.power((x - 0.1) / 0.8, 2))

    def _sin_part(self, x: float) -> float:
        """
        Calculate the sine function.

        Args:
            x: The value to calculate the sine function of.

        Returns:
            The value of the sine function.
        """
        return np.power(np.sin(5 * np.pi * x), 6)
