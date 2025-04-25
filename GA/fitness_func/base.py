from abc import ABC, abstractmethod
from functools import cache

import matplotlib.pyplot as plt
import numpy as np


class BaseFitnessFunction(ABC):
    """
    Base class for fitness functions.

    Args:
        n: The number of dimensions of the fitness function.
    """

    def __init__(self, n: int):
        self.n = n

    @property
    @abstractmethod
    def range(self) -> tuple[float, float]:
        """
        The range of the fitness function.

        Returns:
            Tuple[float, float]: The range of the fitness function.
        """
        pass

    @property
    @abstractmethod
    def global_max_x(self) -> list[float]:
        """
        The global maximum of the fitness function.

        Returns:
            List[float]: The global maximum of the fitness function.
        """
        pass

    @property
    def global_max_y(self) -> float:
        """
        The global maximum of the fitness function.

        Returns:
            float: The global maximum of the fitness function.
        """
        return self.fitness_func(self.global_max_x)

    def validate_input(self, x: list[float]):
        """
        Validate the input to the fitness function.

        Args:
            x: The input to the fitness function.
        """
        if isinstance(x, np.ndarray):
            x = x.tolist()
        assert len(x) == self.n, (
            f"The number of dimensions must be equal to n = {self.n}"
        )

    def fitness_func(self, x: list[float]) -> float:
        """
        The fitness function.

        Args:
            x: The input to the fitness function.

        Returns:
            The fitness of the individual.
        """
        self.validate_input(x)
        return self._fitness_func_impl(x)

    @abstractmethod
    @cache
    def _fitness_func_impl(self, x: list[float]) -> float:
        """
        The fitness function.

        Args:
            x: The input to the fitness function.

        Returns:
            The fitness of the individual.
        """
        pass

    def fitness_func_many(self, x: list[list[float]]) -> list[float]:
        """
        The fitness function for many individuals.
        """
        self.validate_input(x)
        return self._fitness_func_many_impl(x)

    @abstractmethod
    @cache
    def _fitness_func_many_impl(self, x: list[list[float]]) -> list[float]:
        """
        The fitness function for many individuals.

        Args:
            x: The input to the fitness function.

        Returns:
            The fitness of the individuals.
        """
        pass

    def plot(self, num_points=1_000):
        """
        Plot the fitness function.

        Args:
            num_points: The number of points to plot.
        """

        if self.n != 1:
            raise ValueError(
                "The fitness function must be a single-dimensional function"
                "to support plotting."
            )

        x = np.linspace(self.range[0], self.range[1], num_points)
        y = np.array([self.fitness_func([xi]) for xi in x])

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, "b-", label="Function")

        # Plot global maximum
        plt.plot(
            self.global_max_x[0],
            self.global_max_y,
            "r*",
            markersize=15,
            label="Global Maximum",
        )

        plt.grid(True)
        plt.legend()
        plt.title(self.__class__.__name__)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()
