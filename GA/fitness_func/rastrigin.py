import numpy as np
from .base import BaseFitnessFunction
from typing import List, Tuple


class RastriginFunction(BaseFitnessFunction):
    """
    Rastrigin function is a multi-modal function with a global maximum at x = 0.0.
    
    Args:
        n: The number of dimensions of the function.
        a: The parameter of the function.
    """
    def __init__(self, n: int, a: int = 7):
        super().__init__(n)
        self.a = a

    @property
    def range(self) -> Tuple[float, float]:
        """
        Calculate the range of the function.
        
        Returns:
            The range of the function.
        """
        return (-5.12, 5.12)

    @property
    def global_max_x(self) -> List[float]:
        """
        Calculate the global maximum of the function.
        
        Returns:
            The global maximum of the function.
        """
        return np.zeros(self.n, dtype=np.float64)

    def fitness_func(self, x: List[float]) -> float:
        """
        Calculate the fitness of an individual.
        
        Args:
            x: The individual to calculate the fitness of.
            
        Returns:
            The fitness of the individual.
        """
        return self.n * np.abs(self._cosine_part(self.a)) + np.sum(
            self._cosine_part(x[i]) for i in range(self.n)
        )
        
    def fitness_func_many(self, x: List[List[float]]) -> List[float]:
        """
        Calculate the fitness of many individuals.
        
        Args:
            x: The individuals to calculate the fitness of.
            
        Returns:
            The fitness of the individuals.
        """
        return self.n * np.abs(self._cosine_part(self.a)) + np.sum(
            self._cosine_part(x),
            axis=1
        )

    def _cosine_part(self, x: float) -> float:
        """
        Calculate the cosine part of the function.
        
        Args:
            x: The value to calculate the cosine part of.
            
        Returns:
            The value of the cosine part of the function.
        """
        return 10 * np.cos(2 * np.pi * x) - pow(x, 2)
