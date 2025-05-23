import numpy as np
import pygad

from GA.encoding import BaseEncoderDecoder
from GA.fitness_func.base import BaseFitnessFunction


class FitnessFunctionCalculator:
    """
    Fitness function calculator for a population of individuals.

    Args:
        fitness_function: The fitness function to use.
        encoder_decoder: The encoder decoder to use.
    """

    def __init__(
        self, fitness_function: BaseFitnessFunction, encoder_decoder: BaseEncoderDecoder
    ):
        self.fitness_function = fitness_function
        self.encoder_decoder = encoder_decoder

        self.number_evaluations = 0

    def fitness(
        self,
        pygad_instance: pygad.GA,
        solution: np.ndarray | list[int],
        solution_idx: int,
    ) -> float | list[float]:
        """
        Calculate the fitness of an individual.

        Args:
            pygad_instance: The pygad instance.
            solution: The solution to calculate the fitness of.
            solution_idx: The index of the solution in the population.
        Returns:
            The fitness of the individual/individuals.
        """
        # Detect if we're processing a single solution or batch
        if len(solution.shape) == 1:
            decoded = self.encoder_decoder.decode(solution)
            fitness_value = self.fitness_function.fitness_func(decoded)
            self.number_evaluations += 1
            return fitness_value
        else:
            decoded = self.encoder_decoder.decode_many(solution)
            fitness_values = self.fitness_function.fitness_func_many(decoded)
            self.number_evaluations += solution.shape[0]
            return fitness_values

    def reset(self) -> None:
        """
        Reset the number of evaluations.
        """
        self.number_evaluations = 0
