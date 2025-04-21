import pygad
from GA.fitness_func import BaseFitnessFunction
from GA.encoding import BaseEncoderDecoder
from typing import List


class FitnessFunctionCalculator:
    """
    Fitness function calculator for a population of individuals.
    
    Args:
        fitness_function: The fitness function to use.
        encoder_decoder: The encoder decoder to use.
    """
    def __init__(self, fitness_function: BaseFitnessFunction, encoder_decoder: BaseEncoderDecoder):
        self.fitness_function = fitness_function
        self.encoder_decoder = encoder_decoder

    def fitness(self, pygad_instance: pygad.GA, solution: List[int] | List[List[int]], solution_idx: int) -> float:
        """
        Calculate the fitness of an individual.
        
        Args:
            pygad_instance: The pygad instance.
            solution: The solution to calculate the fitness of.
            solution_idx: The index of the solution in the population.
        Returns:
            The fitness of the individual.
        """
        if len(solution.shape) == 1:
            decoded = self.encoder_decoder.decode(solution)
            fitness = self.fitness_function.fitness_func(decoded)
        else:
            decoded = self.encoder_decoder.decode_many(solution)
            fitness = self.fitness_function.fitness_func_many(decoded)
        return fitness
