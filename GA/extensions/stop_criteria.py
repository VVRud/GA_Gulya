import pygad
import numpy as np
from typing import Optional
from .history_data import HistoryData


class StopCriteria:
    STOP_WORD = "stop"
    
    def __init__(
        self,
        history_data: HistoryData,
        max_generations: int,
        gomogenity_threshold: float = 0.99,
        fitness_change_history_length: int = 10,
        fitness_change_threshold: float = 0.0001
    ):
        self.history_data = history_data
        self.max_generations = max_generations
        self.gomogenity_threshold = gomogenity_threshold
        self.fitness_change_history_length = fitness_change_history_length
        self.fitness_change_threshold = fitness_change_threshold

        self.is_converged = False

    def is_gomogenic(self, population_number: int) -> bool:
        """
        Check gomogenity of the population by each gene.
        
        Args:
            population_number: The number of the population to check.
        Returns:
            If the population is gomogenic.
        """
        return np.all(self.history_data.gomogenity_history[population_number] > self.gomogenity_threshold)

    def fitness_change(self, population_number: int) -> Optional[float]:
        """
        Check fitness change of the population.
        
        Args:
            population_number: The number of the population to check.
        Returns:
            The fitness change of the population.
        """
        if len(self.history_data.avg_fitness_history) > self.fitness_change_history_length:
            return abs(
                self.history_data.avg_fitness_history[population_number]
                - self.history_data.avg_fitness_history[population_number - self.fitness_change_history_length]
            )
        return None

    def stop_condition(self, pygad_instance: pygad.GA) -> Optional[str]:
        """
        Check if the stop condition is met.
        
        Args:
            pygad_instance: The pygad instance.
        Returns:
            The stop condition.
        """
        generation = pygad_instance.generations_completed - 1
        self.history_data.extend(pygad_instance)
        
        if generation > self.max_generations:
            pygad_instance.logger.info(f"Hard stop: max generations reached ({generation} > {self.max_generations})")
            return self.STOP_WORD

        if self.is_gomogenic(generation):
            pygad_instance.logger.info(f"Convergence condition met: gomogenity reached ({self.is_gomogenic(generation)})")
            self.is_converged = True
            return self.STOP_WORD

        fitness_change = self.fitness_change(generation)
        if fitness_change is not None and fitness_change < self.fitness_change_threshold:
            pygad_instance.logger.info(f"Convergence condition met: fitness change reached ({fitness_change} < {self.fitness_change_threshold})")
            self.is_converged = True
            return self.STOP_WORD

        return None
