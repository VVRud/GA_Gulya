import pygad
import numpy as np
import time
from typing import Optional, List


class HistoryData:
    """
    History data for the genetic algorithm.
    
    Args:
        log_step: The step to log the history.
    """
    def __init__(self, log_step: Optional[int] = 100):
        self.gomogenity_history: List[List[float]] = []
        self.avg_fitness_history: List[float] = []
        self.log_step = log_step
        self.last_time = time.monotonic()
        self.start_time = self.last_time

    @property
    def avg_gomogenity_history(self) -> List[float]:
        """
        The average gomogenity of the population.
        """
        return np.array(self.gomogenity_history).mean(axis=1)
    
    def extend(self, pygad_instance: pygad.GA) -> None:
        """
        Extend the history data from the pygad instance.
        
        Args:
            pygad_instance: The pygad instance.
        """
        if self.log_step and pygad_instance.generations_completed % self.log_step == 0:
            self._log_step(pygad_instance)
        self._extend_history(pygad_instance)

    def _extend_history(self, pygad_instance: pygad.GA) -> None:
        """
        Extend the history data.
        
        Args:
            pygad_instance: The pygad instance.
        """
        self.gomogenity_history.append(self._calculate_gomogenity(pygad_instance))
        self.avg_fitness_history.append(self._calculate_average_fitness(pygad_instance))

    def _calculate_average_fitness(self, pygad_instance: pygad.GA) -> float:
        """
        Calculate the average fitness of the population.
        
        Args:
            pygad_instance: The pygad instance.
        """
        return pygad_instance.last_generation_fitness.mean()

    def _calculate_gomogenity(self, pygad_instance: pygad.GA) -> List[float]:
        """
        Calculate the gomogenity of the population.
        
        Args:
            pygad_instance: The pygad instance.
        """
        gomogenity_by_1 = pygad_instance.population.mean(axis=0)
        gomogenity_by_0 = 1 - gomogenity_by_1
        stacked_gomogenity = np.vstack((gomogenity_by_1, gomogenity_by_0))
        max_gomogenity_by_gene = stacked_gomogenity.max(axis=0)
        return max_gomogenity_by_gene
    
    def _log_step(self, pygad_instance: pygad.GA) -> None:
        """
        Log the step of the population.
        
        Args:
            pygad_instance: The pygad instance.
        """
        curr_time = time.monotonic()
        pygad_instance.logger.info(
            f"Generation: {pygad_instance.generations_completed:05d}, "
            f"Fitness: {self.avg_fitness_history[-1]:.2f}, "
            f"Gomogenity: {self.avg_gomogenity_history[-1]:.2f}, "
            f"Elapsed time total: {curr_time - self.start_time:.2f}, "
            f"Elapsed time per generation: {curr_time - self.last_time:.2f}"
        )
        self.last_time = curr_time

    def plot_gomogenity(self) -> None:
        """
        Plot the gomogenity of the population.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 16))
        plt.plot(self.avg_gomogenity_history)
        plt.title("Gomogenity of the population")
        plt.xlabel("Generation")
        plt.ylabel("Gomogenity")
        plt.show()

    def plot_fitness(self) -> None:
        """
        Plot the fitness of the population.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 16))
        plt.plot(self.avg_fitness_history)
        plt.title("Fitness of the population")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()
