from .tournament_base import TournamentBase
import numpy as np
import pygad
from typing import Tuple


class TournamentWithReturn(TournamentBase):
    """Tournament selection with replacement between tournaments."""
    
    def select(self, last_generation_fitness: np.ndarray[np.float64], num_parents_mating: int, ga_instance: pygad.GA) -> Tuple[np.ndarray[np.int64, np.int64], np.ndarray[np.int64]]:
        """
        Select parents from the tournament.
        Args:
            last_generation_fitness: Fitness values of the last generation.
            num_parents_mating: Number of parents to select.
            ga_instance: Instance of the GA class.
        Returns:
            Parents chromosomes and their indices.
        """
        fitness_indices = np.arange(last_generation_fitness.shape[0])
        parents_indices = np.empty((num_parents_mating,), dtype=np.int64)
        
        for i in range(num_parents_mating):
            tournament_indices = np.random.choice(
                fitness_indices,
                size=self.tournament_size,
                replace=False,
            )
            tournament_fitness = last_generation_fitness[tournament_indices]
            parents_indices[i] = self.get_tournament_winner(
                tournament_indices.reshape(1,-1), 
                tournament_fitness.reshape(1,-1)
            )

        return self.get_parents(parents_indices, ga_instance)

