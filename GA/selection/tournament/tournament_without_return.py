from .tournament_base import TournamentBase
import numpy as np
import pygad
from typing import Tuple


class TournamentWithoutReturn(TournamentBase):
    """Tournament selection without replacement between tournaments."""

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
        num_parents_in_tournament = num_parents_mating // self.tournament_size
        num_copies = int(np.ceil(num_parents_mating / num_parents_in_tournament))
        fitness_indices = np.arange(last_generation_fitness.shape[0])
        
        parents_indices = np.empty((num_copies * num_parents_in_tournament,), dtype=np.int64)
        for i in range(num_copies):
            tournament_indices = np.random.choice(
                fitness_indices,
                size=(num_parents_in_tournament, self.tournament_size),
                replace=False,
            )
            tournament_fitness = last_generation_fitness[tournament_indices]
            parents_indices[i * num_parents_in_tournament:(i + 1) * num_parents_in_tournament] = self.get_tournament_winner(
                tournament_indices,
                tournament_fitness
            )

        return self.get_parents(parents_indices[:num_parents_mating], ga_instance)

