import numpy as np
import pygad

from .tournament_base import TournamentBase


class TournamentWithPartialReturn(TournamentBase):
    """Tournament selection with partial replacement between tournaments."""

    def select(
        self,
        last_generation_fitness: np.ndarray[np.float64],
        num_parents_mating: int,
        ga_instance: pygad.GA,
    ) -> tuple[np.ndarray[np.int64, np.int64], np.ndarray[np.int64]]:
        """
        Select parents from the tournament.
        Args:
            last_generation_fitness: Fitness values of the last generation.
            num_parents_mating: Number of parents to select.
            ga_instance: Instance of the GA class.
        Returns:
            Parents chromosomes and their indices.
        """
        fitness_indices = np.repeat(
            np.arange(last_generation_fitness.shape[0]), self.tournament_size
        )
        tournaments_indices = np.random.choice(
            fitness_indices,
            size=(num_parents_mating, self.tournament_size),
            replace=False,
        )
        tournament_fitness = last_generation_fitness[tournaments_indices]
        parents_indices = self.get_tournament_winner(
            tournaments_indices, tournament_fitness
        )

        return self.get_parents(parents_indices, ga_instance)
