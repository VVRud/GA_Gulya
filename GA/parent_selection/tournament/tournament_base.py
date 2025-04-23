import numpy as np
import pygad

from GA.parent_selection.base_selection import BaseSelection


class TournamentBase(BaseSelection):
    """Base class for tournament selection methods."""

    def __init__(self, tournament_size: int):
        self.tournament_size = tournament_size

    def get_tournament_winner(
        self, tournament_indices: np.ndarray, tournament_fitness: np.ndarray
    ) -> np.ndarray:
        """
        Get indices of tournament winners.
        Args:
            tournament_indices: Indices of the tournament.
            tournament_fitness: Fitness values of the tournament.
        Returns:
            Indices of the tournament winners.
        """
        tournament_winner_indices = np.argmax(tournament_fitness, axis=1)
        return tournament_indices[
            np.arange(len(tournament_winner_indices)), tournament_winner_indices
        ]

    def get_parents(
        self, parents_indices: np.ndarray, ga_instance: pygad.GA
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get parent chromosomes from indices.
        Args:
            parents_indices: Indices of the parents.
            ga_instance: Instance of the GA class.
        Returns:
            Parents chromosomes and their indices.
        """
        parents = ga_instance.population[parents_indices]
        return parents, parents_indices
