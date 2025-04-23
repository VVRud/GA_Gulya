from collections.abc import Callable

import pygad


class GeneticAlgorithm(pygad.GA):
    def __init__(
        self,
        with_steady_state: bool = False,
        steady_state_selection_type: Callable | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.with_steady_state = with_steady_state
        self.steady_state_selection_type = steady_state_selection_type
        self.num_offspring = (
            self.num_parents_mating if self.with_steady_state else self.num_offspring
        )

    def run_update_population(self, *args, **kwargs):
        if not self.with_steady_state:
            return super().run_update_population(*args, **kwargs)

        self.population = self.steady_state_selection_type(
            self.population,
            self.last_generation_fitness,
            self.last_generation_offspring_mutation,
            self,
        )
