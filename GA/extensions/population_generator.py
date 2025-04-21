import numpy as np
from typing import List


class PopulationGenerator:
    def __init__(self, n: int, population_size: int, num_runs: int):
        self.n = n
        self.population_size = population_size
        self.num_runs = num_runs
        self.populations = None
        self.initialize()

    def initialize(self):
        if self.populations is not None:
            raise ValueError("Population already initialized")

        self.populations = np.random.randint(0, 2, (self.num_runs, self.population_size, self.n))
        
    def __getitem__(self, run_number: int) -> List[int]:
        if self.populations is None:
            raise ValueError("Population not initialized")

        return self.populations[run_number]
