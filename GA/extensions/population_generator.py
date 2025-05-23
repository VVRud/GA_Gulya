from pathlib import Path

import numpy as np


class PopulationGenerator:
    def __init__(
        self, n: int, population_size: int, num_runs: int, populations_path: Path
    ):
        self.n = n
        self.population_size = population_size
        self.num_runs = num_runs
        self.populations = None
        self.populations_path = (
            populations_path / f"{n}_{population_size}_{num_runs}.npy"
        )
        # Initialize immediately
        self.initialize()

    def initialize(self):
        """
        Initialize populations using optimized random generation.
        """
        if self.populations_path.exists():
            self.populations = np.load(self.populations_path)
            return

        if self.populations is not None:
            raise ValueError("Population already initialized")

        # Use efficient random generation for binary values
        # For binary values (0,1), random integers are faster than random uniform
        # Pre-allocate the entire array at once to avoid multiple allocations
        self.populations = np.random.randint(
            0,
            2,
            size=(self.num_runs, self.population_size, self.n),
            dtype=np.int8,  # Use smallest integer type to save memory
        )

        np.save(self.populations_path, self.populations)

    def __getitem__(self, run_number: int) -> np.ndarray:
        """
        Get the population for a specific run.

        Args:
            run_number: The run number.
        Returns:
            The population for the run.
        """
        if self.populations is None:
            raise ValueError("Population not initialized")

        return self.populations[run_number]
