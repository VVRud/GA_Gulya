import functools
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
from pygad import GA
from tqdm import tqdm

from GA.encoding import BinaryEncoderDecoder, GrayEncoderDecoder
from GA.extensions import (
    FitnessFunctionCalculator,
    HistoryData,
    PopulationGenerator,
    StopCriteria,
)
from GA.fitness_func import (
    AckleyFunction,
    Deb2Function,
    Deb4Function,
    RastriginFunction,
)
from GA.selection import (
    ExponentialRankRWSSelection,
    ExponentialRankSUSSelection,
    LinearRankRWSSelection,
    LinearRankSUSSelection,
    RWSSelection,
    SUSSelection,
    TournamentWithoutReturn,
    TournamentWithPartialReturn,
    TournamentWithReturn,
)
from GA.uniform_mutation import UniformMutation

# Optimize logging - set it up once at startup and disable debug-level logging
level = logging.ERROR
logger = logging.getLogger(__name__)
logger.setLevel(level)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(level)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

POPULATIONS_DIR = Path("populations")
POPULATIONS_DIR.mkdir(exist_ok=True)

# Pre-create directories
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_JSON_PATH = RESULTS_DIR / "json"
RESULTS_JSON_PATH.mkdir(exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "csv"
RESULTS_CSV_PATH.mkdir(exist_ok=True)

# Constants
NUM_RUNS = 100
MAX_GENERATIONS = 1_000_000
CPU_COUNT = os.cpu_count()

# Cache selection types to avoid repeated lookups
SELECTION_TYPES = {
    "rws": RWSSelection,
    "sus": SUSSelection,
    "lin_rank_rws": LinearRankRWSSelection,
    "exp_rank_rws": ExponentialRankRWSSelection,
    "lin_rank_sus": LinearRankSUSSelection,
    "exp_rank_sus": ExponentialRankSUSSelection,
    "tour_with": TournamentWithReturn,
    "tour_without": TournamentWithoutReturn,
    "tour_with_partial": TournamentWithPartialReturn,
}

# Pre-calculate fitness functions for all dimensions
DIMENSIONS = [1, 2, 3, 5]
FITNESS_FUNCTIONS = {
    "rastrigin": {
        dimension: RastriginFunction(n=dimension) for dimension in DIMENSIONS
    },
    "deb4": {dimension: Deb4Function(n=dimension) for dimension in DIMENSIONS},
    "ackley": {dimension: AckleyFunction(n=dimension) for dimension in DIMENSIONS},
    "deb2": {dimension: Deb2Function(n=dimension) for dimension in DIMENSIONS},
}

# Pre-calculate encoders/decoders
ENCODERS_DECODERS = {
    fitness_function_name: {
        "gray": GrayEncoderDecoder(
            values_range=fitness_function[1].range, points_after_decimal=2
        ),
        "binary": BinaryEncoderDecoder(
            values_range=fitness_function[1].range, points_after_decimal=2
        ),
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}

# Pre-generate populations
POPULATION_SIZES = [100, 200, 300, 400]
POPULATIONS = {
    fitness_function_name: {
        dimension: {
            population_size: PopulationGenerator(
                n=ENCODERS_DECODERS[fitness_function_name]["binary"].num_bits
                * dimension,
                population_size=population_size,
                num_runs=NUM_RUNS,
                populations_path=POPULATIONS_DIR
            )
            for population_size in POPULATION_SIZES
        }
        for dimension in fitness_function
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}


@functools.lru_cache(maxsize=128)
def get_selector(selector_name, selector_param):
    """Cache selector instances for reuse"""
    selector_type = SELECTION_TYPES[selector_name]
    if selector_param is not None:
        return selector_type(selector_param)
    return selector_type()


def main(params: dict[str, Any]):
    # Extract parameters once
    fitness_function_name = params["fitness_function"]
    dimension = params["dimension"]
    encoding_type = params["encoding_type"]
    population_size = params["population"]
    mutation_probability = params["mutation_probability"]
    crossover_type = params["crossover_type"]
    crossover_probability = params["crossover_probability"]
    selector_name = params["parent_selection_type"]["name"]
    selector_param = params["parent_selection_type"]["param"]

    # Get pre-computed objects
    populations = POPULATIONS[fitness_function_name][dimension][population_size]
    fitness_function = FITNESS_FUNCTIONS[fitness_function_name][dimension]
    encoder_decoder = ENCODERS_DECODERS[fitness_function_name][encoding_type]
    fitness_function_calculator = FitnessFunctionCalculator(
        fitness_function, encoder_decoder
    )

    # Initialize operators
    mutator = UniformMutation(mutation_probability=mutation_probability)
    selector = get_selector(selector_name, selector_param)

    # Pre-allocate metrics list with expected size
    metrics = []
    for i in range(NUM_RUNS):
        # Initialize per-run objects
        history_data = HistoryData(log_step=100)
        stop_criteria = StopCriteria(history_data, max_generations=MAX_GENERATIONS)

        ga_instance = GA(
            # Basic parameters
            logger=logger,
            parallel_processing=["thread", CPU_COUNT],
            num_generations=10_000_000,
            keep_parents=0,
            keep_elitism=0,
            save_best_solutions=True,
            suppress_warnings=True,
            # Gene parameters
            gene_type=int,
            gene_space=[0, 1],
            initial_population=populations[i],
            # Parent selection parameters
            parent_selection_type=selector.select,
            num_parents_mating=population_size,
            # Crossover parameters
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            # Mutation parameters
            mutation_type=mutator.mutate,
            fitness_func=fitness_function_calculator.fitness,
            fitness_batch_size=population_size,
            on_generation=stop_criteria.stop_condition,
        )

        # Skip unnecessary logging during run
        if logger.level <= logging.INFO:
            logger.info(ga_instance.summary())

        ga_instance.run()

        # Get results efficiently
        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        best_solution_decoded = encoder_decoder.decode(best_solution)

        # Calculate metrics
        generations = ga_instance.generations_completed
        x_distance = np.linalg.norm(
            best_solution_decoded - fitness_function.global_max_x
        )
        y_distance = abs(best_solution_fitness - fitness_function.global_max_y)
        is_successful = (
            stop_criteria.is_converged and (x_distance < 0.01) and (y_distance < 0.01)
        )

        # Use float() for NumPy values to avoid serialization issues
        metrics.append(
            {
                "Run": i + 1,
                "IsSuc": bool(is_successful),
                "NI": generations,
                "NFE": fitness_function_calculator.number_evaluations,
                "F_max": float(best_solution_fitness),
                "x_max": [float(x) for x in best_solution_decoded],
                "F_avg": float(history_data.avg_fitness_history[-1]),
                "FC": float(
                    best_solution_fitness - history_data.avg_fitness_history[-1]
                ),
                "PA": float(y_distance),
                "DA": float(x_distance),
            }
        )
        fitness_function_calculator.reset()

    # Calculate summary once at the end
    summary = calculate_summary_metrics(metrics)

    final_metrics = {
        "params": params,
        "metrics": metrics,
        "summary": summary,
    }

    # Create filename once
    filename = (
        "__".join(
            [
                str(params["population"]),
                params["fitness_function"],
                str(params["dimension"]),
                params["encoding_type"],
                params["parent_selection_type"]["name"],
                params["crossover_type"],
                str(params["crossover_probability"]),
                str(params["mutation_probability"]),
            ]
        )
        + ".json"
    )

    # Write results in a single operation
    with open(RESULTS_JSON_PATH / filename, "w") as f:
        json.dump(final_metrics, f, indent=4)

    return final_metrics


def calculate_summary_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate summary statistics from metrics"""

    successful_runs = [m for m in metrics if m["IsSuc"]]
    failed_runs = [m for m in metrics if not m["IsSuc"]]

    statistics_metrics = {
        "Suc": float(len(successful_runs)) / len(metrics),
        "Failed": {},
        "Successful": {},
    }

    # Calculate statistics for both groups at once
    for key in ["NI", "NFE", "F_max", "F_avg", "FC", "PA", "DA"]:
        statistics_metrics["Failed"][key] = calculate_statistics(failed_runs, key)
        statistics_metrics["Successful"][key] = calculate_statistics(
            successful_runs, key
        )

    return statistics_metrics


def calculate_statistics(metrics: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Calculate statistics for a specific metric key"""
    if not metrics:
        return {
            f"Min__{key}": None,
            f"Max__{key}": None,
            f"Avg__{key}": None,
            f"Sigma__{key}": None,
        }

    # Use numpy for efficient statistical calculations
    metrics_values = np.array([m[key] for m in metrics])
    return {
        f"Min__{key}": float(np.min(metrics_values)),
        f"Max__{key}": float(np.max(metrics_values)),
        f"Avg__{key}": float(np.mean(metrics_values)),
        f"Sigma__{key}": float(np.std(metrics_values)),
    }


if __name__ == "__main__":
    # Load parameters once
    with open("simple_params.json") as f:
        parameters_simple = json.load(f)

    params_key = input("Enter the key: ")
    params = parameters_simple[params_key]

    # Run all parameter combinations in parallel using a process pool
    with mp.Pool(processes=CPU_COUNT) as pool:
        r = list(tqdm(pool.imap(main, params), total=len(params)))
