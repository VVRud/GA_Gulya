import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
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
from GA.genetic_algorithm import GeneticAlgorithm
from GA.mutation.uniform_mutation import UniformMutation
from GA.parent_selection import (
    EliteSelection,
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
from GA.steady_state_selection import (
    RandomCommaSelection,
    RandomPlusSelection,
    WorstCommaSelection,
    WorstPlusSelection,
)

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

CWD = Path.cwd()

PARAMETERS_PATH = CWD / "parameters"

POPULATIONS_DIR = CWD / "populations"
POPULATIONS_DIR.mkdir(exist_ok=True)

# Pre-create directories
RESULTS_DIR = CWD / "results_100"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_JSON_PATH = RESULTS_DIR / "json"
RESULTS_JSON_PATH.mkdir(exist_ok=True)

# Set this to 1 to run the GA with a single run and generation
OVERRIDE_NUM_RUNS = None
OVERRIDE_NUM_GENERATIONS = None

# Constants
MAX_NUM_RUNS = 100
CPU_COUNT = os.cpu_count()

# Cache selection types to avoid repeated lookups
PARENT_SELECTION_TYPES = {
    "rws": RWSSelection,
    "sus": SUSSelection,
    "lin_rank_rws": LinearRankRWSSelection,
    "exp_rank_rws": ExponentialRankRWSSelection,
    "lin_rank_sus": LinearRankSUSSelection,
    "exp_rank_sus": ExponentialRankSUSSelection,
    "tour_with": TournamentWithReturn,
    "tour_without": TournamentWithoutReturn,
    "tour_with_partial": TournamentWithPartialReturn,
    "elite": EliteSelection,
}

STEADY_STATE_SELECTION_TYPES = {
    "worst_comma": WorstCommaSelection,
    "rand_comma": RandomCommaSelection,
    "worst_plus": WorstPlusSelection,
    "rand_plus": RandomPlusSelection,
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
                num_runs=MAX_NUM_RUNS,
                populations_path=POPULATIONS_DIR,
            )
            for population_size in POPULATION_SIZES
        }
        for dimension in fitness_function
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}


def get_parent_selector(parent_selection_params):
    selector_param = parent_selection_params.get("param", None)
    selector_type = PARENT_SELECTION_TYPES[parent_selection_params["name"]]
    if selector_param is not None:
        return selector_type(selector_param)
    return selector_type()


def get_steady_state_selector(next_population_selection_params):
    if (
        next_population_selection_params is None
        or next_population_selection_params == {}
    ):
        return False, None
    selector_param = next_population_selection_params.get("param", None)
    selector_type = STEADY_STATE_SELECTION_TYPES[
        next_population_selection_params["name"]
    ]
    if selector_param is not None:
        return True, selector_type(selector_param)
    return True, selector_type()


def main(params: dict[str, Any]):
    # Extract parameters once
    num_runs = OVERRIDE_NUM_RUNS or params["num_runs"]
    if num_runs > MAX_NUM_RUNS:
        raise ValueError(f"num_runs must be less than or equal to {MAX_NUM_RUNS}")
    num_generations = OVERRIDE_NUM_GENERATIONS or 10_000_000

    max_generations = params["max_generations"]
    history_check_generations = params["history_check_generations"]
    parents_mating = params["parents_mating"]

    population_size = params["population"]
    fitness_function_name = params["fitness_function"]
    dimension = params["dimension"]
    encoding_type = params["encoding_type"]
    parent_selection_params = params["selection_type"]["parent_selection_type"]
    next_population_selection_params = params["selection_type"].get(
        "next_generation_selection_type", {}
    )
    crossover_type = params["crossover_type"]
    crossover_probability = params["crossover_probability"]
    mutation_probability = params["mutation_probability"]

    results_file = RESULTS_JSON_PATH / (
        "__".join(
            [
                str(population_size),
                fitness_function_name,
                str(dimension),
                encoding_type,
                params["selection_type"]["name"],
                parent_selection_params["name"],
                str(parents_mating),
                str(parent_selection_params["param"]),
                next_population_selection_params.get("name", "None"),
                str(next_population_selection_params.get("param", "None")),
                crossover_type,
                str(crossover_probability),
                str(mutation_probability),
            ]
        )
        + ".json"
    )
    if results_file.exists():
        with results_file.open("r") as f:
            return results_file, json.load(f)

    # Get pre-computed objects
    populations = POPULATIONS[fitness_function_name][dimension][population_size]
    fitness_function = FITNESS_FUNCTIONS[fitness_function_name][dimension]
    encoder_decoder = ENCODERS_DECODERS[fitness_function_name][encoding_type]
    fitness_function_calculator = FitnessFunctionCalculator(
        fitness_function, encoder_decoder
    )

    # Initialize operators
    mutator = UniformMutation(mutation_probability=mutation_probability)
    parent_selector = get_parent_selector(parent_selection_params)
    with_steady_state, steady_state_selector = get_steady_state_selector(
        next_population_selection_params
    )

    # Pre-allocate metrics list with expected size
    metrics = []
    for i in range(num_runs):
        # Initialize per-run objects
        history_data = HistoryData(log_step=100)
        stop_criteria = StopCriteria(
            history_data,
            max_generations=max_generations,
            fitness_change_history_length=history_check_generations,
        )

        ga_instance = GeneticAlgorithm(
            # Basic parameters
            logger=logger,
            parallel_processing=["thread", CPU_COUNT],
            num_generations=num_generations,
            keep_parents=0,
            keep_elitism=0,
            save_best_solutions=True,
            suppress_warnings=True,
            # Gene parameters
            gene_type=int,
            gene_space=[0, 1],
            initial_population=populations[i],
            # Parent selection parameters
            parent_selection_type=parent_selector.select,
            num_parents_mating=parents_mating,
            with_steady_state=with_steady_state,
            steady_state_selection_type=(
                steady_state_selector.select if with_steady_state else None
            ),
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

    # Write results in a single operation
    with results_file.open("w") as f:
        json.dump(final_metrics, f, indent=4)

    return results_file, final_metrics


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
    params_type = input("Enter the params type (generational/steady/merged): ")
    parameters_file = PARAMETERS_PATH / f"{params_type}_params.json"
    if not parameters_file.exists():
        raise FileNotFoundError(
            f"Parameters file {parameters_file} not found. "
            "Please generate one or fix the path."
        )
    with open(parameters_file) as f:
        parameters_simple = json.load(f)

    params_key = input("Enter the option number (v1/v2): ")
    params = parameters_simple[params_key]

    if CPU_COUNT == 1:
        r = [main(param) for param in tqdm(params)]
    else:
        with mp.Pool(processes=CPU_COUNT) as pool:
            r = list(tqdm(pool.imap(main, params), total=len(params)))

    filenames = [values[0] for values in r]
    print("Results:")
    print(f"  Total: {len(filenames)}")
    print(f"  Unique: {len(set(filenames))}")
    print(f"  Duplicates: {len(filenames) - len(set(filenames))}")
