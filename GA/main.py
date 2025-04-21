from pygad import GA
from tqdm import tqdm
import numpy as np
from GA.selection import (
    RWSSelection,
    LinearRankRWSSelection,
    ExponentialRankRWSSelection,
    SUSSelection,
    LinearRankSUSSelection,
    ExponentialRankSUSSelection,
    TournamentWithReturn,
    TournamentWithoutReturn,
    TournamentWithPartialReturn,
)
from GA.uniform_mutation import UniformMutation
from GA.extensions import FitnessFunctionCalculator, StopCriteria, HistoryData, PopulationGenerator
from GA.fitness_func import Deb4Function, RastriginFunction
from GA.encoding import GrayEncoderDecoder, BinaryEncoderDecoder
import logging
import os
from typing import Any
import json
import multiprocessing as mp
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_JSON_PATH = RESULTS_DIR / "json"
RESULTS_JSON_PATH.mkdir(exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "csv"
RESULTS_CSV_PATH.mkdir(exist_ok=True)

level = logging.ERROR
logger = logging.getLogger(__name__)
logger.setLevel(level)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(level)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

NUM_RUNS = 100
MAX_GENERATIONS = 1_000_000

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

FITNESS_FUNCTIONS = {
    "rastrigin": {dimension: RastriginFunction(n=dimension) for dimension in [1, 2, 3, 5]},
    "deb4": {dimension: Deb4Function(n=dimension) for dimension in [1, 2, 3, 5]},
}

# Use the first dimension for the fitness function. Encoder and decoder will be the same for all dimensions.
ENCODERS_DECODERS = {
    fitness_function_name: {
        "gray": GrayEncoderDecoder(values_range=fitness_function[1].range, points_after_decimal=2),
        "binary": BinaryEncoderDecoder(values_range=fitness_function[1].range, points_after_decimal=2),
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}

FITNESS_FUNCTION_CALCULATORS = {
    fitness_function_name: {
        dimension: {
            encoder_decoder_name: FitnessFunctionCalculator(fitness_function[dimension], encoder_decoder)
            for encoder_decoder_name, encoder_decoder in ENCODERS_DECODERS[fitness_function_name].items()
        }
        for dimension in fitness_function.keys()
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}

POPULATIONS = {
    fitness_function_name: {
        dimension: {
            population_size: PopulationGenerator(
                n=ENCODERS_DECODERS[fitness_function_name]["binary"].num_bits,
                population_size=population_size,
                num_runs=NUM_RUNS
            )
            for population_size in [100, 200, 300, 400]
        }
        for dimension in fitness_function.keys()
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}

with open("simple_params.json", "r") as f:
    PARAMETERS_SIMPLE = json.load(f)


def main(params: dict[str, Any]):
    fitness_function_name = params["fitness_function"]
    dimension = params["dimension"]
    encoding_type = params["encoding_type"]
    population_size = params["population"]

    populations = POPULATIONS[fitness_function_name][dimension][population_size]
    fitness_function = FITNESS_FUNCTIONS[fitness_function_name][dimension]
    fitness_function_calculator = FITNESS_FUNCTION_CALCULATORS[fitness_function_name][dimension][encoding_type]
    encoder_decoder = ENCODERS_DECODERS[fitness_function_name][encoding_type]

    # Operators
    # Mutation
    mutator = UniformMutation(mutation_probability=params["mutation_probability"])
    
    # Parent selection
    selector_name = params["parent_selection_type"]["name"]
    selector_param = params["parent_selection_type"]["param"]
    selector_type = SELECTION_TYPES[selector_name]
    if selector_param is not None:
        selector = selector_type(selector_param)
    else:
        selector = selector_type()

    # Crossover
    crossover_type = params["crossover_type"]
    crossover_probability = params["crossover_probability"]

    metrics = []
    for i in range(NUM_RUNS):
        # Stop criteria and history data
        history_data = HistoryData(log_step=100)
        stop_criteria = StopCriteria(history_data, max_generations=MAX_GENERATIONS)

        ga_instance = GA(
            # Basic parameters
            logger=logger,
            parallel_processing=["thread", int(os.cpu_count())],
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
            # TODO(Vlad): Will be different for GG algorithms
            num_parents_mating=population_size,

            # Crossover parameters
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,

            # Mutation parameters
            mutation_type=mutator.mutate,

            fitness_func=fitness_function_calculator.fitness,
            fitness_batch_size=population_size,
            on_generation=stop_criteria.stop_condition
        )
        logger.info(ga_instance.summary())
        ga_instance.run()

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        best_solution_decoded = encoder_decoder.decode(best_solution)
        
        generations = ga_instance.generations_completed
        x_distance = np.linalg.norm(best_solution_decoded - fitness_function.global_max_x)
        y_distance = abs(best_solution_fitness - fitness_function.global_max_y)
        is_successful = stop_criteria.is_converged and (x_distance < 0.01) and (y_distance < 0.01)
        metrics.append({
            "Run": i + 1,
            "IsSuc": bool(is_successful),
            "NI": generations,
            # TODO(Vlad): Add NFE
            "NFE": 0,
            "F_max": float(best_solution_fitness),
            "x_max": [float(x) for x in best_solution_decoded],
            "F_avg": float(history_data.avg_fitness_history[-1]),
            "FC": float(best_solution_fitness - history_data.avg_fitness_history[-1]),
            "PA": float(y_distance),
            "DA": float(x_distance),
        })

    final_metrics = {
        "params": params,
        "metrics": metrics,
        "summary": calculate_summary_metrics(metrics),
    }
    filename = "__".join([
        str(params["population"]),
        params["fitness_function"],
        str(params["dimension"]),
        params["encoding_type"],
        params["parent_selection_type"]["name"],
        params["crossover_type"],
        str(params["crossover_probability"]),
        str(params["mutation_probability"]),
    ]) + ".json"
    with open(RESULTS_JSON_PATH / filename, "w") as f:
        json.dump(final_metrics, f, indent=4)

    return final_metrics


def calculate_summary_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    successful_runs = [m for m in metrics if m["IsSuc"]]
    failed_runs = [m for m in metrics if not m["IsSuc"]]
    statistics_metrics = {
        "Suc": len(successful_runs) / len(metrics),
        "Failed": {},
        "Successful": {},
    }
    for key in ["NI", "NFE", "F_max", "F_avg", "FC", "PA", "DA"]:
        statistics_metrics["Failed"][key] = calculate_statistics(failed_runs, key)
        statistics_metrics["Successful"][key] = calculate_statistics(successful_runs, key)
    return statistics_metrics


def calculate_statistics(metrics: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics_values = [m[key] for m in metrics]
    if len(metrics_values) == 0:
        return {
            f"Min__{key}": None,
            f"Max__{key}": None,
            f"Avg__{key}": None,
            f"Sigma__{key}": None,
        }
    return {
        f"Min__{key}": float(np.min(metrics_values)),
        f"Max__{key}": float(np.max(metrics_values)),
        f"Avg__{key}": float(np.mean(metrics_values)),
        f"Sigma__{key}": float(np.std(metrics_values)),
    }


if __name__ == "__main__":
    for param in tqdm(PARAMETERS_SIMPLE):
        main(param)
    # with mp.Pool(processes=os.cpu_count()) as pool:
    #     r = list(tqdm(pool.imap(main, PARAMETERS_SIMPLE), total=len(PARAMETERS_SIMPLE)))
