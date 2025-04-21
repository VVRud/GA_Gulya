from pygad import GA
from tqdm import tqdm

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

level = logging.ERROR
logger = logging.getLogger(__name__)
logger.setLevel(level)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(level)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

NUM_RUNS = 100

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
    "rastrigin": RastriginFunction(n=1),
    "deb4": Deb4Function(n=1),
}

ENCODERS_DECODERS = {
    fitness_function_name: {
        "gray": GrayEncoderDecoder(values_range=fitness_function.range, points_after_decimal=2),
        "binary": BinaryEncoderDecoder(values_range=fitness_function.range, points_after_decimal=2),
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}

FITNESS_FUNCTION_CALCULATORS = {
    fitness_function_name: {
        encoder_decoder_name: FitnessFunctionCalculator(fitness_function, encoder_decoder)
        for encoder_decoder_name, encoder_decoder in ENCODERS_DECODERS[fitness_function_name].items()
    }
    for fitness_function_name, fitness_function in FITNESS_FUNCTIONS.items()
}

POPULATIONS = {
    fitness_function_name: {
        population_size: PopulationGenerator(
            n=ENCODERS_DECODERS[fitness_function_name]["binary"].num_bits,
            population_size=population_size,
            num_runs=NUM_RUNS
        )
        for population_size in [100, 200, 300, 400]
    }
    for fitness_function_name in FITNESS_FUNCTIONS
}

with open("simple_params.json", "r") as f:
    PARAMETERS_SIMPLE = json.load(f)


def main(params: dict[str, Any]):
    population_size = params["population"]
    population = POPULATIONS[params["fitness_function"]][population_size]
    fitness_function = FITNESS_FUNCTIONS[params["fitness_function"]]
    fitness_function_calculator = FITNESS_FUNCTION_CALCULATORS[params["fitness_function"]][params["encoding_type"]]
    encoder_decoder = ENCODERS_DECODERS[params["fitness_function"]][params["encoding_type"]]

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

    # Stop criteria and history data
    history_data = HistoryData(log_step=100)
    stop_criteria = StopCriteria(history_data, max_generations=1_000_000)

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
        initial_population=population[0],

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
        on_generation=stop_criteria.stop_condition
    )
    logger.info(ga_instance.summary())
    ga_instance.run()

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_solution_decoded = encoder_decoder.decode(best_solution)
    logger.info(f"Best solution: {best_solution_decoded}, Best fitness: {best_solution_fitness}")
    logger.info(f"Optimal solution: {fitness_function.global_max_x}, Optimal fitness: {fitness_function.global_max_y}")


if __name__ == "__main__":
    # for param in tqdm(PARAMETERS_SIMPLE):
    #     main(param)
    with mp.Pool(processes=os.cpu_count()) as pool:
        r = list(tqdm(pool.imap(main, PARAMETERS_SIMPLE), total=len(PARAMETERS_SIMPLE)))
