import itertools
import json
from collections import defaultdict
from pathlib import Path

PARAMETERS_PATH = Path.cwd() / "parameters"
PARAMETERS_PATH.mkdir(exist_ok=True)

# Common Parameters
NUM_RUNS = 10
MAX_GENERATIONS = 100_000
HISTORY_CHECK_GENERATIONS = 10
DIMENSIONS = [1, 2, 3, 5]

ENCODING_TYPES = ["binary", "gray"]

V1_FITNESS_FUNCTIONS = [
    "rastrigin",
    "deb4",
]

V2_FITNESS_FUNCTIONS = [
    "ackley",
    "deb2",
]

CROSSOVER_TYPES = ["single_point", "uniform"]
CROSSOVER_PROBABILITIES = [0, 0.6, 0.8, 1]

MUTATION_PROBABILITIES = {
    100: [0, 0.001, 0.01, 0.1],
    200: [0, 0.0005, 0.005, 0.01, 0.1],
    300: [0, 0.0003, 0.003, 0.01, 0.1],
    400: [0, 0.0002, 0.0005, 0.002, 0.01, 0.1],
}

# generational parameters
POPULATION_SIZES = [100, 200, 300, 400]

BASE_PARENT_SELECTION_TYPES = [
    {"name": "generational", "parent_selection_type": {"name": "sus", "param": None}},
    {"name": "generational", "parent_selection_type": {"name": "rws", "param": None}},
    {
        "name": "generational",
        "parent_selection_type": {"name": "tour_with", "param": 2},
    },
    {
        "name": "generational",
        "parent_selection_type": {"name": "tour_without", "param": 2},
    },
    {
        "name": "generational",
        "parent_selection_type": {"name": "tour_with_partial", "param": 2},
    },
    {
        "name": "generational",
        "parent_selection_type": {"name": "tour_with", "param": 4},
    },
    {
        "name": "generational",
        "parent_selection_type": {"name": "tour_without", "param": 4},
    },
]

BASE_TYPES = ["rws", "sus"]
RANK_TYPES = ["exp", "lin"]

EXP_RANK_EXPONENTS = {
    100: [0.98010, 0.96060],
    200: [0.99003, 0.98015],
    300: [0.99334, 0.98673],
    400: [0.99503, 0.99004],
}
LIN_RANK_EXPONENTS = [2, 1.6]


SELECTION_TYPES = defaultdict(list)
for population in POPULATION_SIZES:
    SELECTION_TYPES[population].extend(BASE_PARENT_SELECTION_TYPES)
    for rank_type in RANK_TYPES:
        for base_type in BASE_TYPES:
            if rank_type == "exp":
                for exponent in EXP_RANK_EXPONENTS[population]:
                    SELECTION_TYPES[population].append(
                        {
                            "name": "generational",
                            "parent_selection_type": {
                                "name": f"{rank_type}_rank_{base_type}",
                                "param": exponent,
                            },
                        }
                    )
            elif rank_type == "lin":
                for exponent in LIN_RANK_EXPONENTS:
                    SELECTION_TYPES[population].append(
                        {
                            "name": "generational",
                            "parent_selection_type": {
                                "name": f"{rank_type}_rank_{base_type}",
                                "param": exponent,
                            },
                        }
                    )

BASE_PARAMS = []
for (
    dimension,
    encoding_type,
    crossover_type,
    crossover_probability,
    population,
) in itertools.product(
    DIMENSIONS,
    ENCODING_TYPES,
    CROSSOVER_TYPES,
    CROSSOVER_PROBABILITIES,
    POPULATION_SIZES,
):
    for mutation_probability, selection_type in itertools.product(
        MUTATION_PROBABILITIES[population], SELECTION_TYPES[population]
    ):
        BASE_PARAMS.append(
            {
                "num_runs": NUM_RUNS,
                "max_generations": MAX_GENERATIONS,
                "parents_mating": population,
                "history_check_generations": HISTORY_CHECK_GENERATIONS,
                "population": population,
                "dimension": dimension,
                "encoding_type": encoding_type,
                "selection_type": selection_type,
                "crossover_type": crossover_type,
                "crossover_probability": crossover_probability,
                "mutation_probability": mutation_probability,
            }
        )

FINAL_PARAMS = {
    "v1": [
        {**base_param, "fitness_function": fitness_function}
        for base_param in BASE_PARAMS
        for fitness_function in V1_FITNESS_FUNCTIONS
    ],
    "v2": [
        {**base_param, "fitness_function": fitness_function}
        for base_param in BASE_PARAMS
        for fitness_function in V2_FITNESS_FUNCTIONS
    ],
}

for k, v in FINAL_PARAMS.items():
    print(k, len(v))
    FINAL_PARAMS[k] = sorted(
        v,
        key=lambda x: (
            x["population"],
            x["dimension"],
            x["fitness_function"],
            x["encoding_type"],
            x["selection_type"]["parent_selection_type"]["name"],
            x["selection_type"]["parent_selection_type"].get("param", 0),
            x["crossover_type"],
            x["crossover_probability"],
            x["mutation_probability"],
        ),
    )

with open(PARAMETERS_PATH / "generational_params.json", "w") as f:
    json.dump(FINAL_PARAMS, f, indent=4)
