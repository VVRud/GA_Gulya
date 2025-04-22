import itertools
import json


# Common parameters
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

# GG parameters
POPULATION_SIZES = [100, 200, 300, 400]

GENERATION_GAP = [0.05, 0.1, 0.2, 0.5]

PARENT_SELECTION_TYPES = ["elite", "rws"]

NEXT_GENERATION_SELECTION_TYPES = ["worst_comma", "rand_comma", "worst_plus", "rand_plus"]

BASE_PARAMS = []
for (
    dimension,
    encoding_type,
    crossover_type,
    crossover_probability,
    generation_gap,
    parent_selection_type,
    next_generation_selection_type,
    population
) in itertools.product(
    DIMENSIONS,
    ENCODING_TYPES,
    CROSSOVER_TYPES,
    CROSSOVER_PROBABILITIES,
    GENERATION_GAP,
    PARENT_SELECTION_TYPES,
    NEXT_GENERATION_SELECTION_TYPES,
    POPULATION_SIZES
):
    for mutation_probability in MUTATION_PROBABILITIES[population]:
        BASE_PARAMS.append({
            "population": population,

            "dimension": dimension,
            "encoding_type": encoding_type,

            "generation_gap": generation_gap,
            "parent_selection_type": parent_selection_type,
            "next_generation_selection_type": next_generation_selection_type,

            "crossover_type": crossover_type,
            "crossover_probability": crossover_probability,

            "mutation_probability": mutation_probability,
        })

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

FINAL_PARAMS["v1"] = sorted(FINAL_PARAMS["v1"], key=lambda x: (
    x["population"],
    x["dimension"],

    x["fitness_function"],
    x["encoding_type"],

    x["generation_gap"],
    x["parent_selection_type"],
    x["next_generation_selection_type"],

    x["crossover_type"],
    x["crossover_probability"],
    x["mutation_probability"],
))

FINAL_PARAMS["v2"] = sorted(FINAL_PARAMS["v2"], key=lambda x: (
    x["population"],
    x["dimension"],

    x["fitness_function"],
    x["encoding_type"],

    x["generation_gap"],
    x["parent_selection_type"],
    x["next_generation_selection_type"],

    x["crossover_type"],
    x["crossover_probability"],
    x["mutation_probability"],
))

with open("gg_params.json", "w") as f:
    json.dump(FINAL_PARAMS, f, indent=4)
