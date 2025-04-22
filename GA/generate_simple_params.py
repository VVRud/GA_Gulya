from collections import defaultdict
import itertools
import json


# Common Parameters
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

# Simple parameters
POPULATION_SIZES = [100, 200, 300, 400]

BASE_PARENT_SELECTION_TYPES = [
    {
        "name": "sus",
        "param": None
    },
    {
        "name": "rws",
        "param": None
    },

    {
        "name": "tour_with",
        "param": 2
    },
    {
        "name": "tour_without",
        "param": 2
    },
    {
        "name": "tour_with_partial",
        "param": 2
    },

    {
        "name": "tour_with",
        "param": 4
    },
    {
        "name": "tour_without",
        "param": 4
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


PARENT_SELECTION_TYPES = defaultdict(list)
for population in POPULATION_SIZES:
    PARENT_SELECTION_TYPES[population].extend(BASE_PARENT_SELECTION_TYPES)
    for rank_type in RANK_TYPES:
        for base_type in BASE_TYPES:
            if rank_type == "exp":
                for exponent in EXP_RANK_EXPONENTS[population]:
                    PARENT_SELECTION_TYPES[population].append({
                        "name": f"{rank_type}_rank_{base_type}",
                        "param": exponent
                    })
            elif rank_type == "lin":
                for exponent in LIN_RANK_EXPONENTS:
                    PARENT_SELECTION_TYPES[population].append({
                        "name": f"{rank_type}_rank_{base_type}",
                        "param": exponent
                    })

BASE_PARAMS = []
for (
    dimension,
    encoding_type,
    crossover_type,
    crossover_probability,
    population
) in itertools.product(
    DIMENSIONS,
    ENCODING_TYPES,
    CROSSOVER_TYPES,
    CROSSOVER_PROBABILITIES,
    POPULATION_SIZES
):
    for (
        mutation_probability,
        parent_selection_type
    ) in itertools.product(
        MUTATION_PROBABILITIES[population],
        PARENT_SELECTION_TYPES[population]
    ):
        BASE_PARAMS.append(
            {
                "population": population,
                
                "dimension": dimension,
                "encoding_type": encoding_type,

                "parent_selection_type": parent_selection_type,

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

FINAL_PARAMS["v1"] = sorted(FINAL_PARAMS["v1"], key=lambda x: (
    x["dimension"],
    x["population"],

    x["fitness_function"],
    x["encoding_type"],

    x["parent_selection_type"]["name"],
    x["crossover_type"],
    x["crossover_probability"],
    x["mutation_probability"])
)

FINAL_PARAMS["v2"] = sorted(FINAL_PARAMS["v2"], key=lambda x: (
    x["dimension"],
    x["population"],

    x["fitness_function"],
    x["encoding_type"],

    x["parent_selection_type"]["name"],
    x["crossover_type"],
    x["crossover_probability"],
    x["mutation_probability"])
)

with open("simple_params.json", "w") as f:
    json.dump(FINAL_PARAMS, f, indent=4)
