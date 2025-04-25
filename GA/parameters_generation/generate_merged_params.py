import json
from pathlib import Path

from GA.parameters_generation.generate_generational_params import (
    FINAL_PARAMS as GENERATIONAL_PARAMS,
)
from GA.parameters_generation.generate_steady_params import (
    FINAL_PARAMS as STEADY_PARAMS,
)

PARAMETERS_PATH = Path.cwd() / "parameters"
PARAMETERS_PATH.mkdir(exist_ok=True)

MERGED_PARAMS = {
    "v1": STEADY_PARAMS["v1"] + GENERATIONAL_PARAMS["v1"],
    "v2": STEADY_PARAMS["v2"] + GENERATIONAL_PARAMS["v2"],
}

for k, v in MERGED_PARAMS.items():
    print(k, len(v))
    MERGED_PARAMS[k] = sorted(
        v,
        key=lambda x: (
            x["population"],
            x["dimension"],
            x["fitness_function"],
            x["encoding_type"],
            x["selection_type"]["name"],
            x["selection_type"]["parent_selection_type"]["name"],
            x["selection_type"]["parent_selection_type"].get("param", 0),
            x["selection_type"]
            .get("next_generation_selection_type", {})
            .get("name", ""),
            x["selection_type"]
            .get("next_generation_selection_type", {})
            .get("param", 0),
            x["crossover_type"],
            x["crossover_probability"],
            x["mutation_probability"],
        ),
    )


with open(PARAMETERS_PATH / "merged_params.json", "w") as f:
    json.dump(MERGED_PARAMS, f)
