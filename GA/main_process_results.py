import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from xlsxwriter.worksheet import Worksheet

warnings.filterwarnings("ignore")

RESULTS_PATH = Path.cwd() / "results"
JSON_DOCS = RESULTS_PATH / "json"

XLSX_DOCS = RESULTS_PATH / "xlsx"
XLSX_DOCS.mkdir(exist_ok=True)


PARAM_NAMES = {
    "fitness_function": "Fitness Function",
    "encoding_type": "Encoding Type",
    "parents_mating": "Parents Mating",
    "parent_selection_type_name": "Parent Selection Type",
    "parent_selection_type_param": "Parent Selection Type Param",
    "next_generation_selection_type_name": "Next Generation Selection Type",
    "next_generation_selection_type_param": "Next Generation Selection Type Param",
    "crossover_type": "Crossover Type",
    "crossover_probability": "Crossover Probability",
    "mutation_type": "Mutation Type",
    "mutation_probability": "Mutation Probability",
}

RUN_NAMES = {
    "Run": "Run",
    "IsSuc": "Is Suc",
    "NI": "NI",
    "NFE": "NFE",
    "F_max": "F_max",
    "x_max": "x_max",
    "F_avg": "F_avg",
    "FC": "FC",
    "PA": "PA",
    "DA": "DA",
}

SUMMARY_NAMES = (
    ["Suc"]
    + [
        f"{statistic_name}__{metric_name}"
        for metric_name in RUN_NAMES
        for statistic_name in ["Min", "Max", "Avg", "Sigma"]
        if metric_name not in ["Run", "IsSuc", "x_max"]
    ]
    + [
        f"{statistic_name}__{metric_name}__f"
        for metric_name in RUN_NAMES
        for statistic_name in ["Min", "Max", "Avg", "Sigma"]
        if metric_name not in ["Run", "IsSuc", "x_max"]
    ]
)

COLUMN_PADDING = 2


def read_json_files() -> dict[str, dict[str, list[dict]]]:
    data = defaultdict(lambda: defaultdict(list))
    for file in tqdm(
        sorted(JSON_DOCS.glob("*.json")), position=0, desc="Reading JSON files"
    ):
        with open(file) as f:
            result = json.load(f)
            result = unnest_selection_type(result)
            result = unnest_summary(result)
            key = "_".join(
                [
                    result["params"]["fitness_function"],
                    str(result["params"]["dimension"]),
                    str(result["params"]["population"]),
                ]
            )
            subkey = (
                "steady_state"
                if result["params"].get("next_generation_selection_type_name", None)
                is not None
                else "generational"
            )
            data[key][subkey].append(result)
    return data


def unnest_selection_type(result: dict) -> dict:
    selection = result["params"].pop("selection_type")
    result["params"]["mutation_type"] = "uniform"
    result["params"]["parent_selection_type_name"] = selection["parent_selection_type"][
        "name"
    ]
    result["params"]["parent_selection_type_param"] = selection[
        "parent_selection_type"
    ].get("param", None)
    result["params"]["next_generation_selection_type_name"] = selection.get(
        "next_generation_selection_type", {}
    ).get("name", None)
    result["params"]["next_generation_selection_type_param"] = selection.get(
        "next_generation_selection_type", {}
    ).get("param", None)
    return result


def unnest_summary(result: dict) -> dict:
    failed_metrics = result["summary"].pop("Failed")
    success_metrics = result["summary"].pop("Successful")
    for v in success_metrics.values():
        result["summary"].update(v)

    for v in failed_metrics.values():
        result["summary"].update(
            {
                f"{metric_name}__f": metric_value
                for metric_name, metric_value in v.items()
            }
        )

    return result


def process_data(data: dict[str, dict[str, list[dict]]]) -> None:
    for filename, reproductions in tqdm(
        data.items(), position=0, desc="Processing data"
    ):
        with pd.ExcelWriter(
            XLSX_DOCS / f"{filename}.xlsx", engine="xlsxwriter"
        ) as writer:
            stats_df = []
            for reproduction_type, values in tqdm(
                reproductions.items(),
                position=1,
                leave=False,
                desc="Processing reproductions",
            ):
                runs_data = [
                    {**value["params"], **metric}
                    for value in values
                    for metric in value["metrics"]
                ]
                runs_df = pd.DataFrame.from_records(runs_data)
                runs_df = runs_df.reindex(
                    columns=list(PARAM_NAMES.keys()) + list(RUN_NAMES.keys())
                )
                runs_df = runs_df.rename(columns=PARAM_NAMES)
                runs_df = runs_df.rename(columns=RUN_NAMES)

                summary_data = [
                    {**value["params"], **value["summary"]} for value in values
                ]
                summary_df = pd.DataFrame.from_records(summary_data)
                summary_df = summary_df.reindex(
                    columns=list(PARAM_NAMES.keys()) + SUMMARY_NAMES
                )
                summary_df = summary_df.rename(columns=PARAM_NAMES)
                stats_df.append(summary_df)

                write_excel_file(writer, runs_df, reproduction_type)
                write_excel_file(
                    writer,
                    summary_df,
                    f"{reproduction_type}_SUMMARY".upper(),
                )

            stats_df = pd.concat(stats_df)
            write_excel_file(writer, stats_df, "FINAL_SUMMARY".upper())


def write_excel_file(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]
    adjust_column_widths(worksheet, df)
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, df.shape[0], df.shape[1] - 1)


def adjust_column_widths(worksheet: Worksheet, df: pd.DataFrame) -> None:
    for col_idx, col_name in enumerate(df.columns):
        width = (
            max(len(col_name), df[col_name].astype(str).map(len).max()) + COLUMN_PADDING
        )
        worksheet.set_column(col_idx, col_idx, width)


if __name__ == "__main__":
    data = read_json_files()
    process_data(data)
