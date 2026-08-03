# 1. Packages
from pathlib import Path

import pandas as pd


# 2. Paths
# Run from the repository root:
# ad-neuron-intervention/

RESULTS_DIR = Path("results")

if not RESULTS_DIR.exists():
    raise FileNotFoundError(
        "Cannot find Chat_analysis/results. "
        "Please run the script from the repository root "
        "or from the Results folder."
    )

OUTPUT_CSV = "all_task_performance_main.csv"
OUTPUT_XLSX = "all_task_performance_main.xlsx"


# 3. Conditions
SELECTED_CONDITIONS = [
    "Original",
    "0.6Alpha_10000Neurons",
    "0.6Alpha_AllSigNeurons",
]


# 4. Read and filter one result file
def read_result(path):
    """Read one result file and retain the selected conditions."""

    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    data["ID"] = data["ID"].astype(str)

    data = data[
        data["Condition"].isin(SELECTED_CONDITIONS)
    ].copy()

    return data


# 5. Verbal fluency
animal = read_result(
    RESULTS_DIR
    / "verbal_fluency"
    / "animal_fluency_with_meta.csv"
)

animal = animal[
    [
        "ID",
        "Condition",
        "Role",
        "total_num",
    ]
].rename(
    columns={
        "total_num": "Animal",
    }
)


letter = read_result(
    RESULTS_DIR
    / "verbal_fluency"
    / "letter_fluency_with_meta.csv"
)

letter = letter[
    [
        "ID",
        "total_num",
    ]
].rename(
    columns={
        "total_num": "Letter",
    }
)


# 6. Coreference
coreference = read_result(
    RESULTS_DIR
    / "coreference"
    / "scores_with_meta.csv"
)

coreference = coreference[
    [
        "ID",
        "coref_score",
    ]
].rename(
    columns={
        "coref_score": "Coref_score",
    }
)


# 7. Working memory
working_memory = read_result(
    RESULTS_DIR
    / "working_memory"
    / "scores.csv"
)

working_memory = working_memory[
    [
        "ID",
        "dg_backward",
        "dg_forward",
        "dglt_backward",
        "dglt_forward",
    ]
]


# 8. Immediate recall
immediate = read_result(
    RESULTS_DIR
    / "immediate_recall"
    / "scores_with_auto_metrics.csv"
)

immediate = immediate[
    [
        "ID",
        "n_entities",
        "n_events",
    ]
].rename(
    columns={
        "n_entities": "immediate_entity_n",
        "n_events": "immediate_event_n",
    }
)


# 9. Delayed recall
delay = read_result(
    RESULTS_DIR
    / "delayed_recall"
    / "scores_with_auto_metrics.csv"
)

delay = delay[
    [
        "ID",
        "n_entities",
        "n_events",
    ]
].rename(
    columns={
        "n_entities": "delay_entity_n",
        "n_events": "delay_event_n",
    }
)


# 10. Procedure
procedure = read_result(
    RESULTS_DIR
    / "procedure"
    / "scores_with_meta.csv"
)

procedure = procedure[
    [
        "ID",
        "step_score",
    ]
].rename(
    columns={
        "step_score": "procedure_key_steps_score",
    }
)


# 11. Scene construction
scene = read_result(
    RESULTS_DIR
    / "scene_construction"
    / "scores_with_auto_metrics.csv"
)

scene = scene[
    [
        "ID",
        "EP",
        "NONEP",
    ]
].rename(
    columns={
        "EP": "scene_ep_span_rate",
        "NONEP": "scene_nonep_span_rate",
    }
)


# 12. Merge all tasks
performance = animal.copy()

task_tables = [
    letter,
    coreference,
    working_memory,
    immediate,
    delay,
    procedure,
    scene,
]

for task_table in task_tables:
    performance = performance.merge(
        task_table,
        on="ID",
        how="left",
        validate="one_to_one",
    )


# 13. Add Alpha and Scope
condition_to_alpha = {
    "Original": 0.0,
    "0.6Alpha_10000Neurons": 0.6,
    "0.6Alpha_AllSigNeurons": 0.6,
}

condition_to_scope = {
    "Original": 0,
    "0.6Alpha_10000Neurons": 10000,
    "0.6Alpha_AllSigNeurons": "AllSig",
}

performance.insert(
    3,
    "Alpha",
    performance["Condition"].map(condition_to_alpha),
)

performance.insert(
    4,
    "Scope",
    performance["Condition"].map(condition_to_scope),
)


# 14. Use exactly the same column order as the provided file
output_columns = [
    "ID",
    "Condition",
    "Role",
    "Alpha",
    "Scope",
    "Animal",
    "Letter",
    "Coref_score",
    "dg_backward",
    "dg_forward",
    "dglt_backward",
    "dglt_forward",
    "immediate_entity_n",
    "immediate_event_n",
    "delay_entity_n",
    "delay_event_n",
    "procedure_key_steps_score",
    "scene_ep_span_rate",
    "scene_nonep_span_rate",
]

performance = performance[output_columns]


# 15. Sort conditions and roles
condition_order = {
    "Original": 0,
    "0.6Alpha_10000Neurons": 1,
    "0.6Alpha_AllSigNeurons": 2,
}

performance["_condition_order"] = (
    performance["Condition"].map(condition_order)
)

performance = (
    performance
    .sort_values(
        [
            "_condition_order",
            "Role",
        ]
    )
    .drop(columns="_condition_order")
    .reset_index(drop=True)
)


# 16. Check the merged results
expected_n = 25 * len(SELECTED_CONDITIONS)

if len(performance) != expected_n:
    print(
        f"Warning: expected {expected_n} rows, "
        f"but obtained {len(performance)} rows."
    )

missing_counts = performance.isna().sum()

if missing_counts.sum() > 0:
    print("\nMissing values:")
    print(
        missing_counts[
            missing_counts > 0
        ]
    )
else:
    print("No missing task scores were found.")


print("\nCases per condition:")
print(
    performance
    .groupby("Condition")
    .size()
)

print("\nOutput shape:")
print(performance.shape)

print("\nPreview:")
print(performance.head())


# 17. Save
performance.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

performance.to_excel(
    OUTPUT_XLSX,
    index=False,
)

print(f"\nSaved CSV: {OUTPUT_CSV}")
print(f"Saved Excel: {OUTPUT_XLSX}")