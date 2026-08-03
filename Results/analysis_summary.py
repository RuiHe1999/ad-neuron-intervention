# 1. Packages
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# 2. Constants
results_dir = Path("Chat_analysis/results")
summary_dir = results_dir / "summary"
os.makedirs(summary_dir, exist_ok=True)

alpha_order = [-0.6, 0.2, 0.6]

scope_order = [
    "2000",
    "10000",
    "AllSig",
]

condition_order = [
    "Original",
    "-0.6Alpha_2000Neurons",
    "-0.6Alpha_10000Neurons",
    "-0.6Alpha_AllSigNeurons",
    "0.2Alpha_2000Neurons",
    "0.2Alpha_10000Neurons",
    "0.2Alpha_AllSigNeurons",
    "0.6Alpha_2000Neurons",
    "0.6Alpha_10000Neurons",
    "0.6Alpha_AllSigNeurons",
]

scope_labels = {
    "2000": "2,000 neurons",
    "10000": "10,000 neurons",
    "AllSig": "All significant",
}

scope_colors = {
    "2000": "#4C78A8",
    "10000": "#E45756",
    "AllSig": "#72B7B2",
}

scope_markers = {
    "2000": "o",
    "10000": "s",
    "AllSig": "D",
}

scope_offsets = {
    "2000": -0.17,
    "10000": 0.00,
    "AllSig": 0.17,
}

# Core task outcome specifications
# Ordered to keep each task group within one row of the 3 x 4 figure.
core_specs = [
    (
        "immediate_recall",
        "scores_with_auto_metrics.csv",
        None,
        "n_entities",
        "Entities",
    ),
    (
        "immediate_recall",
        "scores_with_auto_metrics.csv",
        None,
        "n_events",
        "Events",
    ),
    (
        "delayed_recall",
        "scores_with_auto_metrics.csv",
        None,
        "n_entities",
        "Entities",
    ),
    (
        "delayed_recall",
        "scores_with_auto_metrics.csv",
        None,
        "n_events",
        "Events",
    ),
    (
        "working_memory",
        "scores.csv",
        None,
        "total_error",
        "Total errors",
    ),
    (
        "verbal_fluency",
        "animal_fluency_with_meta.csv",
        "animal",
        "total_num",
        "Correct unique words (animals)",
    ),
    (
        "verbal_fluency",
        "letter_fluency_with_meta.csv",
        "letter",
        "total_num",
        "Correct unique words (letter C)",
    ),
    (
        "coreference",
        "scores_with_meta.csv",
        None,
        "coref_score",
        "Total score",
    ),
    (
        "procedure",
        "scores_with_meta.csv",
        None,
        "step_score",
        "Procedural steps",
    ),
    (
        "scene_construction",
        "scores_with_auto_metrics.csv",
        None,
        "EP",
        "Episodic details",
    ),
    (
        "scene_construction",
        "scores_with_auto_metrics.csv",
        None,
        "NONEP",
        "Non-episodic details",
    ),
    (
        "scene_construction",
        "scores_with_auto_metrics.csv",
        None,
        "OTHER",
        "Other statements",
    ),
]


# Panel indices sharing a task-level title
task_groups = [
    ([0, 1], "Immediate recall"),
    ([2, 3], "Delayed recall"),
    ([4], "Working memory"),
    ([5, 6], "Verbal fluency"),
    ([7], "Coreference resolution"),
    ([8], "Procedural discourse"),
    ([9, 10, 11], "Scene construction"),
]


# Linguistic outcome specifications
nlp_plot_specs = [
    ("n_tokens", "Token\ncount"),
    ("mattr", "Lexical\nrichness"),
    ("LexicalH", "Lexical\nsurprisal"),
    ("lex_density", "Lexical idea\ndensity"),
    ("DEPID_R", "Propositional\nidea density"),
    ("clause_ratio", "Clause\nratio"),
]

nlp_tasks = [
    ("immediate_recall", "Immediate\nrecall"),
    ("delayed_recall", "Delayed\nrecall"),
    ("scene_construction", "Scene\nconstruction"),
]


# Final-size typography
# The figure width is approximately the ACL full-text width,
# so LaTeX will not substantially shrink the fonts.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 7.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 3. Helper functions
def read_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

    return pd.read_csv(path)


def parse_condition(condition):
    condition = str(condition)

    if condition.lower() == "original":
        return pd.Series({
            "alpha": 0.0,
            "scope": "Original",
        })

    alpha_match = re.search(
        r"([+-]?\d*\.?\d+)Alpha",
        condition,
        flags=re.IGNORECASE,
    )

    scope_match = re.search(
        r"_(\d+)Neurons",
        condition,
        flags=re.IGNORECASE,
    )

    alpha = (
        float(alpha_match.group(1))
        if alpha_match
        else np.nan
    )

    if "allsig" in condition.lower():
        scope = "AllSig"
    elif scope_match:
        scope = scope_match.group(1)
    else:
        scope = None

    return pd.Series({
        "alpha": alpha,
        "scope": scope,
    })


def significance_label(q):
    if pd.isna(q):
        return ""

    if q < 0.001:
        return "***"

    if q < 0.01:
        return "**"

    if q < 0.05:
        return "*"

    return ""


def load_gee(folder, task=None):
    data = read_csv(
        results_dir
        / folder
        / "gee_condition_vs_original.csv"
    )

    if task is not None:
        if "task" not in data.columns:
            raise ValueError(
                f"The task column is missing from "
                f"{folder} results."
            )

        data = data[
            data["task"].astype(str) == task
        ].copy()

    parsed = data["condition"].apply(
        parse_condition
    )

    return pd.concat(
        [
            data.reset_index(drop=True),
            parsed.reset_index(drop=True),
        ],
        axis=1,
    )


def summarize_raw(data, var):
    data = data[
        ["Condition", var]
    ].copy()

    data[var] = pd.to_numeric(
        data[var],
        errors="coerce",
    )

    data = data.dropna(
        subset=["Condition", var]
    )

    data["Condition"] = (
        data["Condition"].astype(str)
    )

    data = data[
        data["Condition"].isin(condition_order)
    ]

    summary = (
        data
        .groupby(
            "Condition",
            observed=True,
        )[var]
        .agg([
            "mean",
            "std",
            "count",
        ])
        .reset_index()
        .rename(columns={
            "Condition": "condition",
            "count": "n",
        })
    )

    summary["se"] = (
        summary["std"]
        / np.sqrt(summary["n"])
    )

    summary.loc[
        summary["n"] <= 1,
        "se",
    ] = 0.0

    summary["ci"] = 1.96 * summary["se"]

    summary["ci_low"] = (
        summary["mean"]
        - summary["ci"]
    )

    summary["ci_high"] = (
        summary["mean"]
        + summary["ci"]
    )

    parsed = summary["condition"].apply(
        parse_condition
    )

    return pd.concat(
        [
            summary.reset_index(drop=True),
            parsed.reset_index(drop=True),
        ],
        axis=1,
    )


def save_figure(fig, name):
    fig.savefig(
        summary_dir / f"{name}.svg",
        bbox_inches="tight",
    )

    fig.savefig(
        summary_dir / f"{name}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        summary_dir / f"{name}.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

# 4. Core cognitive task outcomes
core_frames = []

for panel, spec in enumerate(core_specs):

    folder, raw_file, task, var, label = spec

    raw = read_csv(
        results_dir
        / folder
        / raw_file
    )

    if "Condition" not in raw.columns:
        raise ValueError(
            f"Condition column is missing from "
            f"{folder}/{raw_file}."
        )

    if var not in raw.columns:
        raise ValueError(
            f"Missing raw outcome for "
            f"{folder}: {var}"
        )

    raw_summary = summarize_raw(
        raw,
        var,
    )

    gee = load_gee(
        folder,
        task,
    )

    gee = gee[
        gee["var"].astype(str) == var
    ].copy()

    if gee.empty:
        raise ValueError(
            f"Missing GEE result for "
            f"{folder}: {var}"
        )

    q_values = (
        gee[[
                "condition",
                "p_adj",
            ]
        ]
        .drop_duplicates(
            subset="condition",
            keep="last",
        )
    )

    raw_summary = raw_summary.merge(
        q_values,
        on="condition",
        how="left",
    )

    raw_summary["panel"] = panel
    raw_summary["folder"] = folder
    raw_summary["task"] = task
    raw_summary["var"] = var
    raw_summary["outcome"] = label

    core_frames.append(
        raw_summary
    )


core_summary = pd.concat(
    core_frames,
    ignore_index=True,
)

core_summary["condition"] = pd.Categorical(
    core_summary["condition"],
    categories=condition_order,
    ordered=True,
)

core_summary = core_summary.sort_values(
    [
        "panel",
        "condition",
    ]
)

core_summary.to_csv(
    summary_dir
    / "core_outcomes_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# Final page-width figure
# Twelve outcomes fit a 3 x 4 layout.
core_nrows = 3
core_ncols = 4

fig, axes = plt.subplots(
    core_nrows,
    core_ncols,
    figsize=(7.2, 5.5),
    sharex=True,
)

axes = axes.ravel()


for panel, (ax, spec) in enumerate(
    zip(axes, core_specs)
):

    folder, raw_file, task, var, label = spec

    data = core_summary[
        core_summary["panel"] == panel
    ].copy()

    original = data[
        data["condition"].astype(str)
        == "Original"
    ]

    if original.empty:
        raise ValueError(
            f"Original condition is missing "
            f"for {folder}: {var}"
        )

    original = original.iloc[0]

    original_mean = float(
        original["mean"]
    )

    original_ci = float(
        original["ci"]
    )

    panel_low = [
        original_mean - original_ci
    ]

    panel_high = [
        original_mean + original_ci
    ]

    star_items = []


    # Original model
    ax.errorbar(
        0,
        original_mean,
        yerr=original_ci,
        marker="o",
        markersize=3.4,
        linewidth=0.85,
        capsize=1.8,
        color="0.20",
        zorder=4,
    )


    # Edited models
    for scope in scope_order:

        scope_data = (
            data[
                data["scope"] == scope
            ]
            .set_index("alpha")
        )

        x_values = []
        means = []
        errors = []
        q_values = []

        for x, alpha in enumerate(
            alpha_order,
            start=1,
        ):

            if alpha not in scope_data.index:
                continue

            row = scope_data.loc[alpha]

            if isinstance(
                row,
                pd.DataFrame,
            ):
                row = row.iloc[0]

            x_values.append(
                x + scope_offsets[scope]
            )

            means.append(
                float(row["mean"])
            )

            errors.append(
                float(row["ci"])
            )

            q_values.append(
                pd.to_numeric(
                    row["p_adj"],
                    errors="coerce",
                )
            )


        if not x_values:
            continue


        ax.plot(
            [0] + x_values,
            [original_mean] + means,
            color=scope_colors[scope],
            linewidth=0.95,
            zorder=2,
        )

        ax.errorbar(
            x_values,
            means,
            yerr=errors,
            linestyle="none",
            marker=scope_markers[scope],
            markersize=3.2,
            capsize=1.8,
            elinewidth=0.8,
            color=scope_colors[scope],
            zorder=3,
        )


        ci_low = (
            np.array(means)
            - np.array(errors)
        )

        ci_high = (
            np.array(means)
            + np.array(errors)
        )

        panel_low.extend(
            ci_low.tolist()
        )

        panel_high.extend(
            ci_high.tolist()
        )


        for x_value, upper, q_value in zip(
            x_values,
            ci_high,
            q_values,
        ):

            star = significance_label(
                q_value
            )

            if star:
                star_items.append(
                    (
                        x_value,
                        float(upper),
                        star,
                    )
                )


    base_min = min(panel_low)
    base_max = max(panel_high)

    base_range = (
        base_max
        - base_min
    )

    if (
        not np.isfinite(base_range)
        or base_range == 0
    ):
        base_range = max(
            abs(base_max),
            1.0,
        )


    # Put significance markers above confidence intervals
    star_offset = 0.025 * base_range

    for x_value, upper, star in star_items:

        star_y = (
            upper
            + star_offset
        )

        panel_high.append(
            star_y
        )

        ax.text(
            x_value,
            star_y,
            star,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            clip_on=False,
        )


    y_min = min(panel_low)
    y_max = max(panel_high)

    y_range = (
        y_max
        - y_min
    )

    if (
        not np.isfinite(y_range)
        or y_range == 0
    ):
        y_range = max(
            abs(y_max),
            1.0,
        )

    lower_padding = (
        0.08 * y_range
    )

    upper_padding = (
        0.12 * y_range
    )

    if y_min >= 0:
        y_min = max(
            0.0,
            y_min - lower_padding,
        )
    else:
        y_min = (
            y_min
            - lower_padding
        )

    ax.set_ylim(
        y_min,
        y_max + upper_padding,
    )


    # Panel formatting
    ax.set_title(
        label,
        fontsize=7.3,
        pad=2.5,
    )

    ax.set_xticks(
        range(4)
    )

    ax.set_xticklabels([
        "Orig.",
        "−0.6",
        "0.2",
        "0.6",
    ])

    ax.tick_params(
        axis="both",
        labelsize=6.2,
        pad=1.5,
        length=2.5,
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=4)
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.grid(
        axis="y",
        linewidth=0.4,
        alpha=0.18,
    )


    panel_row = panel // core_ncols

    if panel_row < core_nrows - 1:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )
    else:
        ax.set_xlabel(
            r"Scaling factor $\alpha$",
            fontsize=6.8,
            labelpad=2,
        )


# Hide axes that are not used by an outcome.
for ax in axes[len(core_specs):]:
    ax.set_visible(False)


for ax in axes[::core_ncols]:
    ax.set_ylabel(
        "Mean outcome",
        fontsize=7,
        labelpad=2,
    )


# Shared legend
core_legend_handles = [
    Line2D(
        [0],
        [0],
        color=scope_colors[scope],
        marker=scope_markers[scope],
        linewidth=1.0,
        markersize=3.8,
        label=scope_labels[scope],
    )
    for scope in scope_order
]

fig.legend(
    handles=core_legend_handles,
    loc="upper center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 0.03),
    fontsize=8,
    handlelength=1.5,
    columnspacing=1.4,
)


fig.tight_layout(
    rect=[
        0.025,
        0.025,
        1,
        0.88,
    ],
    h_pad=2.8,
    w_pad=0.9,
)

fig.canvas.draw()


# Add task-level headings after layout has been calculated
for panel_indices, group_title in task_groups:

    positions = [
        axes[index].get_position()
        for index in panel_indices
    ]

    left = min(
        position.x0
        for position in positions
    )

    right = max(
        position.x1
        for position in positions
    )

    top = max(
        position.y1
        for position in positions
    )

    fig.text(
        (left + right) / 2,
        top + 0.028,
        group_title,
        ha="center",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
    )

plt.show()

save_figure(
    fig,
    "core_task_outcomes",
)

# 5. Linguistic outcomes
nlp_frames = []


for task_index, (folder, task_label) in enumerate(
    nlp_tasks
):

    raw = read_csv(
        results_dir
        / folder
        / "scores_with_auto_metrics.csv"
    )

    if "Condition" not in raw.columns:
        raise ValueError(
            f"Condition column is missing from "
            f"{folder}/scores_with_auto_metrics.csv."
        )

    gee = load_gee(
        folder
    )


    for var, label in nlp_plot_specs:

        if var not in raw.columns:
            raise ValueError(
                f"Missing linguistic outcome "
                f"for {folder}: {var}"
            )

        raw_summary = summarize_raw(
            raw,
            var,
        )

        gee_var = gee[
            gee["var"].astype(str) == var
        ].copy()

        if gee_var.empty:
            raise ValueError(
                f"Missing GEE result for "
                f"{folder}: {var}"
            )

        q_values = (
            gee_var[
                [
                    "condition",
                    "p_adj",
                ]
            ]
            .drop_duplicates(
                subset="condition",
                keep="last",
            )
        )

        raw_summary = raw_summary.merge(
            q_values,
            on="condition",
            how="left",
        )

        raw_summary["task_index"] = task_index
        raw_summary["folder"] = folder
        raw_summary["task_label"] = task_label
        raw_summary["var"] = var
        raw_summary["outcome"] = label

        nlp_frames.append(
            raw_summary
        )


nlp_summary = pd.concat(
    nlp_frames,
    ignore_index=True,
)

nlp_summary["condition"] = pd.Categorical(
    nlp_summary["condition"],
    categories=condition_order,
    ordered=True,
)

nlp_summary = nlp_summary.sort_values(
    [
        "var",
        "task_index",
        "condition",
    ]
)

nlp_summary.to_csv(
    summary_dir
    / "linguistic_outcomes_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# Final page-width figure
fig = plt.figure(
    figsize=(7.5, 4.7)
)

gs = fig.add_gridspec(
    nrows=4,
    ncols=7,
    height_ratios=[0.20, 1, 1, 1],
    width_ratios=[0.72, 1, 1, 1, 1, 1, 1],
    hspace=0.30,
    wspace=0.45,
)


# Feature headers
for col_index, (_, feature_label) in enumerate(
    nlp_plot_specs
):

    header_ax = fig.add_subplot(
        gs[
            0,
            col_index + 1,
        ]
    )

    header_ax.set_facecolor("#DDEFD8")

    header_ax.set_xticks([])
    header_ax.set_yticks([])

    for spine in header_ax.spines.values():
        spine.set_visible(False)

    header_ax.text(
        0.5,
        0.5,
        feature_label,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        linespacing=1.0,
        transform=header_ax.transAxes,
    )


# Empty upper-left cell
empty_ax = fig.add_subplot(
    gs[0, 0]
)

empty_ax.axis(
    "off"
)


# Legend
nlp_legend_handles = [
    Line2D(
        [0],
        [0],
        color="0.20",
        marker="o",
        linestyle="none",
        markersize=3.8,
        label="Original",
    )
]

nlp_legend_handles.extend([
    Line2D(
        [0],
        [0],
        color=scope_colors[scope],
        marker=scope_markers[scope],
        linewidth=1.0,
        markersize=3.6,
        label=scope_labels[scope],
    )
    for scope in scope_order
])


axes = np.empty(
    (
        len(nlp_tasks),
        len(nlp_plot_specs),
    ),
    dtype=object,
)

column_low = [
    [] for _ in nlp_plot_specs
]

column_high = [
    [] for _ in nlp_plot_specs
]


for row_index, (folder, task_label) in enumerate(
    nlp_tasks
):

    # Row task label
    task_ax = fig.add_subplot(
        gs[
            row_index + 1,
            0,
        ]
    )

    task_ax.set_xticks([])
    task_ax.set_yticks([])

    for spine in task_ax.spines.values():
        spine.set_visible(
            False
        )

    task_ax.text(
        0.5,
        0.5,
        task_label,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        linespacing=1.0,
        transform=task_ax.transAxes,
    )


    for col_index, (var, feature_label) in enumerate(
        nlp_plot_specs
    ):

        if row_index == 0:
            ax = fig.add_subplot(
                gs[
                    row_index + 1,
                    col_index + 1,
                ]
            )
        else:
            ax = fig.add_subplot(
                gs[
                    row_index + 1,
                    col_index + 1,
                ],
                sharey=axes[
                    0,
                    col_index,
                ],
            )

        axes[
            row_index,
            col_index,
        ] = ax


        data = nlp_summary[
            (
                nlp_summary["task_index"]
                == row_index
            )
            & (
                nlp_summary["var"].astype(str)
                == var
            )
        ].copy()


        original = data[
            data["condition"].astype(str)
            == "Original"
        ]

        if original.empty:
            raise ValueError(
                f"Original condition is missing "
                f"for {folder}: {var}"
            )

        original = original.iloc[0]

        original_mean = float(
            original["mean"]
        )

        original_ci = float(
            original["ci"]
        )

        panel_low = [
            original_mean
            - original_ci
        ]

        panel_high = [
            original_mean
            + original_ci
        ]

        star_items = []


        # Original model
        ax.errorbar(
            0,
            original_mean,
            yerr=original_ci,
            marker="o",
            markersize=2.9,
            linewidth=0.75,
            elinewidth=0.7,
            capsize=1.5,
            color="0.20",
            zorder=4,
        )


        # Edited conditions
        for scope in scope_order:

            scope_data = (
                data[
                    data["scope"] == scope
                ]
                .set_index("alpha")
            )

            x_values = []
            means = []
            errors = []
            q_values = []


            for x, alpha in enumerate(
                alpha_order,
                start=1,
            ):

                if alpha not in scope_data.index:
                    continue

                result_row = scope_data.loc[
                    alpha
                ]

                if isinstance(
                    result_row,
                    pd.DataFrame,
                ):
                    result_row = (
                        result_row.iloc[0]
                    )

                x_values.append(
                    x + scope_offsets[scope]
                )

                means.append(
                    float(
                        result_row["mean"]
                    )
                )

                errors.append(
                    float(
                        result_row["ci"]
                    )
                )

                q_values.append(
                    pd.to_numeric(
                        result_row["p_adj"],
                        errors="coerce",
                    )
                )


            if not x_values:
                continue


            ax.plot(
                [0] + x_values,
                [original_mean] + means,
                color=scope_colors[scope],
                linewidth=0.85,
                zorder=2,
            )

            ax.errorbar(
                x_values,
                means,
                yerr=errors,
                linestyle="none",
                marker=scope_markers[scope],
                markersize=2.8,
                capsize=1.5,
                elinewidth=0.7,
                color=scope_colors[scope],
                zorder=3,
            )


            ci_low = (
                np.array(means)
                - np.array(errors)
            )

            ci_high = (
                np.array(means)
                + np.array(errors)
            )

            panel_low.extend(
                ci_low.tolist()
            )

            panel_high.extend(
                ci_high.tolist()
            )


            for x_value, upper, q_value in zip(
                x_values,
                ci_high,
                q_values,
            ):

                star = significance_label(
                    q_value
                )

                if star:
                    star_items.append(
                        (
                            x_value,
                            float(upper),
                            star,
                        )
                    )


        base_min = min(
            panel_low
        )

        base_max = max(
            panel_high
        )

        base_range = (
            base_max
            - base_min
        )

        if (
            not np.isfinite(base_range)
            or base_range == 0
        ):
            base_range = max(
                abs(base_max),
                1.0,
            )


        star_offset = (
            0.04 * base_range
        )


        for x_value, upper, star in star_items:

            star_y = (
                upper
                + star_offset
            )

            panel_high.append(
                star_y
            )

            ax.text(
                x_value,
                star_y,
                star,
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                clip_on=False,
            )


        column_low[
            col_index
        ].extend(
            panel_low
        )

        column_high[
            col_index
        ].extend(
            panel_high
        )


        # Axis formatting
        ax.set_xticks(
            range(4)
        )

        ax.set_xticklabels([
            "Orig.",
            "−0.6",
            "0.2",
            "0.6",
        ])

        ax.tick_params(
            axis="both",
            labelsize=5.9,
            pad=1.2,
            length=2.2,
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )

        ax.spines["top"].set_visible(
            False
        )

        ax.spines["right"].set_visible(
            False
        )

        ax.grid(
            axis="y",
            linewidth=0.35,
            alpha=0.16,
        )


        if row_index < len(nlp_tasks) - 1:
            ax.tick_params(
                axis="x",
                labelbottom=False,
            )
        else:
            ax.set_xlabel(
                r"Scaling factor $\alpha$",
                fontsize=6.3,
                labelpad=2,
            )


# Same y-axis range within each feature column
for col_index in range(
    len(nlp_plot_specs)
):

    y_min = min(
        column_low[col_index]
    )

    y_max = max(
        column_high[col_index]
    )

    y_range = (
        y_max
        - y_min
    )

    if (
        not np.isfinite(y_range)
        or y_range == 0
    ):
        y_range = max(
            abs(y_max),
            1.0,
        )

    lower_padding = (
        0.08 * y_range
    )

    upper_padding = (
        0.18 * y_range
    )

    if y_min >= 0:
        y_min = max(
            0.0,
            y_min - lower_padding,
        )
    else:
        y_min = (
            y_min
            - lower_padding
        )

    y_max = (
        y_max
        + upper_padding
    )

    for row_index in range(
        len(nlp_tasks)
    ):
        axes[
            row_index,
            col_index,
        ].set_ylim(
            y_min,
            y_max,
        )


fig.legend(
    handles=nlp_legend_handles,
    loc="upper center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.56, 0.01),
    fontsize=7.0,
    handlelength=1.4,
    columnspacing=1.1,
)


fig.subplots_adjust(
    left=0.080,
    right=0.995,
    bottom=0.105,
    top=0.875,
)

plt.show()

save_figure(
    fig,
    "linguistic_task_outcomes",
)