# 1. Packages
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)


# 2. Paths
MAIN_FILE = "all_task_performance_main.xlsx"
RANDOM_FILE = "all_task_performance_random.xlsx"

OUTPUT_DIR = Path("random_neuron_control")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 3. Conditions
ORIGINAL = "Original"

RANDOM_10000 = "0.6Alpha_10000RandomNeurons"
RANDOM_ALLSIG = "0.6Alpha_AllSigRandomNeurons"

AD_10000 = "0.6Alpha_10000Neurons"
AD_ALLSIG = "0.6Alpha_AllSigNeurons"

# Original is the GEE reference condition.
CONDITION_ORDER = [
    ORIGINAL,
    RANDOM_10000,
    RANDOM_ALLSIG,
    AD_10000,
    AD_ALLSIG,
]

# Plot order is symmetric around Original.
PLOT_ORDER = [
    RANDOM_ALLSIG,
    RANDOM_10000,
    ORIGINAL,
    AD_10000,
    AD_ALLSIG,
]

X_POS = {
    RANDOM_ALLSIG: -2,
    RANDOM_10000: -1,
    ORIGINAL: 0,
    AD_10000: 1,
    AD_ALLSIG: 2,
}

X_LABELS = [
    "Random\nAll sig.",
    "Random\n10k",
    "Original",
    "AD-guided\n10k",
    "AD-guided\nAll sig.",
]


# 4. Outcomes
METRIC_SPECS = [
    ("Animal", "Animal fluency"),
    ("Letter", "Letter fluency"),
    ("Coref_score", "Total score"),
    ("procedure_key_steps_score", "Key procedural steps"),
    ("dg_backward", "DG backward"),
    ("dg_forward", "DG forward"),
    ("dglt_backward", "DGLT backward"),
    ("dglt_forward", "DGLT forward"),
    ("immediate_entity_n", "Entities recalled"),
    ("immediate_event_n", "Events recalled"),
    ("delay_entity_n", "Entities recalled"),
    ("delay_event_n", "Events recalled"),
    ("scene_ep_span_rate", "Episodic details"),
    ("scene_nonep_span_rate", "Non-episodic details"),
]

METRICS = [metric for metric, label in METRIC_SPECS]
METRIC_LABELS = dict(METRIC_SPECS)

TASK_LABELS = {
    "Animal": "Verbal fluency",
    "Letter": "Verbal fluency",
    "Coref_score": "Coreference resolution",
    "procedure_key_steps_score": "Procedural discourse",
    "dg_backward": "Working memory",
    "dg_forward": "Working memory",
    "dglt_backward": "Working memory",
    "dglt_forward": "Working memory",
    "immediate_entity_n": "Immediate recall",
    "immediate_event_n": "Immediate recall",
    "delay_entity_n": "Delayed recall",
    "delay_event_n": "Delayed recall",
    "scene_ep_span_rate": "Scene construction",
    "scene_nonep_span_rate": "Scene construction",
}


# 5. Appendix page structure
PAGE_1_GROUPS = [
    {
        "title": "Verbal fluency",
        "items": [
            ("Animal", "Animal fluency", 0, 0),
            ("Letter", "Letter fluency", 0, 1),
        ],
    },
    {
        "title": "Coreference resolution",
        "items": [
            ("Coref_score", "Total score", 0, 2),
        ],
    },
    {
        "title": "Procedural discourse",
        "items": [
            (
                "procedure_key_steps_score",
                "Key procedural steps",
                0,
                3,
            ),
        ],
    },
    {
        "title": "Working memory",
        "items": [
            ("dg_backward", "DG backward", 1, 0),
            ("dg_forward", "DG forward", 1, 1),
            ("dglt_backward", "DGLT backward", 1, 2),
            ("dglt_forward", "DGLT forward", 1, 3),
        ],
    },
]

PAGE_2_GROUPS = [
    {
        "title": "Immediate recall",
        "items": [
            (
                "immediate_entity_n",
                "Entities recalled",
                0,
                0,
            ),
            (
                "immediate_event_n",
                "Events recalled",
                0,
                1,
            ),
        ],
    },
    {
        "title": "Delayed recall",
        "items": [
            (
                "delay_entity_n",
                "Entities recalled",
                0,
                2,
            ),
            (
                "delay_event_n",
                "Events recalled",
                0,
                3,
            ),
        ],
    },
    {
        "title": "Scene construction",
        "items": [
            (
                "scene_ep_span_rate",
                "Episodic details",
                1,
                1,
            ),
            (
                "scene_nonep_span_rate",
                "Non-episodic details",
                1,
                2,
            ),
        ],
    },
]


# 6. Plot style
SCOPE_CONFIG = {
    "10000": {
        "random": RANDOM_10000,
        "ad": AD_10000,
        "random_x": -1,
        "ad_x": 1,
        "color": "#E45756",
        "marker": "s",
        "label": "10,000 neurons",
    },
    "AllSig": {
        "random": RANDOM_ALLSIG,
        "ad": AD_ALLSIG,
        "random_x": -2,
        "ad_x": 2,
        "color": "#72B7B2",
        "marker": "D",
        "label": "All significant neurons",
    },
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# 7. Read data
main = pd.read_excel(MAIN_FILE)
random = pd.read_excel(RANDOM_FILE)

data = pd.concat(
    [main, random],
    ignore_index=True,
)

data["Role"] = (
    data["Role"]
    .astype(str)
    .str.strip()
)

data["Condition"] = (
    data["Condition"]
    .astype(str)
    .str.strip()
)

data = data[
    data["Condition"].isin(CONDITION_ORDER)
].copy()

for metric in METRICS:
    data[metric] = pd.to_numeric(
        data[metric],
        errors="coerce",
    )


# 8. Validate matched design
if data.duplicated(
    subset=["Role", "Condition"]
).any():
    raise ValueError(
        "Duplicate Role × Condition rows were found."
    )

roles_by_condition = [
    set(
        data.loc[
            data["Condition"] == condition,
            "Role",
        ]
    )
    for condition in CONDITION_ORDER
]

common_roles = set.intersection(
    *roles_by_condition
)

if not common_roles:
    raise ValueError(
        "No roles are shared by all five conditions."
    )

data = data[
    data["Role"].isin(common_roles)
].copy()

data["Condition"] = pd.Categorical(
    data["Condition"],
    categories=CONDITION_ORDER,
    ordered=True,
)

print(
    f"Completely matched roles: "
    f"{len(common_roles)}"
)

print("\nCases per condition:")

print(
    data.groupby(
        "Condition",
        observed=True,
    ).size()
)


# 9. Descriptive summaries
long_data = data.melt(
    id_vars=[
        "Role",
        "Condition",
    ],
    value_vars=METRICS,
    var_name="metric",
    value_name="value",
)

summary = (
    long_data
    .dropna(
        subset=[
            "Role",
            "Condition",
            "value",
        ]
    )
    .groupby(
        [
            "metric",
            "Condition",
        ],
        observed=True,
    )["value"]
    .agg([
        "mean",
        "std",
        "count",
    ])
    .reset_index()
    .rename(
        columns={
            "count": "n",
        }
    )
)

summary["se"] = (
    summary["std"]
    / np.sqrt(summary["n"])
).fillna(0)

summary["ci"] = (
    1.96
    * summary["se"]
)

summary["ci_low"] = (
    summary["mean"]
    - summary["ci"]
)

summary["ci_high"] = (
    summary["mean"]
    + summary["ci"]
)

summary["task"] = (
    summary["metric"]
    .map(TASK_LABELS)
)

summary["metric_label"] = (
    summary["metric"]
    .map(METRIC_LABELS)
)

summary.to_csv(
    OUTPUT_DIR
    / "random_control_descriptive_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

summary_lookup = summary.set_index(
    [
        "metric",
        "Condition",
    ]
)


# 10. GEE helper functions
def condition_term(condition):
    """Return the statsmodels coefficient name for one condition."""

    return (
        'C(Condition, Treatment(reference="Original"))'
        f'[T.{condition}]'
    )


def gee_contrast(result, weights, tolerance=1e-12):
    """Test a linear contrast using the robust GEE covariance matrix."""

    contrast_vector = np.zeros(
        len(result.params),
        dtype=float,
    )

    for term, weight in weights.items():

        if term not in result.params.index:
            raise ValueError(
                f"Missing GEE coefficient: {term}"
            )

        term_index = (
            result.params.index
            .get_loc(term)
        )

        contrast_vector[
            term_index
        ] = weight

    estimate = float(
        contrast_vector
        @ result.params.to_numpy()
    )

    variance = float(
        contrast_vector
        @ result.cov_robust
        @ contrast_vector
    )

    # Correct tiny negative values caused by floating-point precision.
    if (
        variance < 0
        and abs(variance) < tolerance
    ):
        variance = 0.0

    if (
        variance < 0
        or not np.isfinite(variance)
    ):
        return (
            estimate,
            np.nan,
            np.nan,
            np.nan,
        )

    se = float(
        np.sqrt(variance)
    )

    if se <= tolerance:

        if abs(estimate) <= tolerance:
            z_value = 0.0
            p_value = 1.0

        else:
            z_value = (
                np.sign(estimate)
                * np.inf
            )
            p_value = 0.0

    else:
        z_value = estimate / se

        p_value = float(
            2
            * norm.sf(
                abs(z_value)
            )
        )

    return (
        estimate,
        se,
        z_value,
        p_value,
    )


# 11. Planned contrasts
CONTRASTS = [
    {
        "contrast": "Random 10k vs Original",
        "comparison": "random_vs_original",
        "scope": "10000",
        "condition_a": RANDOM_10000,
        "condition_b": ORIGINAL,
        "weights": {
            condition_term(
                RANDOM_10000
            ): 1.0,
        },
    },
    {
        "contrast": "Random AllSig vs Original",
        "comparison": "random_vs_original",
        "scope": "AllSig",
        "condition_a": RANDOM_ALLSIG,
        "condition_b": ORIGINAL,
        "weights": {
            condition_term(
                RANDOM_ALLSIG
            ): 1.0,
        },
    },
    {
        "contrast": "Random 10k vs AD 10k",
        "comparison": "random_vs_ad",
        "scope": "10000",
        "condition_a": RANDOM_10000,
        "condition_b": AD_10000,
        "weights": {
            condition_term(
                RANDOM_10000
            ): 1.0,
            condition_term(
                AD_10000
            ): -1.0,
        },
    },
    {
        "contrast": "Random AllSig vs AD AllSig",
        "comparison": "random_vs_ad",
        "scope": "AllSig",
        "condition_a": RANDOM_ALLSIG,
        "condition_b": AD_ALLSIG,
        "weights": {
            condition_term(
                RANDOM_ALLSIG
            ): 1.0,
            condition_term(
                AD_ALLSIG
            ): -1.0,
        },
    },
]


# 12. Fit GEE models
# Gaussian identity GEE is used here because the combined outcomes
# contain exact zeros, while Coref_score can also be negative.
# The repeated-measures model remains:
# outcome ~ Role + Condition, grouped by Role.

gee_rows = []

for metric in METRICS:

    analysis_data = data[
        [
            "Role",
            "Condition",
            metric,
        ]
    ].dropna().copy()

    formula = (
        f'Q("{metric}") ~ '
        f'C(Role) + '
        f'C(Condition, '
        f'Treatment(reference="Original"))'
    )

    result = smf.gee(
        formula=formula,
        groups="Role",
        data=analysis_data,
        family=sm.families.Gaussian(
            link=sm.families.links.Identity()
        ),
        cov_struct=(
            sm.cov_struct.Exchangeable()
        ),
    ).fit()

    condition_means = (
        analysis_data
        .groupby(
            "Condition",
            observed=True,
        )[metric]
        .mean()
    )

    pearson_ratio = (
        float(
            result.pearson_chi2
            / result.df_resid
        )
        if result.df_resid > 0
        else np.nan
    )

    for contrast in CONTRASTS:

        estimate, se, z_value, p_value = (
            gee_contrast(
                result,
                contrast["weights"],
            )
        )

        condition_a = contrast[
            "condition_a"
        ]

        condition_b = contrast[
            "condition_b"
        ]

        gee_rows.append({
            "task": TASK_LABELS[metric],
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "contrast": contrast["contrast"],
            "comparison": contrast["comparison"],
            "scope": contrast["scope"],
            "condition_a": condition_a,
            "condition_b": condition_b,
            "mean_a": float(
                condition_means.loc[
                    condition_a
                ]
            ),
            "mean_b": float(
                condition_means.loc[
                    condition_b
                ]
            ),
            "raw_mean_difference_a_minus_b": float(
                condition_means.loc[
                    condition_a
                ]
                - condition_means.loc[
                    condition_b
                ]
            ),
            "gee_estimate": estimate,
            "se": se,
            "z": z_value,
            "p": p_value,
            "family": "Gaussian identity",
            "pearson_ratio": pearson_ratio,
            "n_obs": int(result.nobs),
            "converged": bool(result.converged),
        })


statistics_results = pd.DataFrame(
    gee_rows
)


# 13. FDR correction
# Four planned comparisons are corrected separately within each outcome.
statistics_results["q"] = np.nan

for metric in METRICS:

    metric_index = (
        statistics_results["metric"]
        == metric
    )

    valid_index = (
        metric_index
        & statistics_results["p"].notna()
    )

    if valid_index.any():

        statistics_results.loc[
            valid_index,
            "q",
        ] = multipletests(
            statistics_results.loc[
                valid_index,
                "p",
            ],
            method="fdr_bh",
        )[1]


statistics_results[
    "significant_fdr"
] = (
    statistics_results["q"]
    < 0.05
)

metric_order = {
    metric: index
    for index, metric in enumerate(
        METRICS
    )
}

contrast_order = {
    contrast["contrast"]: index
    for index, contrast in enumerate(
        CONTRASTS
    )
}

statistics_results["_metric_order"] = (
    statistics_results["metric"]
    .map(metric_order)
)

statistics_results["_contrast_order"] = (
    statistics_results["contrast"]
    .map(contrast_order)
)

statistics_results = (
    statistics_results
    .sort_values(
        [
            "_metric_order",
            "_contrast_order",
        ]
    )
    .drop(
        columns=[
            "_metric_order",
            "_contrast_order",
        ]
    )
    .reset_index(drop=True)
)

statistics_results.to_csv(
    OUTPUT_DIR
    / "random_control_gee_planned_contrasts.csv",
    index=False,
    encoding="utf-8-sig",
)

statistics_lookup = (
    statistics_results
    .set_index(
        [
            "metric",
            "contrast",
        ]
    )
)


# 14. Specificity classification
random_original = (
    statistics_results[
        statistics_results["comparison"]
        == "random_vs_original"
    ][
        [
            "task",
            "metric",
            "metric_label",
            "scope",
            "mean_a",
            "mean_b",
            "raw_mean_difference_a_minus_b",
            "p",
            "q",
        ]
    ]
    .rename(
        columns={
            "mean_a": "random_mean",
            "mean_b": "original_mean",
            "raw_mean_difference_a_minus_b":
                "random_minus_original",
            "p": "p_random_vs_original",
            "q": "q_random_vs_original",
        }
    )
)

random_ad = (
    statistics_results[
        statistics_results["comparison"]
        == "random_vs_ad"
    ][
        [
            "metric",
            "scope",
            "mean_b",
            "raw_mean_difference_a_minus_b",
            "p",
            "q",
        ]
    ]
    .rename(
        columns={
            "mean_b": "ad_guided_mean",
            "raw_mean_difference_a_minus_b":
                "random_minus_ad_guided",
            "p": "p_random_vs_ad",
            "q": "q_random_vs_ad",
        }
    )
)

specificity = random_original.merge(
    random_ad,
    on=[
        "metric",
        "scope",
    ],
    how="inner",
    validate="one_to_one",
)

random_original_significant = (
    specificity[
        "q_random_vs_original"
    ]
    < 0.05
)

random_ad_significant = (
    specificity[
        "q_random_vs_ad"
    ]
    < 0.05
)

specificity["interpretation"] = np.select(
    [
        (
            ~random_original_significant
            & random_ad_significant
        ),
        (
            random_original_significant
            & ~random_ad_significant
        ),
        (
            random_original_significant
            & random_ad_significant
        ),
    ],
    [
        "Selection-specific evidence",
        "Likely non-specific perturbation",
        (
            "Random perturbation occurs but differs "
            "from AD-guided editing"
        ),
    ],
    default=(
        "No FDR-significant random-control evidence"
    ),
)

specificity.to_csv(
    OUTPUT_DIR
    / "random_control_specificity_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# 15. Figure settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.2,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

X_LABELS = ["Rnd\nAllSig", "Rnd\n10k", "Orig", "AD\n10k", "AD\nAllSig"]

PANEL_LAYOUT = [
    # Row 1: Immediate and delayed recall
    ("immediate_entity_n", "Entities", 0, 0),
    ("immediate_event_n", "Events", 0, 1),
    ("delay_entity_n", "Entities", 0, 2),
    ("delay_event_n", "Events", 0, 3),

    # Row 2: Verbal fluency, coreference, and procedure
    ("Animal", "Animal fluency", 1, 0),
    ("Letter", "Letter fluency", 1, 1),
    ("Coref_score", "Total score", 1, 2),
    (
        "procedure_key_steps_score",
        "Key procedural steps",
        1,
        3,
    ),

    # Row 3: Working memory
    ("dg_backward", "DG backward", 2, 0),
    ("dg_forward", "DG forward", 2, 1),
    ("dglt_backward", "DGLT backward", 2, 2),
    ("dglt_forward", "DGLT forward", 2, 3),

    # Row 4: Scene construction
    (
        "scene_ep_span_rate",
        "EP",
        3,
        0,
    ),
    (
        "scene_nonep_span_rate",
        "NONEP",
        3,
        1,
    ),
]

TASK_GROUPS = [
    ("Immediate Recall\n", [(0, 0), (0, 1)]),
    ("Delayed Recall\n", [(0, 2), (0, 3)]),
    ("Verbal fluency\n", [(1, 0), (1, 1)]),
    ("Coreference\n", [(1, 2)]),
    ("Procedure\n", [(1, 3)]),
    ("Working memory\n", [(2, 0), (2, 1), (2, 2), (2, 3)]),
    ("Scene construction\n", [(3, 0), (3, 1)]),
]


# 16. Helpers
def significance_symbol(q_value, symbol):
    if pd.isna(q_value):
        return ""
    if q_value < 0.001:
        return symbol * 3
    if q_value < 0.01:
        return symbol * 2
    if q_value < 0.05:
        return symbol
    return ""


def add_bracket(ax, x1, x2, y, height, label, color):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color=color, linewidth=0.8, clip_on=False, zorder=7)
    ax.text((x1 + x2) / 2, y + height, label, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="0.12", clip_on=False, zorder=8)


def plot_metric(ax, metric, metric_label):
    condition_rows = {condition: summary_lookup.loc[(metric, condition)] for condition in PLOT_ORDER}

    original_mean = float(condition_rows[ORIGINAL]["mean"])
    original_ci = float(condition_rows[ORIGINAL]["ci"])

    panel_low = [original_mean - original_ci]
    panel_high = [original_mean + original_ci]

    ax.axvspan(-2.35, -0.05, color="0.955", zorder=0)
    ax.axvspan(0.05, 2.35, color="0.985", zorder=0)
    ax.axvline(0, color="0.80", linewidth=0.7, zorder=1)

    for scope in ["AllSig", "10000"]:
        config = SCOPE_CONFIG[scope]
        random_row = condition_rows[config["random"]]
        ad_row = condition_rows[config["ad"]]

        random_mean = float(random_row["mean"])
        random_ci = float(random_row["ci"])
        ad_mean = float(ad_row["mean"])
        ad_ci = float(ad_row["ci"])

        ax.plot([config["random_x"], 0], [random_mean, original_mean], linestyle="--", linewidth=1.0, color=config["color"], zorder=2)
        ax.plot([0, config["ad_x"]], [original_mean, ad_mean], linestyle="-", linewidth=1.1, color=config["color"], zorder=2)

        ax.errorbar(config["random_x"], random_mean, yerr=random_ci, fmt=config["marker"], markersize=4.8, markerfacecolor="white", markeredgecolor=config["color"], markeredgewidth=1.0, capsize=2.2, elinewidth=0.8, color=config["color"], zorder=4)
        ax.errorbar(config["ad_x"], ad_mean, yerr=ad_ci, fmt=config["marker"], markersize=4.8, markerfacecolor=config["color"], markeredgecolor=config["color"], capsize=2.2, elinewidth=0.8, color=config["color"], zorder=4)

        panel_low.extend([random_mean - random_ci, ad_mean - ad_ci])
        panel_high.extend([random_mean + random_ci, ad_mean + ad_ci])

    ax.errorbar(0, original_mean, yerr=original_ci, fmt="o", markersize=5.0, markerfacecolor="0.15", markeredgecolor="0.15", capsize=2.2, elinewidth=0.8, color="0.15", zorder=5)

    base_min = min(panel_low)
    base_max = max(panel_high)
    base_range = base_max - base_min
    if not np.isfinite(base_range) or base_range == 0:
        base_range = max(abs(base_max), 1.0)

    contrast_names = {
        ("10000", "original"): "Random 10k vs Original",
        ("AllSig", "original"): "Random AllSig vs Original",
        ("10000", "ad"): "Random 10k vs AD 10k",
        ("AllSig", "ad"): "Random AllSig vs AD AllSig",
    }

    brackets = []
    for scope in ["10000", "AllSig"]:
        config = SCOPE_CONFIG[scope]
        q_original = statistics_lookup.loc[(metric, contrast_names[(scope, "original")]), "q"]
        q_ad = statistics_lookup.loc[(metric, contrast_names[(scope, "ad")]), "q"]

        star = significance_symbol(q_original, "*")
        hash_symbol = significance_symbol(q_ad, "#")

        if star:
            brackets.append({"x1": config["random_x"], "x2": 0, "label": star, "color": "0.25"})
        if hash_symbol:
            brackets.append({"x1": config["random_x"], "x2": config["ad_x"], "label": hash_symbol, "color": config["color"]})

    brackets = sorted(brackets, key=lambda item: abs(item["x2"] - item["x1"]))

    bracket_height = 0.020 * base_range
    bracket_spacing = 0.082 * base_range
    first_bracket_y = base_max + 0.040 * base_range

    for index, bracket in enumerate(brackets):
        bracket_y = first_bracket_y + index * bracket_spacing
        add_bracket(ax, bracket["x1"], bracket["x2"], bracket_y, bracket_height, bracket["label"], bracket["color"])

    y_max = first_bracket_y + max(len(brackets), 1) * bracket_spacing + 0.035 * base_range
    y_min = max(0.0, base_min - 0.08 * base_range) if base_min >= 0 else base_min - 0.08 * base_range

    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(y_min, y_max)
    ax.set_title(metric_label, pad=3.2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(X_LABELS, linespacing=0.8)
    ax.tick_params(axis="both", pad=1.4, length=2.3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.16)


def add_task_header(fig, axes_list, title):
    positions = [ax.get_position() for ax in axes_list]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    top = max(position.y1 for position in positions)

    header_y = top + 0.004
    header_height = 0.024

    header = Rectangle((left, header_y), right - left, header_height, transform=fig.transFigure, facecolor="#E9EEF1", edgecolor="none", clip_on=False, zorder=0)
    fig.add_artist(header)

    fig.text((left + right) / 2, header_y + header_height / 2, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="0.15")


def match_y_limits(axes_list):
    y_min = min(ax.get_ylim()[0] for ax in axes_list)
    y_max = max(ax.get_ylim()[1] for ax in axes_list)
    for ax in axes_list:
        ax.set_ylim(y_min, y_max)


# 17. Legend
legend_handles = [
    Line2D([0], [0], color=SCOPE_CONFIG["10000"]["color"], marker="s", markerfacecolor=SCOPE_CONFIG["10000"]["color"], linewidth=1.0, markersize=4.9, label="10,000 neurons"),
    Line2D([0], [0], color=SCOPE_CONFIG["AllSig"]["color"], marker="D", markerfacecolor=SCOPE_CONFIG["AllSig"]["color"], linewidth=1.0, markersize=4.9, label="All significant neurons"),
    Line2D([0], [0], color="0.35", marker="o", markerfacecolor="white", markeredgecolor="0.35", linestyle="--", linewidth=0.95, markersize=4.9, label="Random neurons"),
    Line2D([0], [0], color="0.35", marker="o", markerfacecolor="0.35", markeredgecolor="0.35", linestyle="-", linewidth=0.95, markersize=4.9, label="AD-guided neurons"),
    Line2D([0], [0], color="0.15", marker="o", markerfacecolor="0.15", linestyle="none", markersize=4.9, label="Original"),
    Line2D([0], [0], color="none", linestyle="none", label="* Random vs Original"),
    Line2D([0], [0], color="none", linestyle="none", label="# Random vs matched AD-guided"),
]


# 18. Create one portrait A4 figure with 4 columns
fig, axes = plt.subplots(4, 4, figsize=(8.27, 11.69))

for ax in axes.ravel():
    ax.set_visible(False)

for metric, metric_label, row, col in PANEL_LAYOUT:
    ax = axes[row, col]
    ax.set_visible(True)
    plot_metric(ax, metric, metric_label)

# Left-most visible panel in each row gets y-label
for row in range(4):
    visible_axes = [ax for ax in axes[row] if ax.get_visible()]
    if visible_axes:
        visible_axes[0].set_ylabel("Mean outcome")

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.982),
    handlelength=1.4,
    columnspacing=1.0,
    handletextpad=0.45,
    borderaxespad=0,
)

fig.subplots_adjust(
    left=0.08,
    right=0.985,
    bottom=0.05,
    top=0.92,
    wspace=0.34,
    hspace=0.72,
)

fig.canvas.draw()

for title, locations in TASK_GROUPS:
    group_axes = [axes[r, c] for r, c in locations]
    add_task_header(fig, group_axes, title)

# 19. Match scales for comparable panels
match_y_limits([axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]])
match_y_limits([axes[2, 0], axes[2, 2]])
match_y_limits([axes[2, 1], axes[2, 3]])

# 20. Save
pdf_path = OUTPUT_DIR / "random_control_task_performance_appendix.pdf"
svg_path = OUTPUT_DIR / "random_control_task_performance_appendix.svg"
png_path = OUTPUT_DIR / "random_control_task_performance_appendix.png"

fig.savefig(pdf_path)
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(png_path, dpi=400, bbox_inches="tight")

plt.show()

print(f"Saved PDF: {pdf_path}")
print(f"Saved SVG: {svg_path}")
print(f"Saved PNG: {png_path}")