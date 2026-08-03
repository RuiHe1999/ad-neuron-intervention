# 1. packages
import os
import re
import warnings

import numpy as np
import pandas as pd
import pingouin as pg

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from nltk.corpus import wordnet as wn

warnings.filterwarnings("ignore")

# 2. constants
task_name = "verbal_fluency"
fluency_output = f"Chat_analysis/results/{task_name}"
auto_output = "Chat_analysis/automated_analysis"
os.makedirs(fluency_output, exist_ok=True)
os.makedirs(auto_output, exist_ok=True)

conditions = [
    "Original",
    "0.2Alpha_2000Neurons", "0.2Alpha_10000Neurons", "0.2Alpha_AllSigNeurons",
    "0.6Alpha_2000Neurons", "0.6Alpha_10000Neurons", "0.6Alpha_AllSigNeurons",
    "-0.6Alpha_2000Neurons", "-0.6Alpha_10000Neurons", "-0.6Alpha_AllSigNeurons",
]

score_vars = [
    "total_num",
    "correctness_ratio",
    "switches_ratio",
]
qc_vars = []
variables = score_vars
ratio_vars = ["correctness_ratio", "switches_ratio"]

var_titles = {
    "total_num": "Number of correct unique words",
    "correctness_ratio": "Ratio of correct unique words",
    "switches_ratio": "Semantic switch ratio",
}

task_titles = {
    "animal": "Animal fluency",
    "letter": "Letter fluency",
}

valid_row_status = {"yes", "y", "true", "1", "correct", "edit"}

# 3. functions
def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def deduplicate_by_latest(df):
    df = df.copy()
    if "updated_at" not in df.columns:
        return df.drop_duplicates(subset=["ID"], keep="last")

    df["_updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return (
        df.sort_values(["ID", "_updated_at"])
          .drop_duplicates(subset=["ID"], keep="last")
          .drop(columns="_updated_at")
    )


def parse_condition(cond):
    s = str(cond).strip()
    s_low = s.lower()

    if s_low == "original":
        alpha_signed = 0.0
    else:
        match = re.search(r"([+-]?\d*\.?\d+)\s*alpha", s_low)
        alpha_signed = float(match.group(1)) if match else np.nan

    if "allsig" in s_low:
        neurons = "AllSig"
    else:
        match = re.search(r"_([0-9]+)\s*neurons", s_low)
        neurons = match.group(1) if match else "0"

    return pd.Series({"alpha_signed": f"{alpha_signed}α", "neurons": str(neurons)})


def split_items(text):
    if pd.isna(text):
        return []

    items = re.split(r"\s*[,;\n]+\s*", str(text).strip())
    return [re.sub(r"\s+", " ", item).strip() for item in items if item.strip()]


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip().replace("_", " ").replace("-", " ")
    x = re.sub(r"^[^a-z0-9\u4e00-\u9fff]+", "", x)
    x = re.sub(r"[^a-z0-9\u4e00-\u9fff]+$", "", x)
    return re.sub(r"\s+", " ", x).strip()


def possible_singular_forms(x):
    forms = [x]
    if len(x) > 3 and x.endswith("ies"):
        forms.append(x[:-3] + "y")
    if len(x) > 3 and x.endswith("es"):
        forms.append(x[:-2])
    if len(x) > 2 and x.endswith("s"):
        forms.append(x[:-1])
    return list(dict.fromkeys(form for form in forms if form))


def load_animal_dict(path):
    animal_dict = pd.read_excel(path)

    if "is_animal" in animal_dict.columns:
        keep = animal_dict["is_animal"].astype(str).str.strip().str.lower()
        animal_dict = animal_dict[keep.isin({"true", "1", "1.0", "yes", "y"})]

    valid_animals = set()
    for col in ["Animals", "中文"]:
        if col in animal_dict.columns:
            valid_animals.update(
                normalize_text(value)
                for value in animal_dict[col].dropna().astype(str)
            )

    return {item for item in valid_animals if item}


def is_valid_status(x):
    if pd.isna(x):
        return False
    return str(x).strip().lower() in valid_row_status


def get_response_column(df):
    for col in ["bot_clean", "bot"]:
        if col in df.columns:
            return col
    raise ValueError("Missing response column: expected 'bot_clean' or 'bot'.")


def get_correct_column(df):
    candidates = [
        "Correct", "correct", "correctness", "Correctness",
        "is_correct", "IsCorrect", "valid", "Valid",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Missing manual correctness column.")


def animal_item_is_correct(item, valid_animals):
    key = normalize_text(item)
    return bool(key) and any(
        form in valid_animals for form in possible_singular_forms(key)
    )


def letter_item_is_correct(item, letter):
    key = normalize_text(item)
    return bool(key) and key.startswith(str(letter).lower())


def best_animal_synset(item):
    key = normalize_text(item).replace(" ", "_")
    synsets = wn.synsets(key, pos=wn.NOUN)

    if not synsets and "_" in key:
        synsets = wn.synsets(key.split("_")[-1], pos=wn.NOUN)
    if not synsets:
        return None

    animal = wn.synset("animal.n.01")

    def score(synset):
        similarity = synset.path_similarity(animal) or 0.0
        hypernyms = set(synset.closure(lambda node: node.hypernyms()))
        return similarity + (0.2 if animal in hypernyms else 0.0) + 0.01 * synset.min_depth()

    return max(synsets, key=score)


def first_noun_synset(item):
    key = normalize_text(item).replace(" ", "_")
    synsets = wn.synsets(key, pos=wn.NOUN)

    if not synsets and "_" in key:
        synsets = wn.synsets(key.split("_")[-1], pos=wn.NOUN)

    return synsets[0] if synsets else None


def wordnet_distance(a, b):
    if a is None or b is None:
        return np.nan
    distance = a.shortest_path_distance(b)
    return float(distance) if distance is not None else np.nan


def semantic_switch_ratio(items, task):
    if len(items) < 2:
        return 0.0

    if task == "animal":
        synsets = [best_animal_synset(item) for item in items]
    else:
        synsets = [first_noun_synset(item) for item in items]

    distances = np.array([
        wordnet_distance(synsets[i - 1], synsets[i])
        for i in range(1, len(synsets))
    ], dtype=float)
    distances = distances[~np.isnan(distances)]

    if len(distances) == 0:
        return 0.0

    threshold = np.median(distances)
    return float(np.mean(distances > threshold))



def compute_fluency_metrics(response, row_status, task, valid_animals=None, letter=None):
    if not is_valid_status(row_status):
        return pd.Series({var: 0.0 for var in score_vars + qc_vars})

    items = [item for item in split_items(response) if normalize_text(item)]
    if len(items) == 0:
        return pd.Series({var: 0.0 for var in score_vars + qc_vars})

    if task == "animal":
        correct_mask = [animal_item_is_correct(item, valid_animals) for item in items]
    elif task == "letter":
        correct_mask = [letter_item_is_correct(item, letter) for item in items]
    else:
        raise ValueError("task must be 'animal' or 'letter'.")

    correct_items = [item for item, correct in zip(items, correct_mask) if correct]
    correct_norm = [normalize_text(item) for item in correct_items]
    unique_correct_norm = list(dict.fromkeys(correct_norm))

    total_num = len(unique_correct_norm)
    correctness_ratio = total_num / len(items)
    switches_ratio = semantic_switch_ratio(unique_correct_norm, task)

    metrics = pd.Series({
        "total_num": int(total_num),
        "correctness_ratio": float(correctness_ratio),
        "switches_ratio": float(switches_ratio),
    })

    for var in ratio_vars:
        if not 0.0 <= metrics[var] <= 1.0:
            raise ValueError(f"{var} outside [0, 1]: {metrics[var]}")

    return metrics


def score_fluency_file(path, task, valid_animals=None, letter=None):
    df = deduplicate_by_latest(read_table(path))
    df["ID"] = df["ID"].astype(str)

    response_col = get_response_column(df)
    correct_col = get_correct_column(df)

    metrics = df.apply(
        lambda row: compute_fluency_metrics(
            response=row[response_col],
            row_status=row[correct_col],
            task=task,
            valid_animals=valid_animals,
            letter=letter,
        ),
        axis=1,
    )

    return pd.concat([df, metrics], axis=1)


def sig_label(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fit_gee(task_res, task):
    gee_rows = []

    for var in variables:
        dat = task_res[["Role", "Condition", var]].copy()
        dat[var] = pd.to_numeric(dat[var], errors="coerce")
        dat = dat.dropna(subset=["Role", "Condition", var])
        dat = dat[dat["Condition"].isin(conditions)].copy()
        dat["Condition"] = pd.Categorical(dat["Condition"].astype(str), categories=conditions)

        if dat.empty or dat["Condition"].nunique() < 2 or dat[var].nunique() < 2:
            continue

        formula = f'Q("{var}") ~ C(Role) + C(Condition, Treatment(reference="Original"))'

        if var == "total_num":
            family = sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
            family_note = "Tweedie"
        else:
            family = sm.families.Gaussian(link=sm.families.links.Identity())
            family_note = "Gaussian"

        result = smf.gee(
            formula,
            groups="Role",
            data=dat,
            family=family,
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()

        dispersion_ratio = (
            float(result.pearson_chi2) / float(result.df_resid)
            if result.df_resid > 0 else np.nan
        )

        for condition in conditions[1:]:
            term = f'C(Condition, Treatment(reference="Original"))[T.{condition}]'
            if term not in result.params.index:
                continue

            gee_rows.append({
                "task": task,
                "var": var,
                "condition": condition,
                "pearson_ratio": dispersion_ratio,
                "family": family_note,
                "beta": float(result.params[term]),
                "se": float(result.bse[term]),
                "z": float(result.tvalues[term]),
                "p": float(result.pvalues[term]),
                "n_obs": int(result.nobs),
            })

    gee_res = pd.DataFrame(gee_rows)

    if not gee_res.empty:
        gee_res["p_adj"] = np.nan
        for var in gee_res["var"].dropna().unique():
            idx = gee_res["var"] == var
            gee_res.loc[idx, "p_adj"] = pg.multicomp(
                gee_res.loc[idx, "p"].values,
                method="fdr_bh",
            )[1]

    return gee_res


# 4. commands
# 4.1 paths
ids_path = "Chat_analysis/annotations/ids.xlsx"
cat_fluency_path = "Chat_analysis/summary_screen/cat_fluency.xlsx"
let_fluency_path = "Chat_analysis/summary_screen/let_fluency.xlsx"
animal_dict_path = "Chat_analysis/animal_dict.xlsx"

# 4.2 fluency scores
valid_animals = load_animal_dict(animal_dict_path)

cat_scores = score_fluency_file(
    cat_fluency_path,
    task="animal",
    valid_animals=valid_animals,
)
cat_scores.to_csv(
    os.path.join(auto_output, "cat_fluency_auto.csv"),
    index=False,
    encoding="utf-8-sig",
)

let_scores = score_fluency_file(
    let_fluency_path,
    task="letter",
    letter="c",
)
let_scores.to_csv(
    os.path.join(auto_output, "let_fluency_auto.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 4.3 merge metadata
ids = pd.read_excel(ids_path)
ids["ID"] = ids["ID"].astype(str)
ids = ids.join(ids["Condition"].apply(parse_condition))
ids["Role"] = ids["Role"].astype(str)
ids["Condition"] = pd.Categorical(ids["Condition"].astype(str), categories=conditions)

cat_res = ids.merge(cat_scores, on="ID", how="left")
let_res = ids.merge(let_scores, on="ID", how="left")

cat_res.to_csv(
    os.path.join(fluency_output, "animal_fluency_with_meta.csv"),
    index=False,
    encoding="utf-8-sig",
)
let_res.to_csv(
    os.path.join(fluency_output, "letter_fluency_with_meta.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 4.4 GEE
cat_gee = fit_gee(cat_res, "animal")
let_gee = fit_gee(let_res, "letter")
gee_res = pd.concat([cat_gee, let_gee], ignore_index=True)

if not gee_res.empty:
    gee_res.to_csv(
        os.path.join(fluency_output, "gee_condition_vs_original.csv"),
        index=False,
        encoding="utf-8-sig",
    )

# 4.5 visualize
alpha_levels = ["-0.6α", "0.2α", "0.6α"]
neuron_levels = ["2000", "10000", "AllSig"]

x_order = ["Original", "-0.6α", "0.2α", "0.6α"]
x_base = {label: i for i, label in enumerate(x_order)}

dodge = {"2000": -0.18, "10000": 0.00, "AllSig": 0.18}
color_map = dict(zip(dodge.keys(), sns.color_palette("Set2", n_colors=3)))

fig, axes = plt.subplots(
    2,
    len(variables),
    figsize=(5.0 * len(variables), 9.2),
    squeeze=False,
)

plot_specs = [
    ("animal", cat_res, cat_gee),
    ("letter", let_res, let_gee),
]

for row_idx, (task, task_res, task_gee) in enumerate(plot_specs):
    for col_idx, var in enumerate(variables):
        ax = axes[row_idx, col_idx]
        plot_dat = task_res[["Condition", "alpha_signed", "neurons", var]].copy()
        plot_dat[var] = pd.to_numeric(plot_dat[var], errors="coerce")
        plot_dat = plot_dat.dropna(subset=[var, "Condition"])

        plot_dat["Condition"] = plot_dat["Condition"].astype(str)
        plot_dat["alpha_signed"] = plot_dat["alpha_signed"].astype(str)
        plot_dat["neurons"] = plot_dat["neurons"].astype(str)

        original = plot_dat[plot_dat["Condition"] == "Original"]
        original_mean = original[var].mean()
        original_se = original[var].std() / np.sqrt(original[var].count())

        summary = (
            plot_dat[plot_dat["Condition"] != "Original"]
            .groupby(["Condition", "alpha_signed", "neurons"], observed=True)[var]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["se"] = (summary["std"] / np.sqrt(summary["count"])).fillna(0)
        summary = summary[
            summary["alpha_signed"].isin(alpha_levels)
            & summary["neurons"].isin(neuron_levels)
        ].copy()

        summary["alpha_signed"] = pd.Categorical(
            summary["alpha_signed"],
            categories=alpha_levels,
            ordered=True,
        )
        summary["neurons"] = pd.Categorical(
            summary["neurons"],
            categories=neuron_levels,
            ordered=True,
        )
        summary = summary.sort_values(["neurons", "alpha_signed"])

        if not task_gee.empty:
            significance = task_gee[task_gee["var"] == var][["condition", "p_adj"]].copy()
            significance = significance.rename(columns={"condition": "Condition"})
            summary = summary.merge(significance, on="Condition", how="left")
            summary["sig"] = summary["p_adj"].apply(sig_label)
        else:
            summary["sig"] = ""

        summary["x"] = summary.apply(
            lambda row: x_base[str(row["alpha_signed"])] + dodge[str(row["neurons"])],
            axis=1,
        )

        y_values = list(summary["mean"].dropna()) + [original_mean]
        y_range = max(y_values) - min(y_values) if len(y_values) > 1 else 1.0
        if y_range == 0 or pd.isna(y_range):
            y_range = 1.0
        star_offset = 0.04 * y_range

        ax.axhline(original_mean, linestyle="--", linewidth=1, color="gray", alpha=0.8)
        ax.errorbar(
            x_base["Original"], original_mean, yerr=original_se,
            fmt="o", markersize=7, capsize=4,
            color="black", label="Original",
        )

        for neurons in neuron_levels:
            tmp = summary[summary["neurons"].astype(str) == neurons].copy()
            if tmp.empty:
                continue

            ax.plot(
                [x_base["Original"]] + list(tmp["x"]),
                [original_mean] + list(tmp["mean"]),
                marker="o", linewidth=1.8, markersize=5.5,
                color=color_map[neurons], label=f"{neurons} neurons",
            )
            ax.errorbar(
                tmp["x"], tmp["mean"], yerr=tmp["se"],
                fmt="none", capsize=4, color=color_map[neurons],
            )

            for _, row in tmp.iterrows():
                if row["sig"]:
                    ax.text(
                        row["x"], row["mean"] + row["se"] + star_offset,
                        row["sig"], ha="center", va="bottom", fontsize=11,
                    )

        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order)
        ax.set_xlabel("Condition")
        ax.set_ylabel(f"Mean {var}")
        ax.set_title(var_titles[var])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col_idx == 0:
            ax.text(
                -0.32, 0.5, task_titles[task],
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=14, fontweight="bold",
            )

handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels,
    loc="upper center", ncol=4, frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(
    os.path.join(fluency_output, "verbal_fluency_all_variables.svg"),
    bbox_inches="tight",
)