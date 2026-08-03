# 1. packages
import os
import re
import json
import ast
import warnings

import numpy as np
import pandas as pd
import pingouin as pg

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# 2. constants
task_name = "delayed_recall"
task_short = "delay"
recall_output = f"Chat_analysis/results/{task_name}"
os.makedirs(recall_output, exist_ok=True)

conditions = [
    "Original",
    "0.2Alpha_2000Neurons", "0.2Alpha_10000Neurons", "0.2Alpha_AllSigNeurons",
    "0.6Alpha_2000Neurons", "0.6Alpha_10000Neurons", "0.6Alpha_AllSigNeurons",
    "-0.6Alpha_2000Neurons", "-0.6Alpha_10000Neurons", "-0.6Alpha_AllSigNeurons",
]

# Immediate/delayed recall: only count mentioned entities and events.
score_vars = ["n_entities", "n_events"]

auto_vars = [
    "n_tokens",
    "lex_density",
    "mattr",
    "DEPID_R",
    "LexicalH",
    "clause_ratio",
]
variables = score_vars + auto_vars

var_titles = {
    "n_entities": "(A) Number of entities\n",
    "n_events": "(B) Number of events\n",
    "n_tokens": "(C) Number of tokens\n",
    "lex_density": "(D) Lexical density\n",
    "mattr": "(E) Moving-average type-token ratio\n",
    "DEPID_R": "(F) Dependency-based idea density without repetitions\n",
    "LexicalH": "(G) Lexical surprisal\n",
    "clause_ratio": "(H) Clause ratio\n",
}

# 3. functions
def read_table(path):
    if path is None:
        return None
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

def parse_choices(x):
    if pd.isna(x):
        return []

    if isinstance(x, dict):
        return x.get("choices", [])

    if isinstance(x, list):
        return x

    s = str(x).strip()
    if s in ["", "nan", "None", "null"]:
        return []

    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return []

    if isinstance(obj, dict):
        return obj.get("choices", [])
    if isinstance(obj, list):
        return obj
    return []

def count_choices(x):
    return len(parse_choices(x))

def extract_recall_scores(df, annotator):
    df = deduplicate_by_latest(df)

    missing = [c for c in ["entities", "events"] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns in recall annotation file: {missing}")

    out = pd.DataFrame({
        "ID": df["ID"].astype(str),
        f"{annotator}_n_entities": df["entities"].apply(count_choices),
        f"{annotator}_n_events": df["events"].apply(count_choices),
    })
    return out

def compute_agreement(annot, var):
    rater_1 = pd.to_numeric(
        annot[f"annotator_1_{var}"],
        errors="coerce"
    )
    rater_2 = pd.to_numeric(
        annot[f"annotator_2_{var}"],
        errors="coerce"
    )

    valid = rater_1.notna() & rater_2.notna()
    rater_1 = rater_1[valid]
    rater_2 = rater_2[valid]
    ids = annot.loc[valid, "ID"]

    # Mean absolute error between the two annotators
    mae = np.mean(np.abs(rater_1 - rater_2))

    tmp = pd.DataFrame({
        "ID": list(ids) * 2,
        "rater": (
            ["annotator_1"] * len(ids)
            + ["annotator_2"] * len(ids)
        ),
        "rating": list(rater_1) + list(rater_2),
    })

    icc = pg.intraclass_corr(
        data=tmp,
        targets="ID",
        raters="rater",
        ratings="rating"
    )

    icc.insert(0, "Var", var)
    icc.insert(1, "N", len(ids))
    icc.insert(2, "MAE", mae)

    return icc.copy()

def parse_condition(cond):
    s = str(cond).strip()
    s_low = s.lower()

    if s_low == "original":
        alpha_signed = 0.0
    else:
        m = re.search(r"([+-]?\d*\.?\d+)\s*alpha", s_low)
        alpha_signed = float(m.group(1)) if m else np.nan

    if "allsig" in s_low:
        neurons = "AllSig"
    else:
        m = re.search(r"_([0-9]+)\s*neurons", s_low)
        neurons = m.group(1) if m else "0"

    return pd.Series({"alpha_signed": f"{alpha_signed}α", "neurons": str(neurons)})

def sig_label(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
    
# 4. commands
# 4.1 paths
ids_path = "Chat_analysis/annotations/ids.xlsx"
annot_1_path = "Chat_analysis/annotator_1/delay.csv"
annot_2_path = "Chat_analysis/annotator_2/delay.csv"
auto_path = "Chat_analysis/automated_analysis/delayed_recall_ling.xlsx"

# 4.2 human scores and complete agreement report
annot_1 = read_table(annot_1_path)
score_1 = extract_recall_scores(annot_1, "annotator_1")

annot_2 = read_table(annot_2_path)
score_2 = extract_recall_scores(annot_2, "annotator_2")

annot = score_1.merge(score_2, on="ID", how="inner", validate="one_to_one")
if annot.empty:
    raise ValueError("No matched IDs were found between the two annotators.")

# MAE with ICC as a supplementary statistic
agreement_rows = []

for var in score_vars:
    agreement_rows.append(compute_agreement(annot, var))

agreement = pd.concat(agreement_rows, ignore_index=True)

icc2k = agreement[agreement["Type"] == "ICC2k"].copy()

icc2k.to_csv(
    os.path.join(recall_output, "agreement_icc2k_mae.csv"),
    index=False,
    encoding="utf-8-sig",
)

# Keep the original downstream scoring rule: average the two annotators.
for var in score_vars:
    annot[var] = annot[[f"annotator_1_{var}", f"annotator_2_{var}"]].mean(axis=1)

score_cols = ["ID"] + score_vars
scores = annot[score_cols].copy()

scores.to_csv(
    os.path.join(recall_output, "scores_avg_annotators.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 4.3 merge metadata and automatic metrics
ids = pd.read_excel(ids_path)
ids["ID"] = ids["ID"].astype(str)
ids = ids.join(ids["Condition"].apply(parse_condition))
ids["Role"] = ids["Role"].astype(str)
ids["Condition"] = pd.Categorical(ids["Condition"].astype(str), categories=conditions)
recall_res = ids.merge(scores, on="ID", how="left")

auto = read_table(auto_path)
auto["ID"] = auto["ID"].astype(str)
recall_res = recall_res.merge(auto, on="ID", how="left")

recall_res.to_csv(
    os.path.join(recall_output, "scores_with_auto_metrics.csv"),
    index=False,
    encoding="utf-8-sig",
)

variables = [v for v in variables if v in recall_res.columns]

# 4.4 GLM
gee_res = []
for var in variables:
    dat = recall_res.copy()
    dat[var] = pd.to_numeric(dat[var], errors="coerce")
    dat = dat.dropna(subset=["Role", "Condition", var])
    dat = dat[dat["Condition"].isin(conditions)].copy()
    dat["Condition"] = pd.Categorical(dat["Condition"].astype(str), categories=conditions)

    if dat.empty or dat["Condition"].nunique() < 2:
        continue

    formula = f'Q("{var}") ~ C(Role) + C(Condition, Treatment(reference="Original"))'
    if var in auto_vars and var != "n_tokens":
        formula += " + n_tokens"
    
    # define family 
    if var in ["n_entities", "n_events", "n_tokens"]:
        family=sm.families.Gamma(link=sm.families.links.Log())
        family_note = "Gamma"
    else:
        family=sm.families.Gaussian(link=sm.families.links.Identity())
        family_note = "Gaussian"

    res = smf.gee(formula, groups="Role", data=dat, family=family, cov_struct=sm.cov_struct.Exchangeable()).fit()    
    dispersion_ratio = float(res.pearson_chi2) / float(res.df_resid) if res.df_resid > 0 else np.nan

    rows = []
    for cond in conditions[1:]:
        term = f'C(Condition, Treatment(reference="Original"))[T.{cond}]'
        if term not in res.params.index:
            continue

        rows.append({
            "var": var,
            "condition": cond,
            "pearson_ratio": dispersion_ratio,
            "family": family_note,
            "beta": float(res.params[term]),
            "se": float(res.bse[term]),
            "z": float(res.tvalues[term]),
            "p": float(res.pvalues[term]),
            "n_obs": int(res.nobs),
        })
    gee_res.append(pd.DataFrame(rows))

gee_res = pd.concat(gee_res, ignore_index=True) if len(gee_res) > 0 else pd.DataFrame()
gee_res["p_adj"] = np.nan
for var in gee_res["var"].dropna().unique():
    idx = gee_res["var"] == var
    gee_res.loc[idx, "p_adj"] = pg.multicomp(gee_res.loc[idx, "p"].values, method="fdr_bh")[1]

gee_res.to_csv(
    os.path.join(recall_output, "gee_condition_vs_original.csv"),
    index=False,
    encoding="utf-8-sig",
)
    
# visualize
alpha_levels = ["-0.6α", "0.2α", "0.6α"]
neuron_levels = ["2000", "10000", "AllSig"]

x_order = ["Original", "-0.6α", "0.2α", "0.6α"]
x_base = {lab: i for i, lab in enumerate(x_order)}

dodge = {"2000": -0.18, "10000": 0.00, "AllSig": 0.18}
color_map = dict(zip(dodge.keys(), sns.color_palette("Set2", n_colors=3)))

n_cols = 3
n_rows = int(np.ceil(len(variables) / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(5.0 * n_cols, 4.6 * n_rows),
    squeeze=False,
)

axes_flat = axes.ravel()

for ax, var in zip(axes_flat, variables):
    plot_dat = recall_res[["Condition", "alpha_signed", "neurons", var]].copy()
    plot_dat[var] = pd.to_numeric(plot_dat[var], errors="coerce")
    plot_dat = plot_dat.dropna(subset=[var, "Condition"])

    plot_dat["Condition"] = plot_dat["Condition"].astype(str)
    plot_dat["alpha_signed"] = plot_dat["alpha_signed"].astype(str)
    plot_dat["neurons"] = plot_dat["neurons"].astype(str)

    orig_dat = plot_dat[plot_dat["Condition"] == "Original"]
    orig_mean = orig_dat[var].mean()
    orig_se = orig_dat[var].std() / np.sqrt(orig_dat[var].count())

    summary = (
        plot_dat[plot_dat["Condition"] != "Original"]
        .groupby(["Condition", "alpha_signed", "neurons"], observed=True)[var]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["se"] = summary["se"].fillna(0)

    summary = summary[
        summary["alpha_signed"].isin(["Original"] + alpha_levels)
        & summary["neurons"].isin(neuron_levels)
    ].copy()

    summary["alpha_signed"] = pd.Categorical(
        summary["alpha_signed"],
        categories=["Original"] + alpha_levels,
        ordered=True,
    )

    summary["neurons"] = pd.Categorical(
        summary["neurons"],
        categories=neuron_levels,
        ordered=True,
    )

    summary = summary.sort_values(["neurons", "alpha_signed"])

    sig_dat = gee_res[gee_res["var"] == var][["condition", "p_adj"]].copy()
    sig_dat = sig_dat.rename(columns={"condition": "Condition"})
    sig_dat["Condition"] = sig_dat["Condition"].astype(str)
    summary = summary.merge(sig_dat, on="Condition", how="left")
    summary["sig"] = summary["p_adj"].apply(sig_label)

    summary["x"] = summary.apply(
        lambda r: x_base[str(r["alpha_signed"])] + dodge[str(r["neurons"])],
        axis=1,
    )

    y_all = list(summary["mean"].dropna()) + [orig_mean]
    y_range = max(y_all) - min(y_all) if len(y_all) > 1 else 1
    if y_range == 0 or pd.isna(y_range):
        y_range = 1
    star_offset = 0.04 * y_range

    ax.axhline(orig_mean, linestyle="--", linewidth=1, color="gray", alpha=0.8)

    ax.errorbar(
        x_base["Original"], orig_mean, yerr=orig_se,
        fmt="o", markersize=7, capsize=4,
        color="black", label="Original",
    )

    for neu in neuron_levels:
        tmp = summary[summary["neurons"].astype(str) == neu].copy()
        if tmp.empty:
            continue

        x_line = [x_base["Original"]] + list(tmp["x"])
        y_line = [orig_mean] + list(tmp["mean"])

        ax.plot(
            x_line, y_line,
            marker="o", linewidth=1.8, markersize=5.5,
            color=color_map[neu], label=f"{neu} neurons",
        )

        ax.errorbar(
            tmp["x"], tmp["mean"], yerr=tmp["se"],
            fmt="none", capsize=4, color=color_map[neu],
        )

        for _, r in tmp.iterrows():
            if r["sig"] != "":
                ax.text(
                    r["x"], r["mean"] + r["se"] + star_offset,
                    r["sig"], ha="center", va="bottom", fontsize=11,
                )

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order)
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Mean {var}")
    ax.set_title(var_titles.get(var, var))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes_flat[len(variables):]:
    ax.axis("off")

handles, labels_legend = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels_legend,
    loc="upper center", ncol=4, frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    os.path.join(recall_output, "all_variables_raw_change_from_original.svg"),
    bbox_inches="tight",
)





















































