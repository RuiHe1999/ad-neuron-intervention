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

warnings.filterwarnings("ignore")

# 2. constants
task_name = "working_memory"
wm_output = f"Chat_analysis/results/{task_name}"
os.makedirs(wm_output, exist_ok=True)

conditions = [
    "Original",
    "0.2Alpha_2000Neurons", "0.2Alpha_10000Neurons", "0.2Alpha_AllSigNeurons",
    "0.6Alpha_2000Neurons", "0.6Alpha_10000Neurons", "0.6Alpha_AllSigNeurons",
    "-0.6Alpha_2000Neurons", "-0.6Alpha_10000Neurons", "-0.6Alpha_AllSigNeurons",
]

wm_files = {
    "dg_forward": "dg_forward.xlsx",
    "dg_backward": "dg_backward.xlsx",
    "dglt_forward": "dglt_forward.xlsx",
    "dglt_backward": "dglt_backward.xlsx",
}

wm_vars = list(wm_files.keys())
error_vars = [f"{var}_error" for var in wm_vars]
total_var = "total_error"
variables = error_vars + [total_var]

var_titles = {
    "dg_forward_error": "(A) Digit forward error",
    "dg_backward_error": "(B) Digit backward error",
    "dglt_forward_error": "(C) Digit-letter forward error\n",
    "dglt_backward_error": "(D) Digit-letter backward error\n",
    "total_error": "(E) Total errors",
}

# 3. functions
def read_table(path):
    if path is None:
        return None
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


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
wm_paths = {
    var: os.path.join("Chat_analysis/annotations", filename)
    for var, filename in wm_files.items()
}

# 4.2 scores
ids = pd.read_excel(ids_path)
ids["ID"] = ids["ID"].astype(str)
ids = ids.join(ids["Condition"].apply(parse_condition))
ids["Role"] = ids["Role"].astype(str)
ids["Condition"] = pd.Categorical(ids["Condition"].astype(str), categories=conditions)

wm_res = ids.copy()

for var, path in wm_paths.items():
    tmp = read_table(path)[["ID", "score"]].copy()
    tmp["ID"] = tmp["ID"].astype(str)
    tmp = tmp.rename(columns={"score": var})
    wm_res = wm_res.merge(tmp, on="ID", how="left")

wm_res[wm_vars] = wm_res[wm_vars].apply(pd.to_numeric, errors="coerce")

for var in wm_vars:
    wm_res[f"{var}_error"] = 1.0 - wm_res[var].clip(lower=0, upper=1)

wm_res[total_var] = wm_res[error_vars].sum(axis=1, min_count=1)

wm_res.to_csv(
    os.path.join(wm_output, "scores.csv"),
    index=False,
    encoding="utf-8-sig",
)

variables = [var for var in variables if var in wm_res.columns]

# 4.3 GEE
gee_res = []
for var in variables:
    dat = wm_res[["Role", "Condition", var]].copy()
    dat[var] = pd.to_numeric(dat[var], errors="coerce")
    dat = dat.dropna(subset=["Role", "Condition", var])
    dat = dat[dat["Condition"].isin(conditions)].copy()
    dat["Condition"] = pd.Categorical(dat["Condition"].astype(str), categories=conditions)

    if dat.empty or dat["Condition"].nunique() < 2:
        continue

    formula = f'Q("{var}") ~ C(Role) + C(Condition, Treatment(reference="Original"))'

    family = sm.families.Gaussian(link=sm.families.links.Identity())
    family_note = "Gaussian"

    res = smf.gee(
        formula,
        groups="Role",
        data=dat,
        family=family,
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()

    dispersion_ratio = (
        float(res.pearson_chi2) / float(res.df_resid)
        if res.df_resid > 0 else np.nan
    )

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

if not gee_res.empty:
    gee_res["p_adj"] = np.nan
    for var in gee_res["var"].dropna().unique():
        idx = gee_res["var"] == var
        gee_res.loc[idx, "p_adj"] = pg.multicomp(
            gee_res.loc[idx, "p"].values,
            method="fdr_bh",
        )[1]

    gee_res.to_csv(
        os.path.join(wm_output, "gee_condition_vs_original.csv"),
        index=False,
        encoding="utf-8-sig",
    )

# 4.4 visualize
alpha_levels = ["-0.6α", "0.2α", "0.6α"]
neuron_levels = ["2000", "10000", "AllSig"]

x_order = ["Original", "-0.6α", "0.2α", "0.6α"]
x_base = {lab: i for i, lab in enumerate(x_order)}

dodge = {"2000": -0.18, "10000": 0.00, "AllSig": 0.18}
color_map = dict(zip(dodge.keys(), sns.color_palette("Set2", n_colors=3)))

fig = plt.figure(figsize=(15.5, 8.8))
gs = fig.add_gridspec(
    nrows=2,
    ncols=3,
    width_ratios=[1, 1, 1.05],
    height_ratios=[1, 1],
    wspace=0.35,
    hspace=0.40,
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]
total_ax = fig.add_subplot(gs[:, 2])
plot_axes = axes + [total_ax]

for ax, var in zip(plot_axes, variables):
    plot_dat = wm_res[["Condition", "alpha_signed", "neurons", var]].copy()
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

    if not gee_res.empty:
        sig_dat = gee_res[gee_res["var"] == var][["condition", "p_adj"]].copy()
        sig_dat = sig_dat.rename(columns={"condition": "Condition"})
        sig_dat["Condition"] = sig_dat["Condition"].astype(str)
        summary = summary.merge(sig_dat, on="Condition", how="left")
        summary["sig"] = summary["p_adj"].apply(sig_label)
    else:
        summary["sig"] = ""

    summary["x"] = summary.apply(
        lambda row: x_base[str(row["alpha_signed"])] + dodge[str(row["neurons"])],
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

        for _, row in tmp.iterrows():
            if row["sig"] != "":
                ax.text(
                    row["x"], row["mean"] + row["se"] + star_offset,
                    row["sig"], ha="center", va="bottom", fontsize=11,
                )

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order)
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Mean {var}")
    ax.set_title(var_titles.get(var, var))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels_legend = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels_legend,
    loc="upper center", ncol=4, frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    os.path.join(wm_output, "working_memory_errors_total.svg"),
    bbox_inches="tight",
)