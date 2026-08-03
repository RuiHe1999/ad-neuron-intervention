import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rbc = torch.load("qwen_8b/mwu_rbc.pt", map_location="cpu")
q = torch.load("qwen_8b/mwu_q.pt", map_location="cpu")

# AD-associated neurons: significant and more active in AD
eligible = (q < 0.05) & (-rbc > 0)

print("All significant AD-associated neurons:", eligible.sum().item())

# Rank eligible neurons globally by RBC
eligible_index = torch.where(eligible.flatten())[0]
eligible_rbc = rbc.flatten()[eligible_index]
ranked_index = eligible_index[torch.argsort(eligible_rbc, descending=True)]


# Create masks for three intervention scopes
top2000 = torch.zeros_like(eligible)
top2000.flatten()[ranked_index[:2000]] = True

top10000 = torch.zeros_like(eligible)
top10000.flatten()[ranked_index[:10000]] = True

masks = {
    "Top 2,000": top2000,
    "Top 10,000": top10000,
    "All significant": eligible
}


# Layer-wise distribution
rows = []

for scope, mask in masks.items():
    for layer in range(mask.shape[0]):
        rows.append({
            "layer": layer + 1,
            "scope": scope,
            "n_selected": mask[layer].sum().item(),
            "selected_percent": mask[layer].float().mean().item() * 100
        })

layer_df = pd.DataFrame(rows)

print(
    layer_df.pivot(
        index="layer",
        columns="scope",
        values="n_selected"
    )
)


# Visualization
sns.set_theme(style="white", context="paper", font_scale=1.1)

g = sns.relplot(
    data=layer_df,
    x="layer",
    y="selected_percent",
    col="scope",
    col_order=["Top 2,000", "Top 10,000", "All significant"],
    kind="line",
    marker="o",
    linewidth=1.8,
    height=3.2,
    aspect=1.05,
    facet_kws={"sharey": False}
)

g.set_axis_labels(
    "Transformer layer",
    "Selected neurons (%)"
)

g.set_titles("{col_name}")

for ax in g.axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])

plt.tight_layout()

plt.savefig(
    "qwen_8b/neuron_layer_distribution.pdf",
    bbox_inches="tight"
)

plt.show()