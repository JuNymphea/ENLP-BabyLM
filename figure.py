import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "errorbar.capsize": 1.5,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5
})


df = pd.read_csv("perplexity.csv")


model_map = {
    "finetune-random_vocab500-uniform_seed8": "random",
    "finetune-mixed-parens0.005_vocab500-uniform_seed8": "mixed",
    "finetune-nested-parens0.49_vocab500-uniform_seed8": "nested",
    "finetune-constituency_seed8": "constituency",
    "finetune-dependency_seed8": "dependency"
}


records = []
for prefix, label in model_map.items():
    steps = df[f"{prefix} - _step"]
    ppl = df[f"{prefix} - eval/perplexity"]
    ppl_min = df[f"{prefix} - eval/perplexity__MIN"]
    ppl_max = df[f"{prefix} - eval/perplexity__MAX"]

    for s, p, pmin, pmax in zip(steps, ppl, ppl_min, ppl_max):
        if 700 <= s <= 3800:
            records.append({
                "step": s,
                "perplexity": p,
                "perplexity_min": pmin,
                "perplexity_max": pmax,
                "model": label
            })

df_long = pd.DataFrame(records)


colors = {
    "random": "#FFB6C1",
    "mixed": "#D2B48C",
    "nested": "#D8BFD8",
    "constituency": "#ADD8E6",
    "dependency": "#90EE90"
}

markers = {
    "random": "o",
    "mixed": "v",
    "nested": "s",
    "constituency": "P",
    "dependency": "^"
}


plt.figure(figsize=(7, 5)) 

for model in df_long["model"].unique():
    sub = df_long[df_long["model"] == model]
    plt.errorbar(
        sub["step"],
        sub["perplexity"],
        yerr=[sub["perplexity"] - sub["perplexity_min"], sub["perplexity_max"] - sub["perplexity"]],
        label=model,
        color=colors[model],
        marker=markers[model],
        linestyle="-"
    )


plt.xlim(700, 3800)
plt.xticks(ticks=[700, 1200, 1700, 2200, 2700, 3200, 3800])


plt.ylabel("Perplexity")
plt.xlabel("Training Steps")
plt.legend(title="Experiments")
plt.tight_layout()


plt.savefig("perplexity", dpi=300)
plt.savefig("perplexity.pdf")
plt.show()