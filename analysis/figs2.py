import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
an, out = sys.argv[1], sys.argv[2]
import os; os.makedirs(out, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL = {"16bit": "#2a78d6", "int8": "#eb6834", "nf4": "#1baf7a"}; MK = {"16bit": "o", "int8": "s", "nf4": "^"}
LAB = {"16bit": "16-bit", "int8": "8-bit (LLM.int8)", "nf4": "4-bit (NF4)"}
BEN = {"arc": "ARC-Challenge", "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}
SHORT = {"HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B", "Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-4B": "Qwen3-4B",
         "Qwen/Qwen3.5-4B": "Qwen3.5-4B", "ibm-granite/granite-4.1-3b": "Granite-4.1-3B",
         "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B", "microsoft/Phi-4-mini-instruct": "Phi-4-mini"}
ORDER = ["Qwen3-1.7B", "Llama-3.2-3B", "SmolLM3-3B", "Granite-4.1-3B", "Phi-4-mini", "Qwen3-4B", "Qwen3.5-4B"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})
def style(ax, xgrid=True):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

# Fig 4: change in answer consistency vs 16-bit (percentage points), 95% paired-bootstrap CI
c = pd.read_csv(os.path.join(an, "consistency_change.csv")); c["m"] = c.model.map(SHORT)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, b in zip(axs, BEN):
    for q in ["int8", "nf4"]:
        t = c[(c.benchmark == b) & (c.quant == q)].set_index("m").loc[ORDER]
        y = np.arange(len(ORDER)) + (-0.15 if q == "int8" else 0.15)
        d, lo, hi = 100 * t.delta, 100 * t.ci_low, 100 * t.ci_high
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="none", ecolor=COL[q], elinewidth=2)
        filled = t.sig_holm.ne("n.s.").to_numpy()
        ax.scatter(d[filled], y[filled], marker=MK[q], s=48, color=COL[q], edgecolor=COL[q], linewidth=1.6, zorder=3)
        ax.scatter(d[~filled], y[~filled], marker=MK[q], s=38, color=SURF, edgecolor=COL[q], linewidth=1.6, zorder=3)
    ax.axvline(0, color=INK2, lw=1); ax.set_title(BEN[b], loc="left"); style(ax)
    ax.set_xlabel("Change in answer consistency vs 16-bit\n(percentage points)")
axs[0].set_yticks(range(len(ORDER))); axs[0].set_yticklabels(ORDER); axs[0].invert_yaxis()
axs[0].legend(handles=[Line2D([], [], marker=MK["int8"], ls="", color=COL["int8"], label=LAB["int8"]),
                       Line2D([], [], marker=MK["nf4"], ls="", color=COL["nf4"], label=LAB["nf4"]),
                       Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=1.4, label="not significant (Holm)")],
              loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handletextpad=0.3)
save(fig, "fig4_consistency_change")

# Fig 5: (a) flips vs 16-bit; (b) prior-correction fix: bias before -> after
f = pd.read_csv(os.path.join(an, "flips_vs_16bit.csv")); f["m"] = f.model.map(SHORT)
x = pd.read_csv(os.path.join(an, "calibration_fix.csv"))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axs[0]; rng = np.random.default_rng(1)
for i, b in enumerate(BEN):
    for k, q in enumerate(["int8", "nf4"]):
        v = 100 * f[(f.benchmark == b) & (f.quant == q)].flip_rate.to_numpy()
        xs = i + (k - 0.5) * 0.32 + rng.uniform(-0.06, 0.06, len(v))
        ax.scatter(xs, v, marker=MK[q], s=26, color=COL[q], edgecolor=SURF, linewidth=1, zorder=3, alpha=0.9)
        ax.plot([i + (k - 0.5) * 0.32 - 0.12, i + (k - 0.5) * 0.32 + 0.12], [v.mean()] * 2, color=INK, lw=2, zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels(BEN.values()); ax.set_ylim(0, None); style(ax, xgrid=False)
ax.set_ylabel("Answers flipped vs 16-bit (%)"); ax.set_title("(a) Correctness flips on identical prompts", loc="left")
ax.legend(handles=[Line2D([], [], marker=MK[q], ls="", color=COL[q], label=LAB[q]) for q in ["int8", "nf4"]] +
          [Line2D([], [], color=INK, lw=2, label="mean of 7 models")], loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handletextpad=0.3)
ax = axs[1]
g = x.groupby(["benchmark", "precision"])[["rstd", "rstd_fixed"]].mean()
for i, b in enumerate(BEN):
    for k, q in enumerate(["16bit", "int8", "nf4"]):
        xx = i + (k - 1) * 0.25; r0, r1 = g.loc[(b, q)]
        ax.plot([xx, xx], [r0, r1], color=COL[q], lw=2, zorder=2)
        ax.scatter([xx], [r0], marker=MK[q], s=44, color=SURF, edgecolor=COL[q], linewidth=1.6, zorder=3)
        ax.scatter([xx], [r1], marker=MK[q], s=44, color=COL[q], edgecolor=COL[q], linewidth=1.6, zorder=3)
ax.set_xticks(range(3)); ax.set_xticklabels(BEN.values()); ax.set_ylim(0, None); style(ax, xgrid=False)
ax.set_ylabel("Position bias, RStd (mean of 7 models)"); ax.set_title("(b) Prior correction: bias before (hollow) → after (filled)", loc="left")
ax.legend(handles=[Line2D([], [], marker=MK[q], ls="", color=COL[q], label=LAB[q]) for q in ["16bit", "int8", "nf4"]],
          loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handletextpad=0.3)
fig.subplots_adjust(wspace=0.25)
save(fig, "fig5_flips_and_fix")
print("ok")
