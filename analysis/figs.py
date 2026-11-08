import sys, os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
res, an, out = sys.argv[1], sys.argv[2], sys.argv[3]
import os; os.makedirs(out, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COL = {"16bit": "#2a78d6", "int8": "#eb6834", "nf4": "#1baf7a"}
MK = {"16bit": "o", "int8": "s", "nf4": "^"}
LAB = {"16bit": "16-bit", "int8": "8-bit (LLM.int8)", "nf4": "4-bit (NF4)"}
BEN = {"arc": "ARC-Challenge", "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}
SHORT = {"HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B", "Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-4B": "Qwen3-4B",
         "Qwen/Qwen3.5-4B": "Qwen3.5-4B", "ibm-granite/granite-4.1-3b": "Granite-4.1-3B",
         "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B", "microsoft/Phi-4-mini-instruct": "Phi-4-mini"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False})
def style(ax, xgrid=True):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------- Fig 1: change in position bias vs 16-bit, with 95% paired-bootstrap CIs
d = pd.read_csv(os.path.join(an, "delta_bias.csv")); d["m"] = d.model.map(SHORT)
order = d[(d.benchmark == "arc") & (d.quant == "nf4")].sort_values("acc_16").m.tolist()  # weakest -> strongest on ARC
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, b in zip(axs, BEN):
    for k, q in enumerate(["int8", "nf4"]):
        t = d[(d.benchmark == b) & (d.quant == q)].set_index("m").loc[order]
        y = np.arange(len(order)) + (-0.15 if q == "int8" else 0.15)
        ax.errorbar(t.delta, y, xerr=[t.delta - t.ci_low, t.ci_high - t.delta], fmt="none", ecolor=COL[q], elinewidth=2, capsize=0)
        sig = t.sig_holm.ne("n.s.").to_numpy()
        ax.scatter(t.delta[sig], y[sig], marker=MK[q], s=48, color=COL[q], edgecolor=COL[q], linewidth=1.6, zorder=3)
        ax.scatter(t.delta[~sig], y[~sig], marker=MK[q], s=38, color=SURF, edgecolor=COL[q], linewidth=1.6, zorder=3)
    ax.axvline(0, color=INK2, lw=1); ax.set_title(BEN[b], loc="left"); style(ax)
    ax.set_xlabel("Change in position bias vs 16-bit\n(RStd, accuracy points)")
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
from matplotlib.lines import Line2D
axs[0].legend(handles=[Line2D([], [], marker=MK["int8"], ls="", color=COL["int8"], label=LAB["int8"]),
                       Line2D([], [], marker=MK["nf4"], ls="", color=COL["nf4"], label=LAB["nf4"]),
                       Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=1.4, label="not significant (Holm)")],
              loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handletextpad=0.3)
save(fig, "fig1_bias_change")

# ---------- Fig 2: accuracy by position of the correct answer, two highlighted cases
pm = pd.read_csv(os.path.join(an, "per_model.csv"))
cases = [("mmlu_pro", "Qwen/Qwen3-1.7B"), ("arc", "HuggingFaceTB/SmolLM3-3B")]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.2))
for ax, (b, m) in zip(axs, cases):
    cols = [c for c in pm.columns if c.startswith("acc_correct_at_")]
    for q in ["16bit", "int8", "nf4"]:
        r = pm[(pm.benchmark == b) & (pm.model == m) & (pm.precision == q)].iloc[0]
        v = r[cols].dropna().to_numpy(float); x = np.arange(len(v))
        ax.plot(x, v, color=COL[q], lw=2, marker=MK[q], ms=6, mec=SURF, mew=1.2, label=LAB[q], zorder=3)
    n = len(v); ax.axhline(1 / n, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0, 1 / n + 0.012, "chance", va="bottom", ha="left", color=INK2, fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(list("ABCDEFGHIJ")[:n]); ax.set_ylim(0, None)
    ax.set_title(f"{SHORT[m]} on {BEN[b]}", loc="left"); style(ax, xgrid=False)
    ax.set_xlabel("Position of the correct answer"); ax.set_ylabel("Accuracy")
axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, handletextpad=0.3)
save(fig, "fig2_position_profiles")

# ---------- Fig 3: leaderboard stability (RQ2) and 16-bit vs quantized ranking agreement (RQ3)
st = pd.read_csv(os.path.join(an, "ranking_stability.csv")); cp = pd.read_csv(os.path.join(an, "cross_precision.csv"))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
ax = axs[0]
for k, q in enumerate(["16bit", "int8", "nf4"]):
    t = st[st.precision == q].set_index("benchmark").loc[list(BEN)]
    x = np.arange(3) + (k - 1) * 0.22
    ax.errorbar(x, t.mean_tau_original_vs_fixed_pos, yerr=[t.mean_tau_original_vs_fixed_pos - t.ci_low, t.ci_high - t.mean_tau_original_vs_fixed_pos],
                fmt="none", ecolor=COL[q], elinewidth=2)
    ax.scatter(x, t.mean_tau_original_vs_fixed_pos, marker=MK[q], s=42, color=COL[q], edgecolor=SURF, linewidth=1.5, zorder=3, label=LAB[q])
    ax.scatter(x, t.splithalf_tau, marker="_", s=160, color=INK2, linewidth=2, zorder=2, label="split-half noise level" if k == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels(BEN.values()); ax.set_ylim(-0.05, 1.05); style(ax, xgrid=False)
ax.set_ylabel("Kendall's τ"); ax.set_title("(a) Ranking: original order vs. answer moved", loc="left")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, handletextpad=0.3)
ax = axs[1]
SC = {"original_order": ("o", "Original answer order"), "avg_all_shifts": ("D", "Averaged over all shifts")}
labels = []
for i, (b, q) in enumerate([(b, q) for b in BEN for q in ["int8", "nf4"]]):
    for j, s in enumerate(SC):
        r = cp[(cp.benchmark == b) & (cp["compare"] == f"16bit_vs_{q}") & (cp.scoring == s)].iloc[0]
        x = i + (j - 0.5) * 0.3
        ax.errorbar(x, r.tau, yerr=[[r.tau - r.ci_low], [r.ci_high - r.tau]], fmt="none", ecolor=COL[q], elinewidth=2)
        ax.scatter(x, r.tau, marker=SC[s][0], s=40, color=COL[q] if s == "avg_all_shifts" else SURF, edgecolor=COL[q], linewidth=1.6, zorder=3)
    labels.append(f"{BEN[b].split('-')[0]}\n{'8-bit' if q == 'int8' else '4-bit'}")
ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1.05); style(ax, xgrid=False)
ax.set_ylabel("Kendall's τ vs 16-bit ranking"); ax.set_title("(b) Does quantization reorder the leaderboard?", loc="left")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=1.6, label=SC["original_order"][1]),
                   Line2D([], [], marker="D", ls="", mfc=INK2, mec=INK2, label=SC["avg_all_shifts"][1])],
          loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, handletextpad=0.3)
fig.subplots_adjust(wspace=0.28)
save(fig, "fig3_ranking")

# ---------- Fig A1 (appendix): heatmap of accuracy deviation by position, all models x precisions
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f08b89", "#b8312f"])
fig, axs = plt.subplots(1, 3, figsize=(11, 6), gridspec_kw={"width_ratios": [4, 4, 10]})
FULL = {v: k for k, v in SHORT.items()}
rows = [(FULL[m], q) for m in order for q in ["16bit", "int8", "nf4"]]
for ax, b in zip(axs, BEN):
    cols = [c for c in pm.columns if c.startswith("acc_correct_at_")]
    M = []
    for m, q in rows:
        r = pm[(pm.benchmark == b) & (pm.model == m) & (pm.precision == q)].iloc[0]
        v = r[cols].dropna().to_numpy(float); M.append(100 * (v - v.mean()))
    M = np.array(M)
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -35, 35), aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(list("ABCDEFGHIJ")[:M.shape[1]])
    ax.set_title(BEN[b], loc="left"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for i in range(1, len(order)): ax.axhline(3 * i - 0.5, color=SURF, lw=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{SHORT[m]}  {('16b', '8b', '4b')[['16bit', 'int8', 'nf4'].index(q)]}" for m, q in rows] if b == "arc" else [], fontsize=7.5)
cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.015); cb.outline.set_visible(False)
cb.set_label("Accuracy at this position minus the model's mean (points)", color=INK2)
save(fig, "figA1_position_heatmap")
print("ok")
