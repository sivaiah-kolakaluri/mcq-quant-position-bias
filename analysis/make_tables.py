# Generates the LaTeX tables in the paper directly from the result CSVs (no hand-typed numbers).
import sys, pandas as pd, numpy as np
from decimal import Decimal, ROUND_HALF_UP
def h(x, nd=1):
    return f"{Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP):.{nd}f}"
an, out = sys.argv[1], sys.argv[2]
import os; os.makedirs(out, exist_ok=True)
pm = pd.read_csv(f"{an}/per_model.csv"); cc = pd.read_csv(f"{an}/consistency_change.csv")
fl = pd.read_csv(f"{an}/flips_vs_16bit.csv"); dc = None
B = {"arc": "ARC-Challenge", "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}
P = {"16bit": "16-bit", "int8": "8-bit", "nf4": "4-bit NF4"}
SHORT = {"HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B", "Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-4B": "Qwen3-4B",
         "Qwen/Qwen3.5-4B": "Qwen3.5-4B", "ibm-granite/granite-4.1-3b": "Granite-4.1-3B",
         "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B", "microsoft/Phi-4-mini-instruct": "Phi-4-mini"}
ORDER = ["Qwen3-1.7B", "Llama-3.2-3B", "SmolLM3-3B", "Granite-4.1-3B", "Phi-4-mini", "Qwen3-4B", "Qwen3.5-4B"]
# per-model consistency per precision
cons = {(r.benchmark, r.model, r.precision): r.consistency for _, r in pm.iterrows()}
flips = {(r.benchmark, r.model, r.quant): r.flip_rate for _, r in fl.iterrows()}
# Table 2: summary (mean over 7 models)
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule",
     r"Benchmark & Precision & Accuracy (\%) & RStd $\downarrow$ & Consistency (\%) & Corr.\ flips (\%) \\", r"\midrule"]
for b in B:
    for i, p in enumerate(P):
        t = pm[(pm.benchmark == b) & (pm.precision == p)]
        acc = 100 * t.acc_all_shifts.mean(); rs = t.position_bias_rstd.mean()
        co = 100 * np.mean([cons[(b, m, p)] for m in t.model]); f = "--" if p == "16bit" else h(100*np.mean([flips[(b, m, p)] for m in t.model]))
        L.append(f"{B[b] if i == 0 else ''} & {P[p]} & {h(acc)} & {h(rs)} & {h(co)} & {f} \\\\")
    if b != "mmlu_pro": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Summary, averaged over the seven models. RStd is position bias in accuracy points (lower is better); consistency is the share of questions answered with the same option content under every rotation; correctness flips are the share of identical prompts whose correctness differs from the 16-bit model.}",
      r"\label{tab:summary}", r"\end{table}"]
open(f"{out}/tab_summary.tex", "w").write("\n".join(L) + "\n")
# Appendix table: per-model accuracy / RStd / consistency
L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{ll" + "rrr" * 3 + "}", r"\toprule",
     r" & & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{3}{c}{Position bias (RStd)} & \multicolumn{3}{c}{Consistency (\%)} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
     r"Benchmark & Model & 16 & 8 & 4 & 16 & 8 & 4 & 16 & 8 & 4 \\", r"\midrule"]
full = {v: k for k, v in SHORT.items()}
for b in B:
    for i, ms in enumerate(ORDER):
        m = full[ms]; row = []
        for metric in ["acc", "rstd", "cons"]:
            for p in P:
                r = pm[(pm.benchmark == b) & (pm.model == m) & (pm.precision == p)].iloc[0]
                v = 100 * r.acc_all_shifts if metric == "acc" else r.position_bias_rstd if metric == "rstd" else 100 * cons[(b, m, p)]
                row.append(h(v))
        L.append(f"{B[b] if i == 0 else ''} & {ms} & " + " & ".join(row) + r" \\")
    if b != "mmlu_pro": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\caption{Per-model results. Columns 16/8/4 are 16-bit, 8-bit (LLM.int8) and 4-bit NF4.}", r"\label{tab:permodel}", r"\end{table}"]
open(f"{out}/tab_permodel.tex", "w").write("\n".join(L) + "\n")
# Appendix: letter-probability mass and tie rate
g = pm.groupby("model").letter_mass_median.median(); tr = pm.groupby("model").tie_rate.mean()
L = [r"\begin{table}[h]", r"\centering", r"\small", r"\begin{tabular}{lrr}", r"\toprule", r"Model & Letter probability mass & Tied prompts (\%) \\", r"\midrule"]
for ms in ORDER: L.append(f"{ms} & {h(g[full[ms]], 2)} & {h(100 * tr[full[ms]])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\caption{Letter probability mass: median over the model's nine runs (3 benchmarks $\times$ 3 precisions) of each run's median share of next-token probability on the option letters. Tied prompts: mean share of prompts whose highest letter log-probability is shared by two or more letters.}", r"\label{tab:lettermass}", r"\end{table}"]
open(f"{out}/tab_lettermass.tex", "w").write("\n".join(L) + "\n")
print(open(f"{out}/tab_summary.tex").read())
