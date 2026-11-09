#!/usr/bin/env python3
"""Writes sections/numbers.tex: one LaTeX macro per number quoted in the text, computed from the
analysis_v2 tables. The paper text uses only these macros, so no number is typed by hand.
Rounding is round-half-up on the decimal representation (92.55 -> 92.6)."""
import sys
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd, numpy as np

an, out = sys.argv[1], sys.argv[2]
pm = pd.read_csv(f"{an}/per_model.csv"); db = pd.read_csv(f"{an}/delta_bias.csv")
cc = pd.read_csv(f"{an}/consistency_change.csv"); fl = pd.read_csv(f"{an}/flips_vs_16bit.csv")
st = pd.read_csv(f"{an}/ranking_stability.csv"); sd = pd.read_csv(f"{an}/stability_difference.csv")
cp = pd.read_csv(f"{an}/cross_precision.csv"); fx = pd.read_csv(f"{an}/calibration_fix.csv")
M = {}
def r(x, nd=1, sign=False):
    q = Decimal(1).scaleb(-nd)
    # round to 9 decimals first, so that e.g. 100 * 0.0055 = 0.5499999999999999 rounds half-up to 0.6
    v = Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP)
    if v == 0: v = abs(v)
    s = f"{v:.{nd}f}"
    if sign and v > 0: s = "+" + s
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")
def put(name, val): assert name not in M, name; M[name] = val
B = {"arc": "Arc", "gpqa": "Gpqa", "mmlu_pro": "Mmlu"}
Q = {"int8": "Int", "nf4": "Nf"}
SH = {"HuggingFaceTB/SmolLM3-3B": "Smol", "Qwen/Qwen3-1.7B": "QwenOne", "Qwen/Qwen3-4B": "QwenFour",
      "Qwen/Qwen3.5-4B": "QwenFive", "ibm-granite/granite-4.1-3b": "Granite",
      "meta-llama/Llama-3.2-3B-Instruct": "Llama", "microsoft/Phi-4-mini-instruct": "Phi"}

# dataset size and ties
nrows = (pm.n_questions * np.where(pm.benchmark == "mmlu_pro", 10, 4))
put("nPrompts", f"{int(nrows.sum()):,}".replace(",", "{,}")); put("nRuns", str(len(pm)))
put("TieRate", r(100 * (pm.tie_rate * nrows).sum() / nrows.sum()))
put("TieMin", r(100 * pm.tie_rate.min())); put("TieMax", r(100 * pm.tie_rate.max()))
# accuracy change
for q in Q:
    d = 100 * (db[db.quant == q].acc_q - db[db.quant == q].acc_16)
    put(f"Acc{Q[q]}Drop", r(-d.mean())); put(f"Acc{Q[q]}Mean", r(d.mean(), sign=True)); put(f"Acc{Q[q]}Min", r(d.min(), sign=True)); put(f"Acc{Q[q]}Max", r(d.max(), sign=True))
# mean accuracy / consistency / rstd per benchmark x precision (Table 2 values)
P = {"16bit": "Sixteen", "int8": "Int", "nf4": "Nf"}
for b in B:
    for p in P:
        t = pm[(pm.benchmark == b) & (pm.precision == p)]
        put(f"Acc{B[b]}{P[p]}", r(100 * t.acc_all_shifts.mean())); put(f"Cons{B[b]}{P[p]}", r(100 * t.consistency.mean()))
        put(f"Rstd{B[b]}{P[p]}", r(t.position_bias_rstd.mean()))
# flips
f = fl.groupby(["benchmark", "quant"]).flip_rate.mean().unstack()
for b in B:
    for q in Q: put(f"Flip{Q[q]}{B[b]}", r(100 * f.loc[b, q]))
ratio = f.nf4 / f.int8; put("FlipRatioMin", r(ratio.min())); put("FlipRatioMax", r(ratio.max()))
# significance counts
for name, t in [("Bias", db), ("Cons", cc)]:
    for q in Q:
        s = t[t.quant == q].sig_holm
        put(f"{name}{Q[q]}Up", str((s == "up").sum())); put(f"{name}{Q[q]}Down", str((s == "down").sum()))
        put(f"{name}{Q[q]}NS", str((s == "n.s.").sum())); put(f"{name}{Q[q]}Sig", str((s != "n.s.").sum()))
put("ConsNfPointDown", str((cc[cc.quant == "nf4"].delta < 0).sum()))
# individual changes (every model x benchmark x quant)
for _, x in db.iterrows(): put(f"DBias{SH[x.model]}{B[x.benchmark]}{Q[x.quant]}", r(x.delta, sign=True))
for _, x in cc.iterrows(): put(f"DCons{SH[x.model]}{B[x.benchmark]}{Q[x.quant]}", r(100 * x.delta, sign=True))
for _, x in db.iterrows(): put(f"PHolmBias{SH[x.model]}{B[x.benchmark]}{Q[x.quant]}", r(x.p_holm, 3))
for _, x in cc.iterrows(): put(f"PHolmCons{SH[x.model]}{B[x.benchmark]}{Q[x.quant]}", r(x.p_holm, 3))
for _, x in cc[cc.quant == "nf4"].iterrows():
    put(f"ConsVal{SH[x.model]}{B[x.benchmark]}Sixteen", r(100 * x.cons_16)); put(f"ConsVal{SH[x.model]}{B[x.benchmark]}Nf", r(100 * x.cons_q))
# per-model accuracy, choice share, position accuracy
for _, x in pm.iterrows():
    k = f"{SH[x.model]}{B[x.benchmark]}{P[x.precision]}"
    put(f"AccVal{k}", r(100 * x.acc_all_shifts)); put(f"ShareA{k}", r(100 * x.choice_share_A))
    for L in "ABCD": put(f"Pos{L}{k}", r(100 * x[f"acc_correct_at_{L}"]))
q = pm[(pm.model == "Qwen/Qwen3-1.7B") & (pm.benchmark == "mmlu_pro") & (pm.precision == "16bit")].iloc[0]
shares = {c[-1]: q[c] for c in pm.columns if c.startswith("choice_share_") and q[c] == q[c]}
top = max(shares, key=shares.get); put("QwenOneMmluTopLetter", top); put("QwenOneMmluTopShare", r(100 * shares[top]))
g = pm[(pm.benchmark == "gpqa") & (pm.precision == "16bit")].acc_all_shifts
put("GpqaAccMin", r(100 * g.min())); put("GpqaAccMax", r(100 * g.max()))
# rankings
for _, x in st.iterrows():
    k = f"{B[x.benchmark]}{P[x.precision]}"
    put(f"Stab{k}", r(x.mean_tau_original_vs_fixed_pos, 2)); put(f"StabLo{k}", r(x.ci_low, 2)); put(f"StabHi{k}", r(x.ci_high, 2))
    put(f"Split{k}", r(x.splithalf_tau, 2))
for _, x in sd.iterrows():
    k = f"{B[x.benchmark]}{Q[x['compare'].split('_')[0]]}"
    put(f"DStab{k}", r(x.delta, 2, sign=True)); put(f"DStabLo{k}", r(x.ci_low, 2)); put(f"DStabHi{k}", r(x.ci_high, 2)); put(f"DStabP{k}", r(x.p, 2))
put("DStabPMin", r(sd.p.min(), 2))
for _, x in cp.iterrows():
    k = f"{B[x.benchmark]}{Q[x['compare'].split('_vs_')[1]]}{'Orig' if x.scoring == 'original_order' else 'Avg'}"
    put(f"Cross{k}", r(x.tau, 2))
# prior correction (means over models, held-out 95%)
gx = fx.groupby(["benchmark", "precision"]).mean(numeric_only=True)
for b in B:
    for p in P:
        x = gx.loc[(b, p)]; k = f"{B[b]}{P[p]}"
        put(f"FixRstd{k}", r(x.rstd)); put(f"FixRstdAfter{k}", r(x.rstd_fixed)); put(f"FixAcc{k}", r(100 * x.acc))
        put(f"FixAccAfter{k}", r(100 * x.acc_fixed)); put(f"FixCons{k}", r(100 * x.consistency)); put(f"FixConsAfter{k}", r(100 * x.consistency_fixed))
for _, x in fx.iterrows():
    k = f"{SH[x.model]}{B[x.benchmark]}{P[x.precision]}"
    put(f"FixRstdM{k}", r(x.rstd)); put(f"FixRstdAfterM{k}", r(x.rstd_fixed)); put(f"FixAccM{k}", r(100 * x.acc)); put(f"FixAccAfterM{k}", r(100 * x.acc_fixed))
# letter mass
lm = pm.groupby("model").letter_mass_median.median()
put("MassSmol", r(100 * lm["HuggingFaceTB/SmolLM3-3B"], 0)); put("MassQwenFive", r(100 * lm["Qwen/Qwen3.5-4B"], 0))
others = lm.drop(["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3.5-4B"])
put("MassOthersMin", r(100 * others.min(), 0)); put("MassOthersMax", r(100 * others.max(), 0))

# numerical baseline (repeated 16-bit run at batch size 8), if available
import os
if os.path.exists(f"{an}/numerical_baseline.csv"):
    nb = pd.read_csv(f"{an}/numerical_baseline.csv"); pl = pd.read_csv(f"{an}/numerical_baseline_placebo.csv")
    g = nb.groupby("benchmark")
    for bb, x in g:
        put(f"NbFlip{B[bb]}", r(100 * x.flip_rerun.mean())); put(f"NbSelf{B[bb]}", r(100 * x.flip_self.mean()))
        put(f"NbExcess{B[bb]}", r(100 * (x.flip_rerun - x.flip_self).mean()))
        put(f"NbFlipMin{B[bb]}", r(100 * x.flip_rerun.min())); put(f"NbFlipMax{B[bb]}", r(100 * x.flip_rerun.max()))
    ratio = nb.flip_int8 / nb.flip_rerun
    put("NbRatioMin", r(ratio.min())); put("NbRatioMax", r(ratio.max()))
    put("NbPairs", str(len(nb))); put("NbIdentMin", r(100 * nb.identical_logprobs.min(), 0)); put("NbIdentMax", r(100 * nb.identical_logprobs.max(), 0))
    put("NbAccMax", r(pl.d_acc.abs().max(), 2)); put("NbRstdMax", r(pl.d_rstd.abs().max())); put("NbConsMax", r(pl.d_cons.abs().max()))
    put("NbPMin", r(min(pl.p_rstd.min(), pl.p_cons.min()), 2))
    # which pairs have the lowest 8-bit / rerun flip ratio, and what they share
    ties16 = pm[pm.precision == "16bit"][["benchmark", "model", "tie_rate"]]
    z = nb.assign(ratio=nb.flip_int8 / nb.flip_rerun).merge(ties16, on=["benchmark", "model"])
    low = z.nsmallest(5, "ratio")
    put("NbLowN", "five"); put("NbLowRatioMax", r(low.ratio.max()))
    put("NbLowGpqa", {1:"one",2:"two",3:"three",4:"four",5:"five"}[int((low.benchmark == "gpqa").sum())])
    put("NbLowTieMin", r(100 * low.tie_rate.min())); put("NbLowTieMax", r(100 * low.tie_rate.max()))
    fl8 = fl[fl.quant == "int8"].merge(pl, on=["benchmark", "model"])
    n_exceed = int(((100 * (fl8.acc_q - fl8.acc_16)).abs() > pl.d_acc.abs().max()).sum())
    put("NbAccExceed", str(n_exceed)); put("NbAccWithin", str(len(nb) - n_exceed))
    put("NbRhoAcc", r(z.ratio.corr(z.acc_orig, method="spearman"), 2))

# answer-change rates (share of prompts whose chosen option changes vs 16-bit), from numerical_baseline.csv
if os.path.exists(f"{an}/numerical_baseline.csv"):
    g = pd.read_csv(f"{an}/numerical_baseline.csv").groupby("benchmark")
    for bb, x in g:
        put(f"ChgInt{B[bb]}", r(100 * x.change_int8.mean())); put(f"ChgNf{B[bb]}", r(100 * x.change_nf4.mean()))
        put(f"ChgRerun{B[bb]}", r(100 * x.change_rerun.mean()))
# robustness: the main conclusions without the two models with low letter probability mass
low = ["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen3.5-4B"]
c5, d5 = cc[~cc.model.isin(low)], db[~db.model.isin(low)]
put("RobPairs", str(int((c5.quant == "nf4").sum())))
put("RobNfConsLower", str(int((c5[c5.quant == "nf4"].delta < 0).sum())))
put("RobNfConsDown", str(int((c5[c5.quant == "nf4"].sig_holm == "down").sum())))
put("RobNfConsUp", str(int((c5[c5.quant == "nf4"].sig_holm == "up").sum())))
put("RobNfBiasUp", str(int((d5[d5.quant == "nf4"].sig_holm == "up").sum())))
put("RobNfBiasDown", str(int((d5[d5.quant == "nf4"].sig_holm == "down").sum())))
put("RobIntConsSig", str(int((c5[c5.quant == "int8"].sig_holm != "n.s.").sum())))
put("RobIntBiasSig", str(int((d5[d5.quant == "int8"].sig_holm != "n.s.").sum())))
if os.path.exists(f"{an}/numerical_baseline.csv"):
    n5 = pd.read_csv(f"{an}/numerical_baseline.csv"); n5 = n5[~n5.model.isin(low)]
    put("RobIntOverRerun", str(int((n5.flip_int8 > n5.flip_rerun).sum())))
    rat = n5.groupby("benchmark").flip_nf4.mean() / n5.groupby("benchmark").flip_int8.mean()
    put("RobRatioMin", r(rat.min())); put("RobRatioMax", r(rat.max()))
# tie-policy check (first tied letter instead of uniform tie-breaking)
if os.path.exists(f"{an}/tie_policy_check.csv"):
    tp = pd.read_csv(f"{an}/tie_policy_check.csv"); tn, ti = tp[tp.quant == "nf4"], tp[tp.quant == "int8"]
    put("TieNfSame", str(int(min(tn.same_sign_acc.sum(), tn.same_sign_rstd.sum(), tn.same_sign_cons.sum()))))
    disc = pd.concat([ti[~ti["same_sign_" + k]][["d_" + k + "_uniform", "d_" + k + "_first"]].abs().max(axis=1) for k in ("acc", "rstd", "cons")])
    put("TieIntDisc", str(len(disc))); put("TieIntDiscMax", r(disc.max(), 2))

with open(out, "w") as fh:
    fh.write("% Auto-generated by gen_numbers.py from the analysis tables. Do not edit by hand.\n")
    for k, v in M.items():
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
