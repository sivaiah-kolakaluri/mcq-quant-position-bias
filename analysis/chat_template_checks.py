#!/usr/bin/env python3
"""
Pre-registered chat-template checks (docs/PREREGISTRATION_chat_template.md).
Compares the chat-template set (analysed on its own with analyze.py) with the plain-prompt results
for SmolLM3-3B and Qwen3.5-4B, and writes one row per model x benchmark with the four checks:
 (a) 8-bit: consistency change not significant and |RStd change| < 1 point
 (b) NF4: consistency lower than at 16-bit (direction); Holm-significant decreases reported
 (c) NF4 correctness-flip rate > 8-bit correctness-flip rate
 (d) direction of the NF4 RStd change, compared with the plain-prompt result
Usage: python analysis/chat_template_checks.py results/chat_template results/analysis_chat results/analysis
"""
import sys, os, json, glob
import numpy as np, pandas as pd

chat_dir, chat_an, plain_an = sys.argv[1], sys.argv[2], sys.argv[3]
cc, db, fl = (pd.read_csv(f"{chat_an}/{n}.csv") for n in ("consistency_change", "delta_bias", "flips_vs_16bit"))
dbp = pd.read_csv(f"{plain_an}/delta_bias.csv"); pmp = pd.read_csv(f"{plain_an}/per_model.csv")
pm = pd.read_csv(f"{chat_an}/per_model.csv")
rows = []
for (b, m), _ in cc.groupby(["benchmark", "model"]):
    g = lambda d, q: d[(d.benchmark == b) & (d.model == m) & (d.quant == q)].iloc[0]
    c8, c4, d8, d4 = g(cc, "int8"), g(cc, "nf4"), g(db, "int8"), g(db, "nf4")
    f8, f4, p4 = g(fl, "int8"), g(fl, "nf4"), g(dbp, "nf4")
    mass = lambda d: d[(d.benchmark == b) & (d.model == m)].letter_mass_median
    rows.append(dict(
        benchmark=b, model=m,
        mass_plain_min=mass(pmp).min(), mass_chat_min=mass(pm).min(), mass_chat_max=mass(pm).max(),
        acc16_chat=g(fl, "int8").acc_16,
        a_int8_cons_delta=100 * c8.delta, a_int8_cons_sig=c8.sig_holm, a_int8_rstd_delta=d8.delta,
        a_holds=(c8.sig_holm == "n.s.") and abs(d8.delta) < 1,
        b_nf4_cons_delta=100 * c4.delta, b_nf4_cons_p_holm=c4.p_holm, b_nf4_cons_sig=c4.sig_holm,
        b_holds=c4.delta < 0,
        c_flip_int8=100 * f8.flip_rate, c_flip_nf4=100 * f4.flip_rate,
        c_change_int8=100 * f8.answer_change_rate, c_change_nf4=100 * f4.answer_change_rate,
        c_holds=f4.flip_rate > f8.flip_rate,
        d_rstd_nf4_chat=d4.delta, d_sig_chat=d4.sig_holm, d_rstd_nf4_plain=p4.delta, d_sig_plain=p4.sig_holm,
        d_same_direction=np.sign(d4.delta) == np.sign(p4.delta)))
t = pd.DataFrame(rows).sort_values(["model", "benchmark"])
t.to_csv(f"{plain_an}/chat_template_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(t.assign(model=t.model.str.split("/").str[-1]).round(3).to_string(index=False))
print("\nholds: (a) %d/6  (b) %d/6  (c) %d/6  (d) same NF4 RStd direction %d/6" %
      (t.a_holds.sum(), t.b_holds.sum(), t.c_holds.sum(), t.d_same_direction.sum()))
