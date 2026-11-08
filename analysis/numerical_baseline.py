#!/usr/bin/env python3
"""
Numerical baseline: compares a repeated 16-bit run (batch size 8) with the original 16-bit run
(batch size 16) on identical prompts, and sets the result beside the 8-bit and NF4 changes.
All rates are tie-aware expected values, computed exactly as in analyze.py (independent uniform
tie-breaking in each run). 'self' compares the original 16-bit run with itself: it is not zero,
because a tied prompt can be broken differently in two runs; it is the floor any comparison has.
Usage: python analysis/numerical_baseline.py results/per_prompt results/rerun16_bs8 results/analysis
"""
import sys, os, glob, json
import numpy as np, pandas as pd

orig_dir, rerun_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

def load(f):
    x = pd.read_csv(f); x["qid"] = x.qid.astype(str)
    x = x.sort_values(["qid", "shift"]).reset_index(drop=True)
    L = np.array([json.loads(v) for v in x.logprobs]); T = (L == L.max(1, keepdims=True)).astype(float)
    P = T / T.sum(1, keepdims=True); cp = x.correct_pos.to_numpy(); sh = x["shift"].to_numpy()
    n = L.shape[1]; nq = x.qid.nunique()
    Pc = np.zeros_like(P)
    for j in range(n): Pc[np.arange(len(x)), (j + sh) % n] += P[:, j]
    cons = np.prod(Pc.reshape(nq, n, n), axis=1).sum(1)
    fc = P[np.arange(len(x)), cp]; A = np.zeros((nq, n)); A[np.repeat(np.arange(nq), n), cp] = fc
    return dict(x=x, L=L, T=T, P=P, fc=fc, cons=cons, rstd=100 * A.mean(0).std(), ties=(T.sum(1) > 1).mean())

def flip(a, b): return float((a["fc"] * (1 - b["fc"]) + b["fc"] * (1 - a["fc"])).mean())
def change(a, b): return float((1 - (a["P"] * b["P"]).sum(1)).mean())

rows = []
for f in sorted(glob.glob(os.path.join(rerun_dir, "*__16bit__*.csv"))):
    base = os.path.basename(f); stem = base.replace("__16bit__", "__{}__")
    o, r = load(os.path.join(orig_dir, base)), load(f)
    assert (o["x"][["qid", "shift", "correct_pos"]].values == r["x"][["qid", "shift", "correct_pos"]].values).all()
    q8, q4 = (load(os.path.join(orig_dir, stem.format(p))) for p in ("int8", "nf4"))
    d = np.abs(o["L"] - r["L"])
    rows.append(dict(
        benchmark=r["x"].benchmark[0], model=r["x"].model[0], prompts=len(d),
        identical_logprobs=float((d.max(1) == 0).mean()), median_abs_diff=float(np.median(d)),
        max_abs_diff=float(d.max()), top_set_changed=float((o["T"] != r["T"]).any(1).mean()),
        acc_orig=float(o["fc"].mean()), acc_rerun=float(r["fc"].mean()),
        rstd_orig=o["rstd"], rstd_rerun=r["rstd"], cons_orig=float(o["cons"].mean()), cons_rerun=float(r["cons"].mean()),
        tie_rate_orig=o["ties"], tie_rate_rerun=r["ties"],
        flip_self=flip(o, o), flip_rerun=flip(o, r), flip_int8=flip(o, q8), flip_nf4=flip(o, q4),
        change_self=change(o, o), change_rerun=change(o, r), change_int8=change(o, q8), change_nf4=change(o, q4)))
t = pd.DataFrame(rows).sort_values(["benchmark", "model"])
t.to_csv(f"{out}/numerical_baseline.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
c = ["benchmark", "model", "identical_logprobs", "top_set_changed", "acc_orig", "acc_rerun",
     "flip_self", "flip_rerun", "flip_int8", "flip_nf4", "change_rerun", "change_int8"]
print((t[c].assign(model=t.model.str.split("/").str[-1])).round(4).to_string(index=False))
print("\nmeans by benchmark (x100):")
print((100 * t.groupby("benchmark")[["flip_self", "flip_rerun", "flip_int8", "flip_nf4",
                                     "change_self", "change_rerun", "change_int8", "change_nf4",
                                     "top_set_changed", "identical_logprobs"]].mean()).round(2).to_string())

# ---- placebo test: the paper's paired bootstrap (20,000 resamples) applied to rerun vs original.
# A sound test should find (almost) nothing here, since the model and precision are the same.
import zlib
def two_sided_p(d):
    k = min((d <= 0).sum(), (d >= 0).sum()); return min(1.0, 2 * (k + 1) / (len(d) + 1))
prow = []
for f in sorted(glob.glob(os.path.join(rerun_dir, "*__16bit__*.csv"))):
    base = os.path.basename(f); o, r = load(os.path.join(orig_dir, base)), load(f)
    n = o["L"].shape[1]; nq = len(o["cons"]); cp = o["x"].correct_pos.to_numpy()
    def Amat(z):
        A = np.zeros((nq, n)); A[np.repeat(np.arange(nq), n), cp] = z["fc"]; return A
    Ao, Ar = Amat(o), Amat(r)
    rng = np.random.default_rng(zlib.crc32(("placebo|" + base).encode()))
    idx_all = rng.integers(0, nq, (20000, nq)); db, dc = [], []
    for s in range(0, 20000, 1000):
        idx = idx_all[s:s + 1000]
        db.append(100 * (Ar[idx].mean(1).std(1) - Ao[idx].mean(1).std(1)))
        dc.append(r["cons"][idx].mean(1) - o["cons"][idx].mean(1))
    db, dc = np.concatenate(db), np.concatenate(dc)
    prow.append(dict(benchmark=o["x"].benchmark[0], model=o["x"].model[0],
                     d_acc=100 * (r["fc"].mean() - o["fc"].mean()), d_rstd=r["rstd"] - o["rstd"], p_rstd=two_sided_p(db),
                     d_cons=100 * (r["cons"].mean() - o["cons"].mean()), p_cons=two_sided_p(dc)))
pt = pd.DataFrame(prow); pt.to_csv(f"{out}/numerical_baseline_placebo.csv", index=False)
print("\nplacebo (rerun vs original, same model and precision):")
print(pt.assign(model=pt.model.str.split("/").str[-1]).round(4).to_string(index=False))
print("max |d_acc| %.2f  max |d_rstd| %.2f  max |d_cons| %.2f  min p_rstd %.4f  min p_cons %.4f" %
      (pt.d_acc.abs().max(), pt.d_rstd.abs().max(), pt.d_cons.abs().max(), pt.p_rstd.min(), pt.p_cons.min()))
