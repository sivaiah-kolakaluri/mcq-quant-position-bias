#!/usr/bin/env python3
"""
Analysis v2: every metric recomputed from the raw per-prompt results with
  * TIE-AWARE scoring: when several option letters share the maximum log-probability
    (exact ties in half precision), each tied letter is chosen with equal probability
    and every metric is its exact expected value under that uniform tie-breaking;
  * 20,000 paired bootstrap resamples for change tests, p = 2*(min tail + 1)/(B + 1),
    Holm over the 42 tests of each metric;
  * a paired bootstrap test of the change in ranking stability, and a split-half
    noise baseline for Kendall's tau.
Usage: python analysis_v2.py <results_dir> <out_dir>
"""
import sys, os, glob, json, zlib
import numpy as np, pandas as pd
from scipy.stats import kendalltau

res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
B_TEST, B_CI, B_STAB, N_SPLIT = 20000, 2000, 2000, 1000
LET = "ABCDEFGHIJ"; PRECS = ["16bit", "int8", "nf4"]; QUANTS = ["int8", "nf4"]

def seed(*parts): return zlib.crc32("|".join(parts).encode())
def tau(a, b):
    t = kendalltau(a, b).statistic
    return float(t) if t == t else np.nan
def two_sided_p(d):
    k = min((d <= 0).sum(), (d >= 0).sum())
    return min(1.0, 2 * (k + 1) / (len(d) + 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

# ------------------------------------------------------------------ load + tie-aware quantities
R = {}
for f in sorted(glob.glob(os.path.join(res, "*.csv"))):
    x = pd.read_csv(f); x["qid"] = x.qid.astype(str)
    x = x.sort_values(["qid", "shift"]).reset_index(drop=True)
    b, m, p = x.benchmark[0], x.model[0], x.precision[0]
    n = int(x.correct_pos.max()) + 1; qids = x.qid.unique(); nq = len(qids)
    assert len(x) == nq * n and (x.groupby("qid")["shift"].nunique() == n).all()
    L = np.array([json.loads(v) for v in x.logprobs])                    # (N, n)
    T = (L == L.max(1, keepdims=True)).astype(float)                      # tied maxima
    P = T / T.sum(1, keepdims=True)                                       # expected choice over positions
    cp = x.correct_pos.to_numpy(); sh = x["shift"].to_numpy()
    fc = P[np.arange(len(x)), cp]                                         # expected correctness per prompt
    # per question: A[q, k] = expected correctness when the answer is at position k
    A = np.zeros((nq, n)); A[np.repeat(np.arange(nq), n), cp] = fc
    # expected choice over option CONTENT (original option index), per question and shift
    Pc = np.zeros_like(P)
    for j in range(n): Pc[np.arange(len(x)), (j + sh) % n] += P[:, j]
    Pc = Pc.reshape(nq, n, n)                                             # (q, shift, content)
    cons_q = np.prod(Pc, axis=1).sum(1)                                   # P(same content under every shift)
    O = fc.reshape(nq, n)[:, 0]                                           # original order (shift 0)
    R[(b, m, p)] = dict(n=n, qids=qids, A=A, fc=fc.reshape(nq, n), P=P.reshape(nq, n, n), O=O,
                        cons=cons_q, L=L, cp=cp, sh=sh, ties=(T.sum(1) > 1).mean(),
                        mass=np.median(np.exp(L).sum(1)), x=x)
keys = sorted({(b, m) for b, m, _ in R})
for b, m in keys:                                                         # identical question sets
    for p in QUANTS: assert (R[(b, m, p)]["qids"] == R[(b, m, "16bit")]["qids"]).all()

# ------------------------------------------------------------------ per model
rows = []
for (b, m, p), r in sorted(R.items()):
    pos = r["A"].mean(0)
    row = dict(benchmark=b, precision=p, model=m, n_questions=len(r["qids"]), acc_original=r["O"].mean(),
               acc_all_shifts=r["A"].mean(), position_bias_rstd=100 * pos.std(), consistency=r["cons"].mean(),
               tie_rate=r["ties"], letter_mass_median=r["mass"])
    share = r["P"].reshape(-1, r["n"]).mean(0)                            # expected share of each letter chosen
    row.update({f"acc_correct_at_{LET[k]}": v for k, v in enumerate(pos)})
    row.update({f"choice_share_{LET[k]}": v for k, v in enumerate(share)})
    rows.append(row)
pm = pd.DataFrame(rows); pm.to_csv(f"{out}/per_model.csv", index=False)

# ------------------------------------------------------------------ change tests vs 16-bit (paired, B = 20,000)
bias_rows, cons_rows, flip_rows = [], [], []
for b, m in keys:
    r16 = R[(b, m, "16bit")]; nq = len(r16["qids"]); rng = np.random.default_rng(seed("test", b, m))
    idx_all = rng.integers(0, nq, (B_TEST, nq))
    for p in QUANTS:
        rq = R[(b, m, p)]
        db, dc = [], []
        for s in range(0, B_TEST, 1000):
            idx = idx_all[s:s + 1000]
            db.append(100 * (rq["A"][idx].mean(1).std(1) - r16["A"][idx].mean(1).std(1)))
            dc.append(rq["cons"][idx].mean(1) - r16["cons"][idx].mean(1))
        db, dc = np.concatenate(db), np.concatenate(dc)
        rs16, rsq = 100 * r16["A"].mean(0).std(), 100 * rq["A"].mean(0).std()
        bias_rows.append(dict(benchmark=b, model=m, quant=p, rstd_16=rs16, rstd_q=rsq, delta=rsq - rs16,
                              ci_low=np.percentile(db, 2.5), ci_high=np.percentile(db, 97.5), p=two_sided_p(db),
                              acc_16=r16["A"].mean(), acc_q=rq["A"].mean()))
        c16, cq = r16["cons"].mean(), rq["cons"].mean()
        cons_rows.append(dict(benchmark=b, model=m, quant=p, cons_16=c16, cons_q=cq, delta=cq - c16,
                              ci_low=np.percentile(dc, 2.5), ci_high=np.percentile(dc, 97.5), p=two_sided_p(dc)))
        # expected flips on identical prompts (independent tie-breaking in the two runs)
        a, q = r16["fc"], rq["fc"]
        flip_q = (a * (1 - q) + q * (1 - a)).mean(1)
        chg_q = (1 - (r16["P"] * rq["P"]).sum(2)).mean(1)
        fb = flip_q[idx_all[:B_CI]].mean(1)
        flip_rows.append(dict(benchmark=b, model=m, quant=p, acc_16=a.mean(), acc_q=q.mean(), flip_rate=flip_q.mean(),
                              flip_ci_low=np.percentile(fb, 2.5), flip_ci_high=np.percentile(fb, 97.5),
                              answer_change_rate=chg_q.mean()))
for rows_, name in [(bias_rows, "delta_bias"), (cons_rows, "consistency_change")]:
    t = pd.DataFrame(rows_); t["p_holm"] = holm(t.p.values)
    t["sig_holm"] = np.where(t.p_holm < 0.05, np.where(t.delta > 0, "up", "down"), "n.s.")
    t.to_csv(f"{out}/{name}.csv", index=False)
pd.DataFrame(flip_rows).to_csv(f"{out}/flips_vs_16bit.csv", index=False)

# ------------------------------------------------------------------ rankings
def stab(C, O):  # mean tau between original-order ranking and each fixed-position ranking
    return np.nanmean([tau(O.mean(1), C[:, :, k].mean(1)) for k in range(C.shape[2])])
st_rows, sd_rows, cp_rows = [], [], []
for b in sorted({k[0] for k in R}):
    models = sorted({m for bb, m, _ in R if bb == b}); nq = len(R[(b, models[0], "16bit")]["qids"])
    C = {p: np.stack([R[(b, m, p)]["A"] for m in models]) for p in PRECS}
    O = {p: np.stack([R[(b, m, p)]["O"] for m in models]) for p in PRECS}
    rng = np.random.default_rng(seed("rank", b)); idx_all = rng.integers(0, nq, (B_STAB, nq))
    boot = {p: np.array([stab(C[p][:, i], O[p][:, i]) for i in idx_all]) for p in PRECS}
    for p in PRECS:
        # split-half noise baseline: tau between rankings on two random halves (original order)
        r2 = np.random.default_rng(seed("split", b, p)); sh_t = []
        for _ in range(N_SPLIT):
            perm = r2.permutation(nq); h1, h2 = perm[: nq // 2], perm[nq // 2:]
            sh_t.append(tau(O[p][:, h1].mean(1), O[p][:, h2].mean(1)))
        st_rows.append(dict(benchmark=b, precision=p, n_models=len(models), mean_tau_original_vs_fixed_pos=stab(C[p], O[p]),
                            ci_low=np.percentile(boot[p], 2.5), ci_high=np.percentile(boot[p], 97.5),
                            splithalf_tau=np.nanmean(sh_t)))
    for q in QUANTS:
        d = boot[q] - boot["16bit"]
        sd_rows.append(dict(benchmark=b, compare=f"{q}_minus_16bit", delta=stab(C[q], O[q]) - stab(C["16bit"], O["16bit"]),
                            ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), p=two_sided_p(d)))
        for name, f in [("original_order", lambda i: tau(O["16bit"][:, i].mean(1), O[q][:, i].mean(1))),
                        ("avg_all_shifts", lambda i: tau(C["16bit"][:, i].mean((1, 2)), C[q][:, i].mean((1, 2))))]:
            bt = np.array([f(i) for i in idx_all])
            cp_rows.append(dict(benchmark=b, compare=f"16bit_vs_{q}", scoring=name, n_models=len(models),
                                tau=f(np.arange(nq)), ci_low=np.nanpercentile(bt, 2.5), ci_high=np.nanpercentile(bt, 97.5)))
pd.DataFrame(st_rows).to_csv(f"{out}/ranking_stability.csv", index=False)
pd.DataFrame(sd_rows).to_csv(f"{out}/stability_difference.csv", index=False)
pd.DataFrame(cp_rows).to_csv(f"{out}/cross_precision.csv", index=False)

# ------------------------------------------------------------------ prior correction (simplified PriDe), tie-aware
fix_rows = []
for b, m in keys:
    qids = R[(b, m, "16bit")]["qids"]; nq = len(qids); n = R[(b, m, "16bit")]["n"]
    r_ = np.random.default_rng(seed("fix", b, m))
    splits = [r_.choice(nq, max(5, int(0.05 * nq)), replace=False) for _ in range(20)]
    for p in PRECS:
        r = R[(b, m, p)]
        Ln = (r["L"] - np.logaddexp.reduce(r["L"], axis=1, keepdims=True)).reshape(nq, n, n)
        cp = r["cp"].reshape(nq, n); sh = r["sh"].reshape(nq, n); vals = []
        for cal in splits:
            test = np.ones(nq, bool); test[cal] = False
            prior = Ln[cal].reshape(-1, n).mean(0)
            def metrics(Z):
                # ties within 1e-9: exact-arithmetic ties must not depend on platform-specific float noise
                T = (Z >= Z.max(2, keepdims=True) - 1e-9).astype(float); P = T / T.sum(2, keepdims=True)
                fc = np.take_along_axis(P, cp[..., None], 2)[..., 0]           # (q, shift)
                A = np.zeros((nq, n)); A[np.arange(nq)[:, None], cp] = fc
                Pc = np.zeros_like(P)
                for j in range(n): Pc[np.arange(nq)[:, None], np.arange(n)[None, :], (j + sh) % n] += P[:, :, j]
                cons = np.prod(Pc, axis=1).sum(1)
                return A[test].mean(), 100 * A[test].mean(0).std(), cons[test].mean()
            vals.append(metrics(Ln) + metrics(Ln - prior))
        a = np.array(vals).mean(0)
        fix_rows.append(dict(benchmark=b, model=m, precision=p, acc=a[0], rstd=a[1], consistency=a[2],
                             acc_fixed=a[3], rstd_fixed=a[4], consistency_fixed=a[5]))
pd.DataFrame(fix_rows).to_csv(f"{out}/calibration_fix.csv", index=False)
print("done")
