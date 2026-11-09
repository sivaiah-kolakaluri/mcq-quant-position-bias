#!/usr/bin/env python3
"""
Tie-policy check: recomputes the changes vs 16-bit with the alternative tie rule "always pick the
first (lowest) tied letter" instead of the paper's uniform random tie-breaking (exact expected
values), and compares the direction of each change. Note that the first-letter rule itself adds a
preference for early positions, which is why the paper does not use it.
Usage: python analysis/tie_policy_check.py results/per_prompt results/analysis
"""
import sys, os, glob, json
import numpy as np, pandas as pd

res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def metrics(f, rule):
    x = pd.read_csv(f); x["qid"] = x.qid.astype(str)
    x = x.sort_values(["qid", "shift"]).reset_index(drop=True)
    L = np.array([json.loads(v) for v in x.logprobs]); n = L.shape[1]; nq = len(L) // n
    if rule == "uniform":
        T = (L == L.max(1, keepdims=True)).astype(float); P = T / T.sum(1, keepdims=True)
    else:  # first tied letter
        P = np.zeros_like(L); P[np.arange(len(L)), L.argmax(1)] = 1.0
    cp = x.correct_pos.to_numpy(); sh = x["shift"].to_numpy()
    fc = P[np.arange(len(L)), cp]
    A = np.zeros((nq, n)); A[np.repeat(np.arange(nq), n), cp] = fc
    Pc = np.zeros_like(P)
    for j in range(n): Pc[np.arange(len(L)), (j + sh) % n] += P[:, j]
    cons = np.prod(Pc.reshape(nq, n, n), axis=1).sum(1).mean()
    return dict(acc=fc.mean(), rstd=100 * A.mean(0).std(), cons=cons, x=x)


rows = []
for f16 in sorted(glob.glob(os.path.join(res, "*__16bit__*.csv"))):
    for q in ("int8", "nf4"):
        fq = f16.replace("__16bit__", f"__{q}__")
        r = {}
        for rule in ("uniform", "first"):
            a, b = metrics(f16, rule), metrics(fq, rule)
            for k in ("acc", "rstd", "cons"):
                r[f"d_{k}_{rule}"] = (100 if k != "rstd" else 1) * (b[k] - a[k])
        x = a["x"]
        rows.append(dict(benchmark=x.benchmark[0], model=x.model[0], quant=q, **r))
t = pd.DataFrame(rows)
for k in ("acc", "rstd", "cons"):
    t[f"same_sign_{k}"] = np.sign(t[f"d_{k}_uniform"]) == np.sign(t[f"d_{k}_first"])
t.to_csv(f"{out}/tie_policy_check.csv", index=False)
for q in ("int8", "nf4"):
    s = t[t.quant == q]
    print(q, "same direction: acc %d/21, RStd %d/21, consistency %d/21" %
          (s.same_sign_acc.sum(), s.same_sign_rstd.sum(), s.same_sign_cons.sum()),
          "| max |difference| between rules: acc %.2f, RStd %.2f, cons %.2f points" %
          tuple((s[f"d_{k}_uniform"] - s[f"d_{k}_first"]).abs().max() for k in ("acc", "rstd", "cons")))
