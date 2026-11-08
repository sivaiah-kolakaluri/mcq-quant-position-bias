# Infers the number format of the logits from the spacing of stored letter log-probabilities.
# bfloat16 logits of size 16-64 are spaced 2^-4 to 2^-2 apart (float16: 2^-6 to 2^-5).
# Usage: python analysis/dtype_check.py results/per_prompt
import sys, glob, os, json, numpy as np, pandas as pd
rows=[]
for f in sorted(glob.glob(sys.argv[1]+"/*.csv")):
    d=pd.read_csv(f, usecols=["model","precision","benchmark","logprobs"]); d=d.sample(min(1500,len(d)), random_state=0)
    ks=[]
    for s in d.logprobs:
        L=np.array(json.loads(s)); x=L-L.max()
        # smallest k (0..12) such that all diffs are multiples of 2^-k (within rounding of 5 decimals)
        for k in range(13):
            y=x*2**k
            if np.all(np.abs(y-np.round(y))<3e-5*2**k): break
        else: k=13
        ks.append(k)
    ks=np.array(ks)
    rows.append(dict(model=d.model.iloc[0].split("/")[-1], prec=d.precision.iloc[0], bench=d.benchmark.iloc[0],
                     med_k=np.median(ks), share_k_le4=(ks<=4).mean().round(3), share_k_ge8=(ks>=8).mean().round(3)))
t=pd.DataFrame(rows); pd.set_option("display.width",200); pd.set_option("display.max_rows",100)
print(t.to_string(index=False))
print(t.groupby("prec")[["med_k","share_k_le4","share_k_ge8"]].agg(["min","max"]))
