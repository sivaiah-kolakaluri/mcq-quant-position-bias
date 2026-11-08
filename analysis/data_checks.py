import sys, glob, os, json, numpy as np, pandas as pd
d = sys.argv[1]
rows=[]
for f in sorted(glob.glob(os.path.join(d,"*.csv"))):
    x = pd.read_csv(f)
    n_opt = x["correct_pos"].max()+1
    lp = x["logprobs"].map(json.loads)
    nan = lp.map(lambda v: any(np.isnan(v)) if isinstance(v,list) else True).sum()
    per_q = x.groupby("qid")["shift"].nunique()
    pred_share = x["pred_pos"].value_counts(normalize=True).max()
    # probability mass on the option letters (sanity: model actually answering with a letter)
    mass = lp.map(lambda v: float(np.exp(v).sum())).median()
    rows.append(dict(file=os.path.basename(f)[:-4], rows=len(x), q=x.qid.nunique(), n_opt=n_opt,
        incomplete_q=(per_q!=n_opt).sum(), dup=x.duplicated(["qid","shift"]).sum(), nan=nan,
        acc=round(x.correct.mean(),3), max_pred_share=round(pred_share,2), letter_mass=round(mass,2)))
t=pd.DataFrame(rows); t.to_csv(sys.argv[2], index=False) if len(sys.argv)>2 else None; pd.set_option("display.width",250); pd.set_option("display.max_rows",100)
print(t.to_string(index=False))
print("\nproblems:", t[(t.incomplete_q>0)|(t.dup>0)|(t.nan>0)].file.tolist())
