# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Aggregate revision runs into tables (csv + LaTeX rows) with mean, std and paired tests
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); TDIR = os.path.join(OUT, "tables"); os.makedirs(TDIR, exist_ok=True)
MET = ["Accuracy", "Precision", "Recall", "F1", "MacroF1", "MCC", "ROC_AUC"]

def jl(path): return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

def ms(v, d=4): return f"{np.mean(v):.{d}f} $\\pm$ {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else f"{np.mean(v):.{d}f}"

def main_table(ds="X-IIoTID"):
    R = pd.DataFrame(jl(os.path.join(OUT, "stationary", ds, "main_runs.jsonl")))
    order = ["LogReg", "RandomForest", "XGBoost", "MLP", "GCN", "GraphSAGE", "GAT", "E-GraphSAGE", "GPS", "TransformerConv", "AGT"]
    order = [o for o in order if o in set(R.model)]
    agt = R[R.model == "AGT"].set_index("seed")
    rows, pv = [], []
    for m in order:
        s = R[R.model == m].set_index("seed")
        row = dict(Model=m, n=len(s), **{k: s[k].mean() for k in MET}, **{k + "_sd": s[k].std(ddof=1) for k in MET},
                   train_s=s["train_s"].mean())
        if m != "AGT":
            com = agt.index.intersection(s.index)
            try: p = wilcoxon(agt.loc[com, "MacroF1"], s.loc[com, "MacroF1"]).pvalue
            except Exception: p = 1.0
            row["p_macroF1_vs_AGT"] = p; row["diff_macroF1"] = (agt.loc[com, "MacroF1"] - s.loc[com, "MacroF1"]).mean()
            try: p2 = wilcoxon(agt.loc[com, "F1"], s.loc[com, "F1"]).pvalue
            except Exception: p2 = 1.0
            row["p_F1_vs_AGT"] = p2
        rows.append(row)
    T = pd.DataFrame(rows)
    mask = T.Model != "AGT"
    T.loc[mask, "p_macroF1_holm"] = holm(T.loc[mask, "p_macroF1_vs_AGT"].values)
    T.loc[mask, "p_F1_holm"] = holm(T.loc[mask, "p_F1_vs_AGT"].values)
    T.to_csv(os.path.join(TDIR, f"main_{ds}.csv"), index=False)
    # pooled per-class scores over all seeds for AGT and XGBoost
    pc = {}
    for m in ["AGT", "XGBoost", "MLP", "GAT"]:
        s = R[R.model == m]
        if not len(s): continue
        yt = np.concatenate(s.ytrue.values); yp = np.concatenate(s.pred.values); C = len(s.classes.iloc[0])
        pc[m] = dict(classes=s.classes.iloc[0],
                     precision=precision_score(yt, yp, average=None, labels=range(C), zero_division=0).tolist(),
                     recall=recall_score(yt, yp, average=None, labels=range(C), zero_division=0).tolist(),
                     f1=f1_score(yt, yp, average=None, labels=range(C), zero_division=0).tolist(),
                     f1_seed_sd=np.std(np.vstack(s.perclass.values), 0, ddof=1).tolist(),
                     support=np.bincount(yt, minlength=C).tolist(), cm=confusion_matrix(yt, yp, labels=range(C)).tolist())
    json.dump(pc, open(os.path.join(TDIR, f"perclass_{ds}.json"), "w"), indent=1)
    print(T[["Model", "n", "Accuracy", "F1", "MacroF1", "MacroF1_sd", "MCC", "p_macroF1_holm", "p_F1_holm"]].round(4).to_string(index=False))
    return T

def ablation():
    R = pd.DataFrame(jl(os.path.join(OUT, "stationary", "X-IIoTID", "ablation_runs.jsonl")))
    full = R[R.variant == "Full AGT"].set_index("seed"); rows = []
    for v in R.variant.unique():
        s = R[R.variant == v].set_index("seed"); com = full.index.intersection(s.index)
        row = dict(Variant=v, n=len(s), params=int(s.params.iloc[0]), **{k: s[k].mean() for k in ["Accuracy", "F1", "MacroF1", "MCC"]},
                   **{k + "_sd": s[k].std(ddof=1) for k in ["Accuracy", "F1", "MacroF1", "MCC"]})
        if v != "Full AGT":
            row["dMacroF1"] = (s.loc[com, "MacroF1"] - full.loc[com, "MacroF1"]).mean()
            row["dF1"] = (s.loc[com, "F1"] - full.loc[com, "F1"]).mean()
            try: row["p"] = wilcoxon(full.loc[com, "MacroF1"], s.loc[com, "MacroF1"]).pvalue
            except Exception: row["p"] = 1.0
            try: row["pF1"] = wilcoxon(full.loc[com, "F1"], s.loc[com, "F1"]).pvalue
            except Exception: row["pF1"] = 1.0
        rows.append(row)
    T = pd.DataFrame(rows); m = T.Variant != "Full AGT"
    T.loc[m, "p_holm"] = holm(T.loc[m, "p"].values); T.loc[m, "pF1_holm"] = holm(T.loc[m, "pF1"].values)
    T.to_csv(os.path.join(TDIR, "ablation.csv"), index=False); print(T.round(4).to_string(index=False)); return T

def sens():
    R = pd.DataFrame(jl(os.path.join(OUT, "stationary", "X-IIoTID", "sensitivity_runs.jsonl")))
    T = R.groupby(["param", "value"]).agg(n=("seed", "count"), val=("val_macroF1", "mean"), val_sd=("val_macroF1", "std"),
                                          F1=("F1", "mean"), F1_sd=("F1", "std"), MacroF1=("MacroF1", "mean"),
                                          MacroF1_sd=("MacroF1", "std"), train_s=("train_s", "mean")).reset_index()
    T.to_csv(os.path.join(TDIR, "sensitivity.csv"), index=False); print(T.round(4).to_string(index=False)); return T

def noise():
    R = jl(os.path.join(OUT, "stationary", "X-IIoTID", "noise_runs.jsonl")); rows = []
    for r in R:
        for m, v in r["res"].items(): rows.append(dict(seed=r["seed"], sigma=r["sigma"], model=m, F1=v["F1"], MacroF1=v["MacroF1"]))
    T = pd.DataFrame(rows).groupby(["model", "sigma"]).agg(F1=("F1", "mean"), F1_sd=("F1", "std"), MacroF1=("MacroF1", "mean"),
                                                         MacroF1_sd=("MacroF1", "std"), n=("seed", "count")).reset_index()
    T.to_csv(os.path.join(TDIR, "noise.csv"), index=False); print(T.round(3).to_string(index=False))
    pc = [dict(sigma=r["sigma"], seed=r["seed"], pc=r["res"]["AGT"]["perclass"]) for r in R]
    cls = R[0]["classes"]; P = pd.DataFrame([dict(sigma=p["sigma"], **dict(zip(cls, p["pc"]))) for p in pc]).groupby("sigma").mean()
    P.to_csv(os.path.join(TDIR, "noise_perclass_agt.csv")); return T

if __name__ == "__main__":
    for st in sys.argv[1:]:
        if st.startswith("main"): main_table(st.split(":")[1] if ":" in st else "X-IIoTID")
        else: globals()[st]()
