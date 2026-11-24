# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Revision figures in plain journal style
import os, json, sys, json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {"XGB-retrain": "#1f4e79", "XGB-sliding": "#4f81bd", "XGB-static": "#9dc3e6", "AGT-rehearsal": "#c00000",
       "AGT-static": "#f4a6a6", "AGT-retrain": "#7f0000", "MLP-rehearsal": "#548235", "MLP-static": "#a9d18e",
       "INSOMNIA-style": "#7f6000", "AGT-finetune": "#e46c0a", "XGB-continued": "#7030a0"}

def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def fig_drift_composition():
    m = json.load(open(os.path.join(OUT, "drift", "windows_meta.json"))); cls = m["classes"]
    rows = [m["init_counts"]] + [w["counts"] for w in m["windows"]]
    P = np.array(rows, float); P = P / P.sum(1, keepdims=True)
    labels = ["Init"] + [str(w["window"]) for w in m["windows"]]
    cmap = plt.get_cmap("tab10"); fig, ax = plt.subplots(figsize=(7.2, 3.0)); bot = np.zeros(len(rows))
    for j, c in enumerate(cls):
        ax.bar(labels, P[:, j], bottom=bot, color=cmap(j), width=0.85, label=c.replace("Lateral _movement", "Lateral movement"))
        bot += P[:, j]
    ax.set_ylabel("Class share"); ax.set_xlabel("Chronological window (Init = initial training period)")
    ax.set_ylim(0, 1); ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(os.path.join(FIG, "fig10_drift_composition.png")); plt.close(fig)

def load_drift(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "drift", pat))):
        d = json.load(open(f))
        for m in d["rows"][0]:
            if m in ("seed", "window"): continue
            out.setdefault(m, []).append([r[m]["F1"] for r in d["rows"]])
    return {k: np.array(v) for k, v in out.items()}

def fig_drift_trajectory():
    base = load_drift("multi_rho0.2_M2500_eps0.0_seed*.json")
    # same settings as Table 13: each adaptive method with its own tuned update settings
    for pat in ["multi_*_tuned_seed*.json", "multi_*_tunedmlp_seed*.json", "multi_*_tunedxgb_seed*.json"]:
        for k, v in load_drift(pat).items(): base[k] = v
    show = ["XGB-retrain", "XGB-sliding", "AGT-rehearsal", "MLP-rehearsal", "XGB-static", "AGT-static"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(1, base["AGT-static"].shape[1] + 1)
    for m in show:
        if m not in base: continue
        mu, sd = base[m].mean(0), base[m].std(0, ddof=1)
        ls = "--" if "static" in m else "-"
        ax.plot(x, mu, ls, color=COL[m], lw=1.6, marker="o", ms=3, label=m)
        ax.fill_between(x, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
    ax.set_xlabel("Chronological window"); ax.set_ylabel("Weighted F1 (prequential)"); ax.set_ylim(0, 1.02); ax.set_xticks(x)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(FIG, "fig11_drift_trajectory.png")); plt.close(fig)

def fig_cost_accuracy():
    rows = []
    for pat in ["multi_rho0.2_M2500_eps0.0_seed*.json", "multi_*_tuned_seed*.json", "multi_*_tunedmlp_seed*.json"]:
        for f in glob.glob(os.path.join(OUT, "drift", pat)):
            d = json.load(open(f)); tag = "tuned" if "tuned" in f else ""
            for m, t in d["timing"].items():
                c = np.mean([x.get("total_s", x.get("graph_s", 0) + x.get("steps_s", 0) + x.get("buffer_s", 0)) for x in t])
                mf = np.mean([r[m]["MacroF1"] for r in d["rows"]])
                rows.append(dict(m=m, tag=tag, cost=c, macro=mf))
    T = pd.DataFrame(rows)
    T = T[~((T.m.isin(["AGT-rehearsal", "MLP-rehearsal"])) & (T.tag == ""))]
    G = T.groupby("m").agg(cost=("cost", "mean"), macro=("macro", "mean"), sd=("macro", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for _, r in G.iterrows():
        ax.errorbar(r.cost, r.macro, yerr=r.sd, fmt="o", color=COL.get(r.m, "k"), ms=6, capsize=2)
        ax.annotate(r.m, (r.cost, r.macro), xytext=(5, 3), textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlabel("Mean wall-clock cost per update (s, log scale)"); ax.set_ylabel("Mean prequential macro-F1")
    fig.savefig(os.path.join(FIG, "fig_cost_accuracy.png")); plt.close(fig); G.to_csv(os.path.join(OUT, "tables", "drift_cost_accuracy.csv"), index=False)

def fig_training():
    p = os.path.join(OUT, "stationary", "X-IIoTID", "agt_hist_seed0.json")
    if not os.path.exists(p): return
    h = json.load(open(p)); fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax[0].plot(h["loss"], color="#1f4e79"); ax[0].set_title("(a) Training loss", fontsize=9); ax[0].set_xlabel("Epoch")
    ax[1].plot(h["acc"], color="#1f4e79"); ax[1].set_title("(b) Training accuracy", fontsize=9); ax[1].set_xlabel("Epoch")
    e, v = zip(*h["val"]); ax[2].plot(e, v, "o-", ms=2.5, color="#c00000"); ax[2].set_title("(c) Validation macro-F1", fontsize=9); ax[2].set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_training.png")); plt.close(fig)

def fig_noise():
    T = pd.read_csv(os.path.join(OUT, "tables", "noise.csv")) if os.path.exists(os.path.join(OUT, "tables", "noise.csv")) else None
    if T is None: return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for m, c in [("XGBoost", "#1f4e79"), ("RandomForest", "#9dc3e6"), ("MLP", "#548235"), ("GAT", "#7030a0"), ("AGT", "#c00000")]:
        s = T[T.model == m]
        if len(s): ax.errorbar(s.sigma, s.F1, yerr=s.F1_sd, marker="o", ms=3, capsize=2, color=c, label=m)
    ax.set_xlabel(r"Gaussian perturbation $\sigma$ (standardized numeric attributes)"); ax.set_ylabel("Weighted F1"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG, "fig12_noise.png")); plt.close(fig)

def fig_comparison():
    T = pd.read_csv(os.path.join(OUT, "tables", "main_X-IIoTID.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); x = np.arange(len(T))
    ax.bar(x - 0.2, T.F1, 0.4, yerr=T.F1_sd, color="#9dc3e6", label="Weighted F1", capsize=2)
    ax.bar(x + 0.2, T.MacroF1, 0.4, yerr=T.MacroF1_sd, color="#1f4e79", label="Macro-F1", capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(T.Model, rotation=30, ha="right"); ax.set_ylim(0.8, 1.0); ax.legend(frameon=False, ncol=2)
    ax.set_ylabel("Score (mean $\\pm$ sd, 10 seeds)"); fig.savefig(os.path.join(FIG, "fig5_comparison.png")); plt.close(fig)

if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig_drift_composition", "fig_drift_trajectory", "fig_cost_accuracy", "fig_training", "fig_noise", "fig_comparison"]:
        try: globals()[f](); print("ok", f)
        except Exception as e: print("FAIL", f, e)
