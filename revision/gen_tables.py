# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Writes LaTeX tables and inline values for the revised manuscript straight from the result files
import os, json, json, glob, numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MS = os.environ.get("AGT_MANUSCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuscript_tables"))
TD = os.path.join(MS, "tables"); os.makedirs(TD, exist_ok=True)
V = {}
def jl(p): return [json.loads(l) for l in open(p)]
def pm(m, s, d=3): return f"{m:.{d}f} $\\pm$ {s:.{d}f}"
def w(name, s): open(os.path.join(TD, name + ".tex"), "w", encoding="utf-8").write(s)
NAMES = {"LogReg": "Logistic regression", "RandomForest": "Random forest", "XGBoost": "XGBoost", "MLP": "MLP", "GCN": "GCN",
         "GraphSAGE": "GraphSAGE", "GAT": "GAT", "E-GraphSAGE": "E-GraphSAGE$^{\\dagger}$", "GPS": "GPS (Performer)",
         "TransformerConv": "TransformerConv", "AGT": "\\textbf{AGT (proposed)}"}
CLS = {"Lateral _movement": "Lateral movement", "crypto-ransomware": "Crypto-ransomware", "C&C": "C\\&C"}

def pstar(p):
    if p != p: return ""
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")

# ---- main stationary table ----
T = pd.read_csv(os.path.join(OUT, "tables", "main_X-IIoTID.csv"))
rows = []
for _, r in T.iterrows():
    cells = [pm(r[k], r[k + "_sd"]) for k in ["Accuracy", "F1", "MacroF1", "MCC"]] + [f"{r['ROC_AUC']:.4f}"]
    star = pstar(r.get("p_macroF1_holm", np.nan))
    cells[2] = cells[2] + star
    rows.append(f"{NAMES[r.Model]} & " + " & ".join(cells) + " \\\\")
    V[f"M_{r.Model}_macro"] = f"{r.MacroF1:.3f}"; V[f"M_{r.Model}_f1"] = f"{r.F1:.3f}"; V[f"M_{r.Model}_acc"] = f"{r.Accuracy:.3f}"
    V[f"M_{r.Model}_macro_sd"] = f"{r.MacroF1_sd:.3f}"; V[f"M_{r.Model}_mcc"] = f"{r.MCC:.3f}"
    if r.Model != "AGT": V[f"M_{r.Model}_p"] = f"{r.p_macroF1_holm:.3f}"; V[f"M_{r.Model}_d"] = f"{r.diff_macroF1:+.3f}"
w("tab_main", "\n".join(rows))

# ---- full corpus ----
F = pd.DataFrame(jl(os.path.join(OUT, "fullcorpus", "runs.jsonl")))
G = F.groupby("model").agg({k: ["mean", "std"] for k in ["Accuracy", "F1", "MacroF1", "MCC", "train_s"]})
rows = []
for m in ["RandomForest", "XGBoost", "MLP", "GAT", "TransformerConv", "AGT"]:
    g = G.loc[m]
    rows.append(f"{NAMES[m]} & " + " & ".join(pm(g[(k, 'mean')], g[(k, 'std')]) for k in ["Accuracy", "F1", "MacroF1", "MCC"])
                + f" & {g[('train_s', 'mean')]:.0f} \\\\")
    V[f"FC_{m}_macro"] = f"{g[('MacroF1', 'mean')]:.3f}"; V[f"FC_{m}_acc"] = f"{g[('Accuracy', 'mean')]:.3f}"; V[f"FC_{m}_train"] = f"{g[('train_s', 'mean')]:.0f}"
w("tab_full", "\n".join(rows))
f0 = F.iloc[0]; V["FC_NTR"] = f"{f0.n_train:,}".replace(",", "{,}"); V["FC_NVA"] = f"{f0.n_val:,}".replace(",", "{,}"); V["FC_NTE"] = f"{f0.n_test:,}".replace(",", "{,}")
sup = dict(zip(f0.classes, f0.support)); V["FC_SUP_CRYPTO"] = str(sup["crypto-ransomware"]); V["FC_SUP_EXPL"] = str(sup["Exploitation"])
a = F[F.model == "AGT"]; V["FC_AGT_PEAK_GB"] = f"{a.peak_MB.mean() / 1000:.1f}"; V["FC_KNN_TRAIN"] = f"{a.knn_train_s.mean():.0f}"; V["FC_KNN_QUERY"] = f"{a.knn_query_s.mean():.0f}"
V["FC_AGT_INFER"] = f"{a.infer_s.mean():.1f}"
pcf = {m: np.mean(np.vstack(F[F.model == m].perclass.values), 0) for m in ["AGT", "XGBoost", "MLP"]}

# ---- per-class (pooled over seeds) + full corpus F1 ----
pc = json.load(open(os.path.join(OUT, "tables", "perclass_X-IIoTID.json")))
o = np.argsort(-np.array(pc["AGT"]["support"])); rows = []
for i in o:
    c = pc["AGT"]["classes"][i]; a = pc["AGT"]; x = pc["XGBoost"]
    rows.append(f"{CLS.get(c, c)} & {a['precision'][i]:.3f} & {a['recall'][i]:.3f} & {a['f1'][i]:.3f} ({a['f1_seed_sd'][i]:.3f}) & "
                f"{x['f1'][i]:.3f} & {a['support'][i] // 10} & {pcf['AGT'][i]:.3f} & {pcf['XGBoost'][i]:.3f} & {sup[c]} \\\\")
    V[f"PC_{c}_f1"] = f"{a['f1'][i]:.3f}"; V[f"PC_{c}_sd"] = f"{a['f1_seed_sd'][i]:.3f}"; V[f"PC_{c}_fc"] = f"{pcf['AGT'][i]:.3f}"
w("tab_perclass", "\n".join(rows))

# ---- multi-dataset (models as rows, datasets as columns) ----
DSL = [("X-IIoTID", 10), ("Edge-IIoTset", 15), ("WUSTL-IIoT-2021", 5), ("CICIoT2023", 8)]
DD = {ds: pd.read_csv(os.path.join(OUT, "tables", f"main_{ds}.csv")).set_index("Model") for ds, _ in DSL}
rows = []
for m in ["LogReg", "RandomForest", "XGBoost", "MLP", "GCN", "GraphSAGE", "GAT", "GPS", "TransformerConv", "AGT"]:
    rows.append(f"{NAMES[m]} & " + " & ".join(f"{DD[ds].loc[m, 'F1']:.3f} / {DD[ds].loc[m, 'MacroF1']:.3f}" for ds, _ in DSL) + " \\\\")
for ds, _ in DSL:
    for m in DD[ds].index: V[f"DS_{ds}_{m}_macro"] = f"{DD[ds].loc[m, 'MacroF1']:.3f}"; V[f"DS_{ds}_{m}_f1"] = f"{DD[ds].loc[m, 'F1']:.3f}"
w("tab_multids", "\n".join(rows))

# ---- ablation ----
A = pd.read_csv(os.path.join(OUT, "tables", "ablation.csv")); rows = []
lab = {"Full AGT": "Full AGT", "- global attention": "$-$ landmark attention",
       "global = node-wise landmark memory (submitted)": "node-wise memory (initial version)",
       "global = parameter-matched FFN": "parameter-matched FFN",
       "- entity embeddings (ordinal codes)": "$-$ entity embeddings (ordinal codes)", "- class weighting": "$-$ class weighting",
       "one local layer": "one local layer", "- graph topology (no edges)": "$-$ graph edges"}
for _, r in A.iterrows():
    d = "--" if r.Variant == "Full AGT" else f"{r.dMacroF1:+.3f}{pstar(r.p_holm)}"
    rows.append(f"{lab[r.Variant]} & {int(r.params):d} & {r.F1:.3f} ({r.F1_sd:.3f}) & {r.MacroF1:.3f} ({r.MacroF1_sd:.3f}) & {d} \\\\")
    V["AB_" + r.Variant] = f"{r.MacroF1:.3f}"
    if r.Variant != "Full AGT": V["ABd_" + r.Variant] = f"{r.dMacroF1:+.3f}"; V["ABp_" + r.Variant] = f"{r.p_holm:.3f}"; V["ABf1_" + r.Variant] = f"{r.dF1:+.4f}"
w("tab_ablation", "\n".join(rows))

# ---- sensitivity ----
S = pd.read_csv(os.path.join(OUT, "tables", "sensitivity.csv"))
for p in ["L", "k"]:
    s = S[S.param == p]
    w(f"tab_sens_{p}", "\n".join(f"{int(r.value)} & {pm(r.val, r.val_sd)} & {pm(r.MacroF1, r.MacroF1_sd)} & {r.train_s:.0f} \\\\" for _, r in s.iterrows()))
    for _, r in s.iterrows():
        V[f"S_{p}{int(r.value)}_val"] = f"{r.val:.3f}"; V[f"S_{p}{int(r.value)}_test"] = f"{r.MacroF1:.3f}"; V[f"S_{p}{int(r.value)}_train"] = f"{r.train_s:.0f}"

# ---- noise ----
N = pd.read_csv(os.path.join(OUT, "tables", "noise.csv")); rows = []
for m in ["XGBoost", "RandomForest", "MLP", "GAT", "AGT"]:
    s = N[N.model == m].sort_values("sigma")
    rows.append(f"{NAMES.get(m, m)} & " + " & ".join(f"{a:.3f}" for a in s.F1) + " & " + " & ".join(f"{a:.3f}" for a in s.MacroF1.iloc[[0, -1]]) + " \\\\")
    V[f"N_{m}_f1_0"] = f"{s.F1.iloc[0]:.3f}"; V[f"N_{m}_f1_5"] = f"{s.F1.iloc[-1]:.3f}"; V[f"N_{m}_mac_0"] = f"{s.MacroF1.iloc[0]:.3f}"; V[f"N_{m}_mac_5"] = f"{s.MacroF1.iloc[-1]:.3f}"
w("tab_noise", "\n".join(rows))

# ---- drift (multi, main) ----
def drift(pat, ms=None):
    res = {}
    for f in sorted(glob.glob(os.path.join(OUT, "drift", pat))):
        d = json.load(open(f))
        for m in d["rows"][0]:
            if m in ("seed", "window") or (ms and m not in ms): continue
            t = d["timing"].get(m, [])
            cost = np.mean([x.get("total_s", x.get("graph_s", 0) + x.get("steps_s", 0) + x.get("buffer_s", 0)) for x in t]) if t else np.nan
            res.setdefault(m, []).append(dict(F1=np.mean([r[m]["F1"] for r in d["rows"]]), Mac=np.mean([r[m]["MacroF1"] for r in d["rows"]]),
                                              Att=np.mean([r[m]["AttackF1"] for r in d["rows"]]), Rec=np.mean([r[m]["AttackRecall"] for r in d["rows"]]),
                                              cost=cost, inf=np.mean([r[m]["infer_s"] for r in d["rows"]])))
    return {m: pd.DataFrame(v) for m, v in res.items()}
base = drift("multi_rho0.2_M2500_eps0.0_seed*.json")
tun = {**{k + " (tuned)": v for k, v in drift("multi_*_eps0.0_recent_S15_eta0.003_tuned_seed*.json").items()},
       **{k + " (tuned)": v for k, v in drift("multi_*_tunedmlp_seed*.json").items()},
       **{k + " (tuned)": v for k, v in drift("multi_*_tunedxgb_seed*.json").items()}}
allm = {**base, **tun}
DL = {"XGB-static": "XGBoost, static", "MLP-static": "MLP, static", "AGT-static": "AGT, static", "INSOMNIA-style": "INSOMNIA-style (MLP)",
      "XGB-continued (tuned)": "XGBoost, continued boosting", "AGT-finetune": "AGT, fine-tuning without buffer",
      "MLP-rehearsal (tuned)": "MLP + rehearsal", "AGT-rehearsal (tuned)": "\\textbf{AGT + rehearsal (proposed)}",
      "XGB-sliding (tuned)": "XGBoost, sliding-window refit", "AGT-retrain": "AGT, full retraining", "XGB-retrain": "XGBoost, full retraining"}
rows = []
for m in DL:
    d = allm[m]
    # tuned runs shared the GPU with other jobs; same work per update as the untuned run, so its uncontended cost is used
    cd = base[m.replace(" (tuned)", "")] if (m.endswith("(tuned)") and not m.startswith("MLP")) else d
    c = "--" if np.isnan(cd.cost.mean()) else f"{cd.cost.mean():.2f}"
    rows.append(f"{DL[m]} & {pm(d.F1.mean(), d.F1.std(ddof=1))} & {pm(d.Mac.mean(), d.Mac.std(ddof=1))} & {pm(d.Att.mean(), d.Att.std(ddof=1))} & {c} \\\\")
    key = m.replace(" (tuned)", "")
    V[f"D_{key}_f1"] = f"{d.F1.mean():.3f}"; V[f"D_{key}_mac"] = f"{d.Mac.mean():.3f}"; V[f"D_{key}_att"] = f"{d.Att.mean():.3f}"; V[f"D_{key}_cost"] = c
    V[f"D_{key}_rec"] = f"{d.Rec.mean():.3f}"
w("tab_drift", "\n".join(rows))
V["D_AGTreh_untuned_mac"] = f"{base['AGT-rehearsal'].Mac.mean():.3f}"; V["D_AGTreh_untuned_f1"] = f"{base['AGT-rehearsal'].F1.mean():.3f}"
V["D_MLPreh_untuned_mac"] = f"{base['MLP-rehearsal'].Mac.mean():.3f}"
# per-seed differences
for o_ in ["MLP-rehearsal (tuned)", "XGB-sliding (tuned)", "XGB-retrain"]:
    a = allm["AGT-rehearsal (tuned)"].Mac.values; b = allm[o_].Mac.values
    V["Ddiff_" + o_.replace(" (tuned)", "")] = f"{np.mean(a - b):+.3f}"; V["Dwins_" + o_.replace(" (tuned)", "")] = str(int(np.sum(a < b)))

# ---- binary: original settings (A) and corrected protocol (B) ----
A_ = drift("binary_rho0.5_M2500_eps0.0_origsetting_seed*.json"); B_ = drift("binary_rho0.2_M2500_eps0.0_*seed*.json")
rows = []
for m, l in [("XGB-static", "XGBoost, static"), ("MLP-static", "MLP, static"), ("AGT-static", "AGT, static"), ("MLP-rehearsal", "MLP + rehearsal"),
             ("AGT-rehearsal", "AGT + rehearsal"), ("XGB-sliding", "XGBoost, sliding-window refit"), ("XGB-continued", "XGBoost, continued boosting"),
             ("XGB-retrain", "XGBoost, full retraining")]:
    a = pm(A_[m].F1.mean(), A_[m].F1.std(ddof=1)) if m in A_ else "--"; b = pm(B_[m].F1.mean(), B_[m].F1.std(ddof=1)) if m in B_ else "--"
    rows.append(f"{l} & {a} & {b} \\\\")
    if m in A_: V[f"BA_{m}"] = f"{A_[m].F1.mean():.3f}"
    if m in B_: V[f"BB_{m}"] = f"{B_[m].F1.mean():.3f}"
w("tab_binary", "\n".join(rows))

# ---- label budget and buffer ----
rows = []
for r_ in ["0.05", "0.1", "0.2", "0.5"]:
    d = drift(f"multi_rho{r_}_M2500_eps0.0_recent_S15_eta0.003_seed*.json") if r_ != "0.2" else \
        {**drift("multi_rho0.2_M2500_eps0.0_recent_S15_eta0.003_epsref_seed*.json"), **drift("multi_rho0.2_M2500_eps0.0_seed[012].json", ["XGB-sliding", "XGB-retrain"])}
    rows.append(f"$\\rho={r_}$ ({int(float(r_) * 10255)} labels) & " + " & ".join(
        (f"{d[m].Mac.mean():.3f}" if m in d else "--") for m in ["AGT-rehearsal", "MLP-rehearsal", "XGB-sliding", "XGB-retrain"]) + " \\\\")
for M_ in ["1000", "2500", "5000", "10000"]:
    d = drift(f"multi_rho0.2_M{M_}_eps0.0_recent_S15_eta0.003_seed*.json") if M_ != "2500" else \
        {**drift("multi_rho0.2_M2500_eps0.0_recent_S15_eta0.003_epsref_seed*.json"), **drift("multi_rho0.2_M2500_eps0.0_seed[012].json", ["XGB-sliding"])}
    rows.append(f"$M={M_}$ & " + " & ".join((f"{d[m].Mac.mean():.3f}" if m in d else "--") for m in ["AGT-rehearsal", "MLP-rehearsal", "XGB-sliding", "XGB-retrain"]) + " \\\\")
w("tab_budget", "\n".join(rows))

# ---- poisoning ----
P = pd.read_csv(os.path.join(OUT, "tables", "poisoning.csv")); rows = []
for m, l in [("XGB-retrain", "XGBoost, full retraining"), ("XGB-sliding", "XGBoost, sliding-window refit"), ("MLP-rehearsal", "MLP + rehearsal"), ("AGT-rehearsal", "AGT + rehearsal")]:
    s = P[P.m == m].sort_values("eps")
    rows.append(f"{l} & " + " & ".join(f"{a:.3f}" for a in s.rec) + " & " + " & ".join(f"{a:.3f}" for a in s.mac) + " \\\\")
    V[f"P_{m}_rec0"] = f"{s.rec.iloc[0]:.3f}"; V[f"P_{m}_rec3"] = f"{s.rec.iloc[-1]:.3f}"; V[f"P_{m}_mac0"] = f"{s.mac.iloc[0]:.3f}"; V[f"P_{m}_mac3"] = f"{s.mac.iloc[-1]:.3f}"
w("tab_poison", "\n".join(rows))

# ---- HPC ----
H = os.path.join(OUT, "hpc"); arr = json.load(open(os.path.join(H, "arrival.json")))
V["ARR_MAXS"] = f"{arr['max_per_s']:,}".replace(",", "{,}"); V["ARR_MAXMIN"] = f"{arr['max_per_min']:,}".replace(",", "{,}")
V["ARR_P99"] = f"{arr['p99_per_s']:.0f}"; V["ARR_MEAN"] = f"{arr['mean_per_active_s']:.1f}"; V["ARR_MAX10"] = f"{arr['max_per_10min']:,}".replace(",", "{,}")
K = pd.DataFrame(jl(os.path.join(H, "knn_scaling.jsonl"))); rows = []
for N_ in sorted(K.N.unique()):
    s = K[K.N == N_].set_index("method")
    def g(m, th=None):
        q = K[(K.N == N_) & (K.method == m) & ((K.threads == th) if th is not None else True)]
        return f"{q.seconds.iloc[0]:.2f}" if len(q) else "--"
    hn = K[(K.N == N_) & (K.method == "CPU approximate (FAISS HNSW)")]
    rows.append(f"{N_:,} & {g('GPU exact (chunked GEMM)')} & {g('CPU exact (FAISS flat)', 1)} & {g('CPU exact (FAISS flat)', 16)} & "
                f"{g('CPU exact (scikit-learn)')} & {hn.seconds.iloc[0]:.2f} ({hn.recall.iloc[0]:.3f}) \\\\".replace(f"{N_:,}", f"{N_:,}".replace(",", "{,}")))
w("tab_knn", "\n".join(rows))
gpu = K[(K.method == "GPU exact (chunked GEMM)")].set_index("N").seconds; cpu = K[(K.method == "CPU exact (FAISS flat)") & (K.threads == 16)].set_index("N").seconds
NF = K.N.max(); V["K_GPU_FULL"] = f"{gpu[NF]:.0f}"; V["K_CPU_FULL"] = f"{cpu[NF]:.0f}"; V["K_SPEED_FULL"] = f"{cpu[NF] / gpu[NF]:.1f}"
hn = K[(K.method == "CPU approximate (FAISS HNSW)") & (K.N == NF)].iloc[0]; V["K_HNSW_FULL"] = f"{hn.seconds:.0f}"; V["K_HNSW_REC"] = f"{hn.recall:.3f}"
V["K_NFULL"] = f"{NF:,}".replace(",", "{,}"); V["K_GPU_80K"] = f"{gpu[80000]:.1f}"; V["K_CPU_80K"] = f"{cpu[80000]:.1f}"
th = pd.DataFrame(jl(os.path.join(H, "thread_scaling.jsonl"))).set_index("threads").seconds
V["TH_1"] = f"{th[1]:.1f}"; V["TH_4"] = f"{th[4]:.1f}"; V["TH_16"] = f"{th[16]:.1f}"
TP = pd.DataFrame(jl(os.path.join(H, "throughput.jsonl")))
for _, r in TP.iterrows(): V[f"TP_{r.device}_{r.batch}"] = f"{r.flows_per_s:,.0f}".replace(",", "{,}"); V[f"TPms_{r.device}_{r.batch}"] = f"{r.total_s * 1000:.0f}"
RG = pd.DataFrame(jl(os.path.join(H, "retrain_growth.jsonl")))
x = RG[RG.model == "XGBoost full refit"]
for _, r in x.iterrows(): V[f"RG_{r.device}_{r.N}"] = f"{r.seconds:.1f}"
u = RG[RG.model == "AGT rehearsal update"].iloc[1:]; V["RG_AGT"] = f"{u.seconds.mean():.2f}"; V["RG_AGT_graph"] = f"{u.graph_s.mean():.3f}"; V["RG_AGT_steps"] = f"{u.steps_s.mean():.2f}"
V["RG_RATIO_CPU"] = f"{x[(x.device == 'cpu')].seconds.max() / u.seconds.mean():.0f}"; V["RG_RATIO_GPU"] = f"{x[(x.device == 'cuda')].seconds.max() / u.seconds.mean():.0f}"
MEM = pd.DataFrame(jl(os.path.join(H, "memory.jsonl"))).set_index("phase")
V["MEM_KNN"] = f"{MEM.peak_alloc_MB.iloc[0] / 1000:.1f}"; V["MEM_TRAIN"] = f"{MEM.peak_alloc_MB.iloc[1] / 1000:.1f}"; V["MEM_INF"] = f"{MEM.peak_alloc_MB.iloc[2] / 1000:.1f}"
V["MEM_STATIC"] = f"{MEM.static_MB.iloc[1]:.0f}"; V["MEM_RES_TRAIN"] = f"{MEM.reserved_MB.iloc[1] / 1000:.1f}"
# drift per-window cost breakdown (main protocol)
T_ = {}
for f in glob.glob(os.path.join(OUT, "drift", "multi_rho0.2_M2500_eps0.0_seed*.json")):
    d = json.load(open(f))
    for m, t in d["timing"].items():
        for x_ in t:
            for k, v in x_.items(): T_.setdefault(m, {}).setdefault(k, []).append(v)
    for r in d["rows"]: T_.setdefault("inf_" + "AGT-rehearsal", {}).setdefault("s", []).append(r["AGT-rehearsal"]["infer_s"])
V["DC_graph"] = f"{np.mean(T_['AGT-rehearsal']['graph_s']) * 1000:.0f}"; V["DC_steps"] = f"{np.mean(T_['AGT-rehearsal']['steps_s']):.2f}"
V["DC_buffer"] = f"{np.mean(T_['AGT-rehearsal']['buffer_s']) * 1000:.2f}"; V["DC_infer"] = f"{np.mean(T_['inf_AGT-rehearsal']['s']):.2f}"
V["DC_total"] = f"{np.mean(T_['AGT-rehearsal']['graph_s']) + np.mean(T_['AGT-rehearsal']['steps_s']) + np.mean(T_['AGT-rehearsal']['buffer_s']):.2f}"
V["DC_xgb_mean"] = f"{np.mean(T_['XGB-retrain']['total_s']):.1f}"; V["DC_xgb_last"] = f"{np.mean(T_['XGB-retrain']['total_s'][14::15]):.1f}"
V["DC_agtre_mean"] = f"{np.mean(T_['AGT-retrain']['total_s']):.1f}"; V["DC_agtre_last"] = f"{np.mean(T_['AGT-retrain']['total_s'][14::15]):.1f}"
V["DC_sliding"] = f"{np.mean(T_['XGB-sliding']['total_s']):.1f}"; V["DC_nodes"] = f"{np.mean(T_['AGT-rehearsal']['n_nodes']):.0f}"
tot = float(V["DC_total"]); V["DC_ratio_mean"] = f"{float(V['DC_xgb_mean']) / tot:.0f}"; V["DC_ratio_last"] = f"{float(V['DC_xgb_last']) / tot:.0f}"
V["DC_ratio_sliding"] = f"{float(V['DC_sliding']) / tot:.0f}"
# stationary graph/AGT timing
R = pd.DataFrame(jl(os.path.join(OUT, "stationary", "X-IIoTID", "main_runs.jsonl"))); a = R[R.model == "AGT"]
V["ST_AGT_train"] = f"{a.train_s.mean():.0f}"; V["ST_graph"] = f"{a.graph_s.mean():.1f}"; V["ST_AGT_infer"] = f"{a.infer_s.mean() * 1000:.0f}"
V["AGT_PARAMS"] = f"{int(a.params.iloc[0]):,}".replace(",", "{,}")
imp = json.load(open(os.path.join(OUT, "stationary", "X-IIoTID", "importance.json")))
V["IMP_purity"] = f"{imp['service_class_purity']:.3f}"; r_ = {x_["feature"]: x_ for x_ in imp["rows"]}
for f_ in ["Service", "read_write_physical.process", "is_with_payload", "Conn_state", "Duration", "anomaly_alert"]:
    V[f"IMP_{f_}"] = f"{r_[f_]['agt_drop']:.3f}"; V[f"IMPfix_{f_}"] = f"{r_[f_]['agt_drop_fixedgraph']:.3f}"; V[f"IMPx_{f_}"] = f"{r_[f_]['xgb_perm_drop']:.3f}"; V[f"IMPg_{f_}"] = f"{r_[f_]['xgb_gain_share']:.3f}"
top = sorted(imp["rows"], key=lambda x_: -x_["agt_drop"])[:10]
FEAT = {"read_write_physical.process": "Physical-process read/write", "is_with_payload": "Has payload", "Avg_system_time": "Avg. system time",
        "anomaly_alert": "Anomaly alert", "Conn_state": "Connection state", "Std_user_time": "Std. user time", "Duration": "Duration",
        "Std_wtps": "Std. write transfers/s", "Is_SYN_ACK": "SYN-ACK flag", "Service": "Service"}
rank_x = {x_["feature"]: i + 1 for i, x_ in enumerate(sorted(imp["rows"], key=lambda z: -z["xgb_gain_share"]))}
rows = [f"{FEAT.get(x_['feature'], x_['feature'])}$^{{{i + 1}}}$ & {x_['agt_drop']:.3f} & {x_['agt_drop_fixedgraph']:.3f} & {x_['xgb_perm_drop']:.3f} & {rank_x[x_['feature']]} \\\\"
        for i, x_ in enumerate(top)]
w("tab_importance", "\n".join(rows))
w("tab_importance_foot", ", ".join(f"$^{{{i + 1}}}$\\texttt{{{x_['feature'].replace('_', chr(92) + '_')}}}" for i, x_ in enumerate(top)))
json.dump(V, open(os.path.join(MS, "values.json"), "w"), indent=1)
print(len(V), "values;", sorted(os.listdir(TD)))
