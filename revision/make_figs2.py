# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Additional revision figures (HPC, per-class, importance, noise, poisoning)
import sys
from make_figs import *

NICE = {"Lateral _movement": "Lateral movement", "crypto-ransomware": "Crypto-ransomware"}
FEAT = {"read_write_physical.process": "Physical-process read/write", "is_with_payload": "Has payload",
        "Avg_system_time": "Avg. system time", "anomaly_alert": "Anomaly alert", "Conn_state": "Connection state",
        "Std_user_time": "Std. user time", "Duration": "Duration", "Std_wtps": "Std. write transfers/s",
        "Is_SYN_ACK": "SYN-ACK flag", "Service": "Service", "File_activity": "File activity",
        "Des_pkts": "Destination packets", "Scr_ip_bytes": "Source IP bytes", "Std_iowait_time": "Std. I/O wait time",
        "Des_port": "Destination port", "Is_pure_ack": "Pure-ACK flag", "Avg_ideal_time": "Avg. idle time", "Scr_pkts": "Source packets", "Avg_kbmemused": "Avg. memory used", "Login_attempt": "Login attempts", "Scr_bytes": "Source bytes", "Protocol": "Protocol"}

def fig_hpc():
    K = pd.DataFrame(jl(os.path.join(OUT, "hpc", "knn_scaling.jsonl"))); TP = pd.DataFrame(jl(os.path.join(OUT, "hpc", "throughput.jsonl")))
    RG = pd.DataFrame(jl(os.path.join(OUT, "hpc", "retrain_growth.jsonl"))); arr = json.load(open(os.path.join(OUT, "hpc", "arrival.json")))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    sty = {"GPU exact (chunked GEMM)": ("#c00000", "o-", "GPU exact"), "CPU exact (FAISS flat)": ("#1f4e79", "s-", "CPU exact (FAISS)"),
           "CPU exact (scikit-learn)": ("#9dc3e6", "^-", "CPU exact (sklearn)"), "CPU approximate (FAISS HNSW)": ("#548235", "d--", "CPU HNSW (approx.)")}
    for m, (c, s, lab) in sty.items():
        d = K[(K.method == m) & ((K.threads == 16) | (K.threads == 0))]
        ax[0].plot(d.N, d.seconds, s, color=c, ms=3, lw=1.2, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Flows N"); ax[0].set_ylabel("k-NN graph build (s)")
    ax[0].set_title("(a) Graph construction", fontsize=9); ax[0].legend(fontsize=5.8, frameon=False)
    for dv, c, lab in [("cuda", "#c00000", "GPU"), ("cpu", "#1f4e79", "CPU (16 threads)")]:
        d = TP[TP.device == dv]; ax[1].plot(d.batch, d.flows_per_s, "o-", color=c, ms=3, label=lab)
    ax[1].axhline(arr["max_per_s"], color="k", ls=":", lw=1); ax[1].text(1.6e4, arr["max_per_s"] * 0.88, "peak 1-s burst\nin X-IIoTID", fontsize=6.3, va="top")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("Flows scored per batch"); ax[1].set_ylabel("Flows per second")
    ax[1].set_title("(b) End-to-end scoring", fontsize=9); ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    x = RG[RG.model == "XGBoost full refit"]
    for dv, c, lab in [("cpu", "#1f4e79", "XGBoost refit (CPU)"), ("cuda", "#4f81bd", "XGBoost refit (GPU)")]:
        d = x[x.device == dv]; ax[2].plot(d.N, d.seconds, "s-", color=c, ms=3, label=lab)
    u = RG[RG.model == "AGT rehearsal update"].seconds.iloc[1:].mean()
    ax[2].plot([x.N.min(), x.N.max()], [u, u], "-", color="#c00000", label="AGT rehearsal update")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("Training flows N"); ax[2].set_ylabel("Seconds per update")
    ax[2].set_title("(c) Keeping the model current", fontsize=9); ax[2].legend(fontsize=6.2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig15_hpc.png")); plt.close(fig)

def fig_perclass():
    pc = json.load(open(os.path.join(OUT, "tables", "perclass_X-IIoTID.json")))
    cls = [NICE.get(c, c) for c in pc["AGT"]["classes"]]; o = np.argsort(-np.array(pc["AGT"]["support"]))
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); x = np.arange(len(cls))
    for j, (m, c) in enumerate([("XGBoost", "#1f4e79"), ("MLP", "#548235"), ("AGT", "#c00000")]):
        f = np.array(pc[m]["f1"])[o]; sd = np.array(pc[m]["f1_seed_sd"])[o]
        ax.bar(x + (j - 1) * 0.27, f, 0.27, yerr=sd, capsize=1.5, color=c, label=m)
    ax.set_xticks(x); ax.set_xticklabels([f"{cls[i]}\n(n={pc['AGT']['support'][i] // 10})" for i in o], fontsize=6.8, rotation=35, ha="right")
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("F1 (pooled over 10 seeds)"); ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(os.path.join(FIG, "fig6_perclass.png")); plt.close(fig)

def fig_confusion():
    pc = json.load(open(os.path.join(OUT, "tables", "perclass_X-IIoTID.json"))); cm = np.array(pc["AGT"]["cm"], float)
    cmn = cm / cm.sum(1, keepdims=True); cls = [NICE.get(c, c) for c in pc["AGT"]["classes"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.6)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if cmn[i, j] >= 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
    ax.set_xticklabels(cls, rotation=40, ha="right", fontsize=7); ax.set_yticklabels(cls, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False); fig.colorbar(im, fraction=0.04, label="Row-normalized rate")
    fig.savefig(os.path.join(FIG, "fig7_confusion.png")); plt.close(fig)

def fig_importance():
    d = json.load(open(os.path.join(OUT, "stationary", "X-IIoTID", "importance.json")))
    r = sorted(d["rows"], key=lambda x: -x["agt_drop"])[:15][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.0)); y = np.arange(len(r))
    ax.barh(y + 0.2, [x["agt_drop"] for x in r], 0.4, xerr=[x["agt_drop_sd"] for x in r], color="#c00000", label="AGT (neighbors re-queried)")
    ax.barh(y - 0.2, [x["xgb_perm_drop"] for x in r], 0.4, color="#1f4e79", label="XGBoost")
    ax.set_yticks(y); ax.set_yticklabels([FEAT.get(x["feature"], x["feature"]) for x in r], fontsize=7.5)
    ax.set_xlabel("Drop in macro-F1 when the attribute is permuted"); ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "fig8_importance.png")); plt.close(fig)

def fig_noise_perclass():
    P = pd.read_csv(os.path.join(OUT, "tables", "noise_perclass_agt.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 3.2)); cmap = plt.get_cmap("tab10")
    for j, c in enumerate(P.columns): ax.plot(P.index, P[c], "o-", ms=3, color=cmap(j), label=NICE.get(c, c))
    ax.set_xlabel(r"Gaussian perturbation $\sigma$"); ax.set_ylabel("AGT per-class F1"); ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.savefig(os.path.join(FIG, "fig16_noise_perclass.png")); plt.close(fig)

def fig_poison():
    rows = []
    for e in ["0.0", "0.1", "0.2", "0.3"]:
        # matched settings (tuned update, recent buffer M=2500) and seeds 0-2 at every epsilon
        pats = ["multi_rho0.2_M2500_eps0.0_recent_S15_eta0.003_epsref_seed*.json"] if e == "0.0" else [f"multi_rho0.2_M2500_eps{e}_recent_S15_eta0.003_seed*.json"]
        if e == "0.0": pats.append("multi_rho0.2_M2500_eps0.0_seed[012].json")
        for f in [g for p in pats for g in glob.glob(os.path.join(OUT, "drift", p))]:
            d = json.load(open(f))
            for m in ["AGT-rehearsal", "MLP-rehearsal", "XGB-sliding", "XGB-retrain"]:
                if "epsref" not in f and e == "0.0" and m.startswith(("AGT", "MLP")): continue
                if m in d["rows"][0]:
                    rows.append(dict(eps=float(e), m=m, rec=np.mean([r[m]["AttackRecall"] for r in d["rows"]]),
                                     mac=np.mean([r[m]["MacroF1"] for r in d["rows"]]), f1=np.mean([r[m]["F1"] for r in d["rows"]])))
    T = pd.DataFrame(rows).groupby(["m", "eps"]).agg(rec=("rec", "mean"), rsd=("rec", "std"), mac=("mac", "mean"), msd=("mac", "std"),
                                                     f1=("f1", "mean"), n=("rec", "count")).reset_index()
    T.to_csv(os.path.join(OUT, "tables", "poisoning.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for m in ["XGB-retrain", "XGB-sliding", "MLP-rehearsal", "AGT-rehearsal"]:
        s = T[T.m == m]
        ax[0].errorbar(s.eps, s.rec, yerr=s.rsd, marker="o", ms=3, capsize=2, color=COL[m], label=m)
        ax[1].errorbar(s.eps, s.mac, yerr=s.msd, marker="o", ms=3, capsize=2, color=COL[m], label=m)
    ax[0].set_xlabel(r"Fraction $\epsilon$ of revealed attack labels flipped"); ax[0].set_ylabel("Attack recall"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_xlabel(r"Fraction $\epsilon$ flipped"); ax[1].set_ylabel("Macro-F1")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_poisoning.png")); plt.close(fig); print(T.round(3).to_string(index=False))

if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f](); print("ok", f)
