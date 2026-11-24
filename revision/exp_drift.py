# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Prequential (test-then-adapt) study under the real temporal ordering of X-IIoTID.
# Usage: python exp_drift.py <task:multi|binary> <seeds> [rho] [M] [poison_eps] [methods]
import sys
import xgboost as xgb
from common import *

TASK = sys.argv[1] if len(sys.argv) > 1 else "multi"
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
RHO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2       # labelled fraction of each window
M = int(sys.argv[4]) if len(sys.argv) > 4 else 2500          # replay-buffer capacity
EPS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0       # fraction of revealed attack labels flipped to Normal
METHODS = sys.argv[6].split(",") if len(sys.argv) > 6 else None
STRIDE = int(os.environ.get("STRIDE", "4"))   # systematic subsample keeps the chronological order
INIT_FRAC = float(os.environ.get("INIT_FRAC", "0.25"))   # first part of the chronology trains the static models
T = int(os.environ.get("NWIN", "15"))   # number of future windows
# update settings: defaults = submitted configuration; drift_cfg.json (chosen on the pseudo-stream) overrides
_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_cfg.json"))) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_cfg.json")) and not os.environ.get("NO_CFG") else {}
S = int(os.environ.get("S_STEPS", _cfg.get("S", 15)))            # gradient steps per update
ETA = float(os.environ.get("ETA", _cfg.get("eta", 1.2e-3)))      # update learning rate
POLICY = os.environ.get("POLICY", _cfg.get("policy", "recent"))  # replay buffer policy: recent | balanced
if "M" in _cfg and len(sys.argv) <= 4: M = int(_cfg["M"])
PSEUDO = os.environ.get("PSEUDO") == "1"                          # tuning stream cut from the initial period only
if PSEUDO: INIT_FRAC, T = 0.3, 6
K = 10
TAG = ("pseudo_" if PSEUDO else "") + f"{TASK}_rho{RHO}_M{M}_eps{EPS}"
if POLICY != "recent" or S != 15 or ETA != 1.2e-3: TAG += f"_{POLICY}_S{S}_eta{ETA}"
if os.environ.get("TAGX"): TAG += "_" + os.environ["TAGX"]
ODIR = os.path.join(OUT, "drift"); os.makedirs(ODIR, exist_ok=True)
HPS = {k: v["best"] for k, v in json.load(open(os.path.join(OUT, "stationary", "X-IIoTID", "tuning.json"))).items()}

# ---------- data in chronological order ----------
raw = load_raw("X-IIoTID", keep=("Timestamp",))
raw["__ts"] = pd.to_numeric(raw["__Timestamp"], errors="coerce")
raw = raw.dropna(subset=["__ts"]).sort_values("__ts", kind="stable").reset_index(drop=True)
raw = raw.iloc[::STRIDE].reset_index(drop=True)
if PSEUDO: raw = raw.iloc[:int(len(raw) * 0.25)].reset_index(drop=True)   # never touches the evaluation windows
CLASSES = sorted(raw["y_str"].unique()); NORMAL = CLASSES.index("Normal")
y_multi = raw["y_str"].map({c: i for i, c in enumerate(CLASSES)}).values
y = y_multi if TASK == "multi" else (y_multi != NORMAL).astype(int)
C = len(CLASSES) if TASK == "multi" else 2
NORM_LBL = NORMAL if TASK == "multi" else 0
n = len(raw); cut = int(n * INIT_FRAC)
init = np.arange(cut); windows = np.array_split(np.arange(cut, n), T)
pr = Prep().fit(raw.iloc[:cut]); num, cat = pr.transform(raw); flat = pr.flat(num, cat)
D = dict(num=num, cat=cat, flat=flat, y=y, prep=pr, classes=CLASSES if TASK == "multi" else ["Normal", "Attack"])
print(f"{TASK}: n={n} init={cut} windows={T}x~{len(windows[0])} rho={RHO} M={M} eps={EPS}", flush=True)

def wmeta():
    rows = []
    for t, w in enumerate(windows, 1):
        ts = raw["__ts"].values[w]
        rows.append(dict(window=t, start=float(ts.min()), end=float(ts.max()), n=len(w),
                         counts=np.bincount(y_multi[w], minlength=len(CLASSES)).tolist()))
    return dict(classes=CLASSES, init_counts=np.bincount(y_multi[init], minlength=len(CLASSES)).tolist(),
                init_start=float(raw["__ts"].values[0]), init_end=float(raw["__ts"].values[cut - 1]), windows=rows)

# ---------- helpers ----------
def xgb_train(X, yv, prev=None, rounds=None, seed=0, sw=None):
    hp = HPS["XGBoost"]
    p = dict(objective="multi:softprob", num_class=C, max_depth=hp["max_depth"], eta=hp["eta"], subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", device="cpu", nthread=16, seed=seed, verbosity=0)  # native-API GPU path gave constant predictions on this GPU
    w = None
    if hp["sw"]:
        cnt = np.bincount(yv, minlength=C); w = (len(yv) / (C * np.maximum(cnt, 1)))[yv]
    return xgb.train(p, xgb.DMatrix(X, label=yv, weight=w), num_boost_round=rounds or 600, xgb_model=prev)

def xgb_proba(b, X): return b.predict(xgb.DMatrix(X)).reshape(len(X), C)

def neural(kind):
    hp = HPS[kind]; return build_model(kind if kind != "MLP" else "MLP", dict(D, classes=D["classes"]), hp), hp

def cw(yv): return class_weights(yv, C)

def initial_fit(kind, seed, sel):
    rng = np.random.default_rng(seed); va = rng.choice(sel, int(0.1 * len(sel)), replace=False)
    tr = np.setdiff1d(sel, va)
    gtr, nb = make_graph(D, tr, k=K); gva, _ = make_graph(D, tr, va, k=K, nbr_ref=nb)
    m, hp = neural(kind)
    m, info = train_net(m, gtr, gva, y[va], epochs=150, lr=hp["lr"], C=C, seed=seed)
    return m, tr, info

def update(m, sel, opt, steps=S, yv=None):
    # few-step update on the graph over the labelled window part and the replay buffer
    t0 = time.time(); g, _ = make_graph(D, sel, k=K)
    if yv is not None: g.y = torch.tensor(yv, dtype=torch.long)
    g = g.to(DEV); tg = time.time() - t0
    w = cw(g.y.cpu().numpy()); m.train(); t1 = time.time()
    for _ in range(steps):
        opt.zero_grad(); loss = F.cross_entropy(m(g), g.y, weight=w); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
    if DEV.type == "cuda": torch.cuda.synchronize()
    return tg, time.time() - t1

def infer(m, ref, w):
    t0 = time.time(); g, _ = make_graph(D, ref, w, k=K); p = predict(m, g)
    return p, time.time() - t0

def wscore(w, p):
    yt = y[w]; yp = p.argmax(1); present = np.unique(yt)
    att_t = (y_multi[w] != NORMAL).astype(int); att_p = (yp != NORM_LBL).astype(int)
    return dict(F1=f1_score(yt, yp, average="weighted", zero_division=0),
                MacroF1=f1_score(yt, yp, labels=present, average="macro", zero_division=0),
                AttackF1=f1_score(att_t, att_p, zero_division=0),
                AttackRecall=recall_score(att_t, att_p, zero_division=0),
                perclass_recall={int(c): float((yp[yt == c] == c).mean()) for c in present},
                perclass_attack_flag={int(c): float((yp[y_multi[w] == c] != NORM_LBL).mean()) for c in np.unique(y_multi[w])})

def refresh(buf, ybuf, A, yA):
    # recent: keep the M most recent labelled flows; balanced: equal share per seen class, most recent first
    b = np.concatenate([buf, A]); yb = np.concatenate([ybuf, yA])
    if POLICY == "recent" or len(b) <= M: return b[-M:], yb[-M:]
    cls = np.unique(yb); keep = []; quota = M // len(cls); spare = 0
    for c in cls:
        idx = np.where(yb == c)[0]
        keep.append(idx[-quota:]); spare += max(0, quota - len(idx))
    keep = np.concatenate(keep)
    if spare > 0:
        rest = np.setdiff1d(np.arange(len(b)), keep); keep = np.concatenate([keep, rest[-spare:]])
    keep = np.sort(keep)[-M:]
    return b[keep], yb[keep]

def init_buffer(tr0, rng):
    if POLICY == "recent": return rng.choice(tr0, min(M, len(tr0)), replace=False)
    yt = y[tr0]; cls = np.unique(yt); quota = M // len(cls); out = []
    for c in cls:
        idx = tr0[yt == c]; out.append(rng.choice(idx, min(quota, len(idx)), replace=False))
    out = np.concatenate(out)
    if len(out) < M: out = np.concatenate([out, rng.choice(np.setdiff1d(tr0, out), M - len(out), replace=False)])
    return out

def poison(lab, yv, rng):
    if EPS <= 0: return yv
    yv = yv.copy(); att = np.where(yv != NORM_LBL)[0]
    flip = rng.choice(att, int(EPS * len(att)), replace=False) if len(att) else []
    yv[flip] = NORM_LBL; return yv

# ---------- run ----------
def run(seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    all_m = ["AGT-static", "AGT-rehearsal", "AGT-finetune", "AGT-retrain", "MLP-static", "MLP-rehearsal",
             "XGB-static", "XGB-continued", "XGB-sliding", "XGB-retrain", "INSOMNIA-style"]
    use = METHODS or all_m
    st = {}; timing = {m_: [] for m_ in use}
    # initial models on the first quarter
    agt0, tr0, info = initial_fit("AGT", seed, init); mlp0, _, _ = initial_fit("MLP", seed, init)
    st["AGT-static"] = agt0
    for nm in ["AGT-rehearsal", "AGT-finetune", "AGT-retrain"]: st[nm] = copy.deepcopy(agt0)
    for nm in ["MLP-static", "MLP-rehearsal", "INSOMNIA-style"]: st[nm] = copy.deepcopy(mlp0)
    opts = {nm: torch.optim.AdamW(st[nm].parameters(), lr=ETA, weight_decay=1e-4)
            for nm in ["AGT-rehearsal", "AGT-finetune", "MLP-rehearsal", "INSOMNIA-style"]}
    xg0 = xgb_train(flat[tr0], y[tr0], seed=seed)
    xgbs = {"XGB-static": xg0, "XGB-continued": xg0, "XGB-sliding": xg0, "XGB-retrain": xg0}
    buf = init_buffer(tr0, rng); ybuf = y[buf]                       # replay buffer seeded from the past
    buf_ft = np.array([], dtype=int)
    ref = {"AGT-static": tr0, "AGT-rehearsal": tr0, "AGT-finetune": tr0, "AGT-retrain": tr0}
    lab_all = [tr0]; lab_y = [y[tr0]]
    ins_pool = buf.copy(); ins_y = y[ins_pool]
    rows = []
    for t, w in enumerate(windows, 1):
        if t > int(os.environ.get("MAXW", "99")): break
        r = dict(seed=seed, window=t)
        # ---- predict the whole window before any of its labels are revealed ----
        for nm in use:
            if nm.startswith("AGT"): p, ti = infer(st[nm], ref[nm], w)
            elif nm.startswith("MLP") or nm == "INSOMNIA-style": p, ti = infer(st[nm], ref["AGT-static"][:50], w)
            else:
                t0 = time.time(); p = xgb_proba(xgbs[nm], flat[w]); ti = time.time() - t0
            r[nm] = wscore(w, p); r[nm]["infer_s"] = ti
            if nm == "INSOMNIA-style": p_ins = p
        # ---- reveal labels for a fraction rho of the window ----
        A = rng.choice(w, int(RHO * len(w)), replace=False); yA = poison(A, y[A], rng)
        # AGT rehearsal (proposed): graph over A_t and the buffer, S steps, refresh buffer
        if "AGT-rehearsal" in use:
            sel = np.concatenate([A, buf]); yv = np.concatenate([yA, ybuf])
            tg, ts_ = update(st["AGT-rehearsal"], sel, opts["AGT-rehearsal"], yv=yv)
            t0 = time.time(); buf_new, ybuf_new = refresh(buf, ybuf, A, yA); tb = time.time() - t0
            timing["AGT-rehearsal"].append(dict(graph_s=tg, steps_s=ts_, buffer_s=tb, n_nodes=len(sel)))
            ref["AGT-rehearsal"] = sel
        if "AGT-finetune" in use:
            tg, ts_ = update(st["AGT-finetune"], A, opts["AGT-finetune"], yv=yA)
            timing["AGT-finetune"].append(dict(graph_s=tg, steps_s=ts_)); ref["AGT-finetune"] = A
        if "MLP-rehearsal" in use:
            sel = np.concatenate([A, buf]); yv = np.concatenate([yA, ybuf])
            tg, ts_ = update(st["MLP-rehearsal"], sel, opts["MLP-rehearsal"], yv=yv)
            timing["MLP-rehearsal"].append(dict(graph_s=0.0, steps_s=ts_))
        lab_all.append(A); lab_y.append(yA)
        if "AGT-retrain" in use:
            allsel = np.concatenate(lab_all); t0 = time.time()
            g, _ = make_graph(D, allsel, k=K); g.y = torch.tensor(np.concatenate(lab_y), dtype=torch.long)
            m_, hp = neural("AGT"); m_, inf_ = train_net(m_, g, None, None, epochs=150, lr=hp["lr"], C=C, seed=seed)
            st["AGT-retrain"] = m_; ref["AGT-retrain"] = allsel
            timing["AGT-retrain"].append(dict(total_s=time.time() - t0, n_nodes=len(allsel)))
        if "XGB-continued" in use:
            t0 = time.time(); sel = np.concatenate([A, buf]); yv = np.concatenate([yA, ybuf])
            xgbs["XGB-continued"] = xgb_train(flat[sel], yv, prev=xgbs["XGB-continued"], rounds=50, seed=seed)
            timing["XGB-continued"].append(dict(total_s=time.time() - t0, n_trees=xgbs["XGB-continued"].num_boosted_rounds()))
        if "XGB-sliding" in use:
            t0 = time.time(); sel = np.concatenate([A, buf]); yv = np.concatenate([yA, ybuf])
            xgbs["XGB-sliding"] = xgb_train(flat[sel], yv, seed=seed)
            timing["XGB-sliding"].append(dict(total_s=time.time() - t0, n=len(sel)))
        if "XGB-retrain" in use:
            t0 = time.time(); allsel = np.concatenate(lab_all)
            xgbs["XGB-retrain"] = xgb_train(flat[allsel], np.concatenate(lab_y), seed=seed)
            timing["XGB-retrain"].append(dict(total_s=time.time() - t0, n=len(allsel)))
        if "INSOMNIA-style" in use:
            # active labelling: the same budget is spent on the most uncertain flows
            unc = 1 - p_ins.max(1); order = np.argsort(-unc); Ai = w[order[:len(A)]]; Ui = w[order[len(A):]]
            yAi = poison(Ai, y[Ai], rng)
            ins_pool = np.concatenate([ins_pool, Ai])[-M:]; ins_y = np.concatenate([ins_y, yAi])[-M:]
            # label estimation for the rest of the window: nearest class centroid of the labelled pool
            t0 = time.time()
            cls_ = np.unique(ins_y); cent = np.stack([flat[ins_pool[ins_y == c]].mean(0) for c in cls_])
            dist = ((flat[Ui][:, None, :] - cent[None]) ** 2).sum(-1); yU = cls_[dist.argmin(1)]
            sel = np.concatenate([Ai, Ui]); yv = np.concatenate([yAi, yU])
            tg, ts_ = update(st["INSOMNIA-style"], sel, opts["INSOMNIA-style"], yv=yv)
            timing["INSOMNIA-style"].append(dict(total_s=time.time() - t0))
        if "AGT-rehearsal" in use or "MLP-rehearsal" in use or "XGB-sliding" in use or "XGB-continued" in use:
            buf, ybuf = (buf_new, ybuf_new) if "AGT-rehearsal" in use else refresh(buf, ybuf, A, yA)
        rows.append(r)
        print(seed, t, {nm: round(r[nm]["F1"], 3) for nm in use}, flush=True)
    return rows, timing

if __name__ == "__main__":
    if not PSEUDO: save_json(wmeta(), os.path.join(ODIR, "windows_meta.json" if (STRIDE, INIT_FRAC, T) == (4, 0.25, 15) else f"windows_meta_{TAG}.json"))
    for seed in SEEDS:
        rows, timing = run(seed)
        save_json(dict(rows=rows, timing=timing, cfg=dict(task=TASK, rho=RHO, M=M, eps=EPS, S=S, eta=ETA, policy=POLICY, pseudo=PSEUDO, T=T,
                  stride=STRIDE, init_frac=INIT_FRAC, k=K)), os.path.join(ODIR, f"{TAG}_seed{seed}.json"))
