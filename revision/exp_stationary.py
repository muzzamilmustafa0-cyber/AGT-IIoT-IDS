# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Stationary experiments: tuning on validation, multi-seed main table, ablation, sensitivity, noise, importance
import sys, itertools
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DS = os.environ.get("DS", "X-IIoTID")
SAMPLE = int(os.environ.get("SAMPLE", "80000"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))
ODIR = os.path.join(OUT, "stationary", DS.replace(" ", "_")); os.makedirs(ODIR, exist_ok=True)
NEURAL = ["AGT", "TransformerConv", "GAT", "GCN", "GraphSAGE", "GPS", "MLP"]
TAB = ["LogReg", "RandomForest", "XGBoost"]

GRIDS = {
    "AGT": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], heads=[4, 8]),
    "TransformerConv": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], heads=[4, 8]),
    "GAT": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], heads=[4, 8]),
    "GCN": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], hid=[128, 256]),
    "GraphSAGE": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], hid=[128, 256]),
    "GPS": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], hid=[64, 128]),
    "MLP": dict(lr=[1e-3, 3e-3], drop=[0.1, 0.2], hid=[256, 512], layers=[2, 3]),
    "LogReg": dict(C=[0.1, 1.0, 10.0], cw=[None, "balanced"]),
    "RandomForest": dict(max_depth=[None, 30], min_samples_leaf=[1, 2], cw=[None, "balanced_subsample"]),
    "XGBoost": dict(max_depth=[6, 10], eta=[0.05, 0.1], sw=[False, True]),
}

def grid(d):
    keys = list(d); return [dict(zip(keys, v)) for v in itertools.product(*[d[k] for k in keys])]

def graphs(D, k=10, feats=None):
    gtr, nb = make_graph(D, D["itr"], k=k, feats=feats)
    gva, _ = make_graph(D, D["itr"], D["iva"], k=k, nbr_ref=nb, feats=feats)
    gte, _ = make_graph(D, D["itr"], D["ite"], k=k, nbr_ref=nb, feats=feats)
    return gtr, gva, gte

def predict_any(model, kind, D, g, qry, k=10, feats=None):
    if kind != "GPS": return predict(model, g)
    # GPS global attention would mix test nodes: score queries in small chunks against the reference set
    out = []
    for s in range(0, len(qry), 2000):
        gc, _ = make_graph(D, D["itr"], qry[s:s + 2000], k=k, nbr_ref=D["_nb"], feats=feats)
        out.append(predict(model, gc))
    return np.vstack(out)

def run_tab(kind, hp, D, Xtr, ytr, Xev, seed):
    C = len(D["classes"]); t0 = time.time()
    if kind == "LogReg":
        m = LogisticRegression(C=hp["C"], class_weight=hp["cw"], max_iter=2000, n_jobs=16)
        m.fit(Xtr, ytr)
    elif kind == "RandomForest":
        m = RandomForestClassifier(n_estimators=300, max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                                   class_weight=hp["cw"], n_jobs=16, random_state=seed)
        m.fit(Xtr, ytr)
    else:
        m = XGBClassifier(n_estimators=600, max_depth=hp["max_depth"], learning_rate=hp["eta"], subsample=0.9,
                          colsample_bytree=0.9, tree_method="hist", device="cuda", random_state=seed, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr) if hp["sw"] else None)
    tt = time.time() - t0
    pr = np.zeros((len(Xev), C)); pr[:, m.classes_] = m.predict_proba(Xev)
    return m, pr, tt

def run_neural(kind, hp, D, gtr, gva, gte, seed, epochs=EPOCHS, cw=True):
    C = len(D["classes"])
    m = build_model(kind, D, hp)
    m, info = train_net(m, gtr, gva, D["y"][D["iva"]], epochs=epochs, lr=hp.get("lr", 3e-3), cw=cw, C=C, seed=seed)
    return m, info

# ---------------- stages ----------------
def stage_tune():
    D = split_stationary(DS, 0, SAMPLE); C = len(D["classes"])
    gtr, gva, gte = graphs(D); D["_nb"] = None
    _, D["_nb"] = make_graph(D, D["itr"])
    yva = D["y"][D["iva"]]
    res = {}
    for kind in TAB + NEURAL:
        rows = []
        for hp in grid(GRIDS[kind]):
            if kind in TAB:
                _, pr, tt = run_tab(kind, hp, D, D["flat"][D["itr"]], D["y"][D["itr"]], D["flat"][D["iva"]], 0)
            else:
                m, info = run_neural(kind, hp, D, gtr, gva, gte, 0)
                pr = predict_any(m, kind, D, gva, D["iva"]); tt = info["train_s"]
            v = f1_score(yva, pr.argmax(1), average="macro", zero_division=0)
            rows.append(dict(hp=hp, val_macroF1=v, train_s=tt)); print(kind, hp, round(v, 4), round(tt, 1), flush=True)
        best = max(rows, key=lambda r: r["val_macroF1"])
        res[kind] = dict(best=best["hp"], best_val=best["val_macroF1"], all=rows)
        save_json(res, os.path.join(ODIR, "tuning.json"))

def best_hp():
    t = json.load(open(os.path.join(OUT, "stationary", "X-IIoTID", "tuning.json")))
    return {k: v["best"] for k, v in t.items()}

def stage_main(seeds):
    hps = best_hp(); path = os.path.join(ODIR, "main_runs.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path): r = json.loads(l); done.add((r["seed"], r["model"]))
    for seed in seeds:
        D = split_stationary(DS, seed, SAMPLE); C = len(D["classes"])
        t0 = time.time(); gtr, nb = make_graph(D, D["itr"]); gtime = time.time() - t0; D["_nb"] = nb
        gva, _ = make_graph(D, D["itr"], D["iva"], nbr_ref=nb); gte, _ = make_graph(D, D["itr"], D["ite"], nbr_ref=nb)
        yte = D["y"][D["ite"]]
        for kind in TAB + NEURAL:
            if (seed, kind) in done: continue
            hp = hps[kind]
            if kind in TAB:
                Xtrv = D["flat"][D["itr"]]
                _, pr, tt = run_tab(kind, hp, D, Xtrv, D["y"][D["itr"]], D["flat"][D["ite"]], seed); extra = {}
            else:
                m, info = run_neural(kind, hp, D, gtr, gva, gte, seed)
                t1 = time.time(); pr = predict_any(m, kind, D, gte, D["ite"]); inf = time.time() - t1
                tt = info["train_s"]; extra = dict(params=nparams(m), infer_s=inf, best_val=info["best_val"])
                if kind == "AGT" and seed == seeds[0]: save_json(info["hist"], os.path.join(ODIR, f"agt_hist_seed{seed}.json"))
            r = dict(seed=seed, model=kind, hp=hp, train_s=tt, graph_s=gtime, **metrics(yte, pr, C),
                     perclass=perclass_f1(yte, pr.argmax(1), C).tolist(), support=np.bincount(yte, minlength=C).tolist(),
                     classes=D["classes"], pred=pr.argmax(1).tolist(), ytrue=yte.tolist(), **extra)
            with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n")
            print(seed, kind, {k: round(r[k], 4) for k in ["Accuracy", "F1", "MacroF1", "MCC"]}, round(tt, 1), flush=True)

def isab_ffn_width(D, hp):
    m = build_model("AGT", D, hp); n_isab = nparams(m.g); h = hp.get("hid", 128)
    return int(round((n_isab - h) / (2 * h + 1)))

ABL = [("Full AGT", {}, True),
       ("- global attention", {"glob": "none"}, True),
       ("global = node-wise landmark memory (submitted)", {"glob": "memory"}, True),
       ("global = parameter-matched FFN", {"glob": "ffn"}, True),
       ("- entity embeddings (ordinal codes)", {"embed": "ordinal"}, True),
       ("- class weighting", {}, False),
       ("one local layer", {"layers": 1}, True),
       ("- graph topology (no edges)", {"conv": "none"}, True)]

def stage_ablation(seeds):
    hp0 = best_hp()["AGT"]; path = os.path.join(ODIR, "ablation_runs.jsonl"); done = set()
    if os.path.exists(path):
        for l in open(path): r = json.loads(l); done.add((r["seed"], r["variant"]))
    for seed in seeds:
        D = split_stationary(DS, seed, SAMPLE); C = len(D["classes"])
        gtr, gva, gte = graphs(D); yte = D["y"][D["ite"]]
        for name, mod, cw in ABL:
            if (seed, name) in done: continue
            hp = dict(hp0, **mod)
            if mod.get("glob") == "ffn": hp["ffn_width"] = isab_ffn_width(D, hp0)
            if "conv" in mod:
                m = GraphNet(D["prep"].cards, D["num"].shape[1], C, conv="none", glob="isab", hid=128,
                             heads=hp["heads"], L=hp.get("L", 64), drop=hp["drop"])
                m, info = train_net(m, gtr, gva, D["y"][D["iva"]], epochs=EPOCHS, lr=hp["lr"], C=C, seed=seed)
            else:
                m, info = run_neural("AGT", hp, D, gtr, gva, gte, seed, cw=cw)
            pr = predict(m, gte)
            r = dict(seed=seed, variant=name, params=nparams(m), train_s=info["train_s"], **metrics(yte, pr, C),
                     perclass=perclass_f1(yte, pr.argmax(1), C).tolist())
            with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n")
            print(seed, name, nparams(m), {k: round(r[k], 4) for k in ["Accuracy", "F1", "MacroF1", "MCC"]}, flush=True)

def stage_sens(seeds):
    hp0 = best_hp()["AGT"]; path = os.path.join(ODIR, "sensitivity_runs.jsonl")
    for seed in seeds:
        D = split_stationary(DS, seed, SAMPLE); C = len(D["classes"])
        yva, yte = D["y"][D["iva"]], D["y"][D["ite"]]
        gtr, gva, gte = graphs(D)
        for L in [0, 16, 32, 64, 128, 256]:
            hp = dict(hp0, L=max(L, 1), glob="isab" if L > 0 else "none")
            m, info = run_neural("AGT", hp, D, gtr, gva, gte, seed)
            r = dict(seed=seed, param="L", value=L, val_macroF1=info["best_val"], train_s=info["train_s"],
                     **metrics(yte, predict(m, gte), C))
            with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n"); print(r["param"], L, round(r["MacroF1"], 4), flush=True)
        for k in [5, 10, 15, 20, 30]:
            t0 = time.time(); g1, g2, g3 = graphs(D, k=k); gs = time.time() - t0
            m, info = run_neural("AGT", hp0, D, g1, g2, g3, seed)
            r = dict(seed=seed, param="k", value=k, val_macroF1=info["best_val"], train_s=info["train_s"], graph_s=gs,
                     **metrics(yte, predict(m, g3), C))
            with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n"); print("k", k, round(r["MacroF1"], 4), flush=True)

def stage_noise(seeds):
    hps = best_hp(); path = os.path.join(ODIR, "noise_runs.jsonl")
    for seed in seeds:
        D = split_stationary(DS, seed, SAMPLE); C = len(D["classes"]); yte = D["y"][D["ite"]]
        gtr, nb = make_graph(D, D["itr"]); D["_nb"] = nb
        gva, _ = make_graph(D, D["itr"], D["iva"], nbr_ref=nb)
        models = {}
        for kind in ["AGT", "GAT", "MLP"]:
            models[kind], _ = run_neural(kind, hps[kind], D, gtr, gva, None, seed)
        xgb, _, _ = run_tab("XGBoost", hps["XGBoost"], D, D["flat"][D["itr"]], D["y"][D["itr"]], D["flat"][D["ite"]], seed)
        rf, _, _ = run_tab("RandomForest", hps["RandomForest"], D, D["flat"][D["itr"]], D["y"][D["itr"]], D["flat"][D["ite"]], seed)
        rng = np.random.default_rng(seed + 1000)
        n_num = D["num"].shape[1]
        for sig in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            num = D["num"].copy(); num[D["ite"]] += rng.normal(0, sig, size=(len(D["ite"]), n_num)).astype(np.float32)
            Dn = dict(D, num=num); flatn = D["prep"].flat(num, D["cat"]); Dn["flat"] = flatn
            gte, _ = make_graph(Dn, D["itr"], D["ite"], nbr_ref=nb)       # test flows re-query the clean reference set
            res = {}
            for kind, m in models.items():
                pr = predict(m, gte); res[kind] = metrics(yte, pr, C); res[kind]["perclass"] = perclass_f1(yte, pr.argmax(1), C).tolist()
            for kind, m in [("XGBoost", xgb), ("RandomForest", rf)]:
                pr = np.zeros((len(yte), C)); pr[:, m.classes_] = m.predict_proba(flatn[D["ite"]]); res[kind] = metrics(yte, pr, C)
            with open(path, "a") as f: f.write(json.dumps(dict(seed=seed, sigma=sig, res=res, classes=D["classes"]), default=float) + "\n")
            print(seed, sig, {k: round(v["F1"], 4) for k, v in res.items()}, flush=True)

def stage_importance(seed=0, reps=3):
    hps = best_hp(); D = split_stationary(DS, seed, SAMPLE); C = len(D["classes"]); yte = D["y"][D["ite"]]
    gtr, nb = make_graph(D, D["itr"]); D["_nb"] = nb; gva, _ = make_graph(D, D["itr"], D["iva"], nbr_ref=nb)
    m, _ = run_neural("AGT", hps["AGT"], D, gtr, gva, None, seed)
    xgb, _, _ = run_tab("XGBoost", hps["XGBoost"], D, D["flat"][D["itr"]], D["y"][D["itr"]], D["flat"][D["ite"]], seed)
    pr_cols = D["prep"]
    gte, _ = make_graph(D, D["itr"], D["ite"], nbr_ref=nb)
    base_agt = f1_score(yte, predict(m, gte).argmax(1), average="macro")
    base_xgb = f1_score(yte, xgb.predict(D["flat"][D["ite"]]), average="macro")
    feats = [("num", j, c) for j, c in enumerate(pr_cols.num)] + [("cat", j, c) for j, c in enumerate(pr_cols.cat)]
    rng = np.random.default_rng(seed); rows = []
    for typ, j, name in feats:
        da, dx, dfix = [], [], []
        for r in range(reps):
            perm = rng.permutation(len(D["ite"]))
            num = D["num"].copy(); cat = D["cat"].copy()
            if typ == "num": num[D["ite"], j] = num[D["ite"][perm], j]
            else: cat[D["ite"], j] = cat[D["ite"][perm], j]
            Dp = dict(D, num=num, cat=cat); Dp["flat"] = pr_cols.flat(num, cat)
            g_re, _ = make_graph(Dp, D["itr"], D["ite"], nbr_ref=nb)                      # neighbours re-queried
            da.append(base_agt - f1_score(yte, predict(m, g_re).argmax(1), average="macro"))
            g_fix = gte.clone(); g_fix.x_num = torch.tensor(num[np.concatenate([D["itr"], D["ite"]])]); g_fix.x_cat = torch.tensor(cat[np.concatenate([D["itr"], D["ite"]])])
            dfix.append(base_agt - f1_score(yte, predict(m, g_fix).argmax(1), average="macro"))   # graph held fixed
            dx.append(base_xgb - f1_score(yte, xgb.predict(Dp["flat"][D["ite"]]), average="macro"))
        rows.append(dict(feature=name, agt_drop=float(np.mean(da)), agt_drop_sd=float(np.std(da)),
                         agt_drop_fixedgraph=float(np.mean(dfix)), xgb_perm_drop=float(np.mean(dx))))
        print(name, round(rows[-1]["agt_drop"], 4), round(rows[-1]["agt_drop_fixedgraph"], 4), round(rows[-1]["xgb_perm_drop"], 4), flush=True)
    # xgboost gain, aggregated from one-hot columns back to the original attribute
    gain = xgb.get_booster().get_score(importance_type="gain"); fnames = list(pr_cols.num) + [f"{c}={v}" for c in pr_cols.cat for v in range(pr_cols.cards[pr_cols.cat.index(c)])]
    agg = {}
    for key, v in gain.items():
        nm = fnames[int(key[1:])]; nm = nm.split("=")[0]; agg[nm] = agg.get(nm, 0) + v
    tot = sum(agg.values())
    for r in rows: r["xgb_gain_share"] = agg.get(r["feature"], 0) / tot
    # how predictive is Service alone of the class
    svc = D["df"]["Service"].astype(str).values if "Service" in D["df"].columns else None
    extra = {}
    if svc is not None:
        ct = pd.crosstab(svc, D["df"]["y_str"]); extra["service_class_purity"] = float(ct.max(1).sum() / ct.values.sum())
    save_json(dict(base_agt=base_agt, base_xgb=base_xgb, rows=rows, **extra), os.path.join(ODIR, "importance.json"))

if __name__ == "__main__":
    st = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(10))
    if st == "tune": stage_tune()
    elif st == "main": stage_main(seeds)
    elif st == "ablation": stage_ablation(seeds)
    elif st == "sens": stage_sens(seeds)
    elif st == "noise": stage_noise(seeds)
    elif st == "importance": stage_importance()
