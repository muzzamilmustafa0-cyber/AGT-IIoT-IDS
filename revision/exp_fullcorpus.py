# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Full-corpus X-IIoTID (820k flows): GPU k-NN graph + mini-batch two-hop training of AGT and baselines.
import sys
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from common import *

ODIR = os.path.join(OUT, "fullcorpus"); os.makedirs(ODIR, exist_ok=True)
HPS = {k: v["best"] for k, v in json.load(open(os.path.join(OUT, "stationary", "X-IIoTID", "tuning.json"))).items()}
K = 10; BS = int(os.environ.get("BS", "4096")); EPOCHS = int(os.environ.get("FEPOCHS", "30"))

def subgraph(T_nodes, nbr_T, nbr_ref, X_num, X_cat, yall, ref_ids, qry_ids=None):
    # T_nodes: target ids; nbr_T: their k reference neighbours; two hops through the reference graph
    h1 = nbr_T.reshape(-1); h2 = nbr_ref[h1].reshape(-1)
    if qry_ids is None:
        allg = torch.cat([T_nodes, h1, h2])                         # ids in reference numbering
        uniq, inv = torch.unique(allg, return_inverse=True)
        nT, n1 = len(T_nodes), len(h1)
        iT, i1, i2 = inv[:nT], inv[nT:nT + n1], inv[nT + n1:]
        gid = ref_ids[uniq]; n_nodes = len(uniq)
    else:
        allg = torch.cat([h1, h2]); uniq, inv = torch.unique(allg, return_inverse=True)
        n1 = len(h1); i1, i2 = inv[:n1], inv[n1:]
        iT = torch.arange(len(T_nodes), device=DEV) + len(uniq)
        gid = torch.cat([ref_ids[uniq], qry_ids]); n_nodes = len(uniq) + len(T_nodes)
    src = torch.cat([i1, i2]); dst = torch.cat([iT.repeat_interleave(K), i1.repeat_interleave(K)])
    e = torch.unique(src * n_nodes + dst); src, dst = e // n_nodes, e % n_nodes
    g = Data(edge_index=torch.stack([src, dst]))
    g.x_num = X_num[gid]; g.x_cat = X_cat[gid]; g.y = yall[gid]
    ctx = torch.zeros(n_nodes, dtype=torch.bool, device=DEV); ctx[i1] = True
    if qry_ids is None: ctx[iT] = True
    g.ctx = ctx; g.num_nodes = n_nodes; g.tgt = iT
    return g

def run(seed, models):
    path = os.path.join(ODIR, "runs.jsonl")
    df = load_raw("X-IIoTID"); classes = sorted(df["y_str"].unique()); C = len(classes)
    y = df["y_str"].map({c: i for i, c in enumerate(classes)}).values; idx = np.arange(len(df))
    itv, ite = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    itr, iva = train_test_split(itv, test_size=0.125, random_state=seed, stratify=y[itv])
    pr = Prep().fit(df.iloc[itr]); num, cat = pr.transform(df); flat = pr.flat(num, cat)
    D = dict(num=num, cat=cat, classes=classes, prep=pr)
    yte = y[ite]
    if any(m_ in models for m_ in ["AGT", "GAT", "TransformerConv", "MLP"]):
        t0 = time.time(); nbr_tr = knn(flat[itr], flat[itr], K, exclude_self=True); t_tr = time.time() - t0
        t0 = time.time(); nbr_va = knn(flat[itr], flat[iva], K); nbr_te = knn(flat[itr], flat[ite], K); t_q = time.time() - t0
        print(f"seed {seed}: k-NN train graph {t_tr:.1f}s, queries {t_q:.1f}s", flush=True)
        X_num = torch.tensor(num, device=DEV); X_cat = torch.tensor(cat, device=DEV); Y = torch.tensor(y, device=DEV)
        ref_ids = torch.tensor(itr, device=DEV); NB = torch.tensor(nbr_tr, device=DEV)
    for kind in models:
        t0 = time.time()
        if kind == "XGBoost":
            hp = HPS["XGBoost"]
            p = dict(objective="multi:softprob", num_class=C, max_depth=hp["max_depth"], eta=hp["eta"], subsample=0.9,
                     colsample_bytree=0.9, tree_method="hist", device="cpu", nthread=16, seed=seed, verbosity=0)  # native-API GPU path gave constant predictions on this GPU
            b = xgb.train(p, xgb.DMatrix(flat[itr], label=y[itr]), num_boost_round=600)
            tt = time.time() - t0; prb = b.predict(xgb.DMatrix(flat[ite])).reshape(len(ite), C); extra = {}
        elif kind == "RandomForest":
            hp = HPS["RandomForest"]
            m = RandomForestClassifier(n_estimators=300, max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                                       class_weight=hp["cw"], n_jobs=16, random_state=seed).fit(flat[itr], y[itr])
            tt = time.time() - t0; prb = np.zeros((len(ite), C)); prb[:, m.classes_] = m.predict_proba(flat[ite]); extra = {}
        else:
            hp = HPS[kind]; torch.manual_seed(seed)
            m = build_model(kind, D, hp).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=5e-4)
            steps = EPOCHS * int(np.ceil(len(itr) / BS)); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
            w = class_weights(y[itr], C); best, bs_ = -1, None
            torch.cuda.reset_peak_memory_stats()
            for ep in range(EPOCHS):
                m.train(); perm = torch.randperm(len(itr), device=DEV)
                for s in range(0, len(itr), BS):
                    T_ = perm[s:s + BS]; g = subgraph(T_, NB[T_], NB, X_num, X_cat, Y, ref_ids)
                    opt.zero_grad(); out = m(g)[g.tgt]; loss = F.cross_entropy(out, g.y[g.tgt], weight=w)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step(); sch.step()
                pv = score(m, iva, nbr_va, NB, X_num, X_cat, Y, ref_ids, C).argmax(1)
                v = f1_score(y[iva], pv, average="macro", zero_division=0)
                print(f"  {kind} epoch {ep + 1} val macroF1 {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if v > best: best, bs_ = v, copy.deepcopy(m.state_dict())
            tt = time.time() - t0; m.load_state_dict(bs_)
            t1 = time.time(); prb = score(m, ite, nbr_te, NB, X_num, X_cat, Y, ref_ids, C); ti = time.time() - t1
            extra = dict(infer_s=ti, peak_MB=torch.cuda.max_memory_allocated() / 1e6, best_val=best, epochs=EPOCHS, batch=BS,
                         knn_train_s=t_tr, knn_query_s=t_q)
        r = dict(seed=seed, model=kind, n_train=len(itr), n_val=len(iva), n_test=len(ite), train_s=tt, **metrics(yte, prb, C),
                 perclass=perclass_f1(yte, prb.argmax(1), C).tolist(), support=np.bincount(yte, minlength=C).tolist(),
                 classes=classes, **extra)
        with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n")
        print(seed, kind, {k: round(r[k], 4) for k in ["Accuracy", "F1", "MacroF1", "MCC"]}, round(tt, 1), flush=True)

@torch.no_grad()
def score(m, q, nbr_q, NB, X_num, X_cat, Y, ref_ids, C):
    m.eval(); out = []; NQ = torch.tensor(nbr_q, device=DEV); Q = torch.tensor(q, device=DEV)
    for s in range(0, len(q), BS):
        g = subgraph(Q[s:s + BS], NQ[s:s + BS], NB, X_num, X_cat, Y, ref_ids, qry_ids=Q[s:s + BS])
        out.append(F.softmax(m(g)[g.tgt], 1).cpu().numpy())
    return np.vstack(out)

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["XGBoost", "RandomForest", "MLP", "TransformerConv", "GAT", "AGT"]
    for s in seeds: run(s, models)
