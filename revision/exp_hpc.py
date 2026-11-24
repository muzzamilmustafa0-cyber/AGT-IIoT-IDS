# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Computational study: k-NN construction (CPU/GPU, exact/approximate, thread scaling), real-time throughput,
# retraining-cost growth and GPU memory breakdown on the full X-IIoTID corpus.
import sys, faiss, platform
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from common import *

ODIR = os.path.join(OUT, "hpc"); os.makedirs(ODIR, exist_ok=True)
HPS = {k: v["best"] for k, v in json.load(open(os.path.join(OUT, "stationary", "X-IIoTID", "tuning.json"))).items()}
K = 10

def full_data(seed=0):
    df = load_raw("X-IIoTID", keep=("Timestamp",))
    classes = sorted(df["y_str"].unique()); yv = df["y_str"].map({c: i for i, c in enumerate(classes)}).values
    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=seed, stratify=yv)
    pr = Prep().fit(df.iloc[itr]); num, cat = pr.transform(df)
    return dict(df=df, y=yv, classes=classes, itr=itr, ite=ite, prep=pr, num=num, cat=cat, flat=pr.flat(num, cat))

def log(path, r):
    with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n")
    print(r, flush=True)

def stage_arrival():
    df = load_raw("X-IIoTID", keep=("Timestamp",)); ts = pd.to_numeric(df["__Timestamp"], errors="coerce").dropna().astype(np.int64)
    per_s = ts.value_counts()
    active = per_s.values
    r = dict(records=int(len(ts)), active_seconds=int(len(per_s)), mean_per_active_s=float(active.mean()),
             p99_per_s=float(np.percentile(active, 99)), max_per_s=int(active.max()),
             max_per_min=int(ts.floordiv(60).value_counts().max()), max_per_10min=int(ts.floordiv(600).value_counts().max()))
    save_json(r, os.path.join(ODIR, "arrival.json")); print(r)

def stage_knn():
    D = full_data(); X = D["flat"][D["itr"]]; path = os.path.join(ODIR, "knn_scaling.jsonl")
    hw = dict(cpu=platform.processor(), threads=os.cpu_count(), gpu=torch.cuda.get_device_name(0))
    save_json(hw, os.path.join(ODIR, "hardware.json"))
    rng = np.random.default_rng(0)
    for N in [10000, 20000, 40000, 80000, 160000, 320000, len(X)]:
        sub = X[rng.choice(len(X), N, replace=False)] if N < len(X) else X
        knn(sub[:2000], sub[:2000], K)                                  # GPU warm-up
        t0 = time.time(); ex = knn(sub, sub, K, exclude_self=True); tg = time.time() - t0
        log(path, dict(N=N, method="GPU exact (chunked GEMM)", threads=0, seconds=tg, recall=1.0))
        for th in [1, 16]:
            if th == 1 and N > 160000: continue
            faiss.omp_set_num_threads(th); idx = faiss.IndexFlatL2(sub.shape[1]); idx.add(sub)
            t0 = time.time(); _, I = idx.search(sub, K + 1); tc = time.time() - t0
            log(path, dict(N=N, method="CPU exact (FAISS flat)", threads=th, seconds=tc, recall=1.0))
        if N <= 160000:
            t0 = time.time(); NearestNeighbors(n_neighbors=K + 1, n_jobs=16).fit(sub).kneighbors(sub); ts_ = time.time() - t0
            log(path, dict(N=N, method="CPU exact (scikit-learn)", threads=16, seconds=ts_, recall=1.0))
        faiss.omp_set_num_threads(16); h = faiss.IndexHNSWFlat(sub.shape[1], 32); h.hnsw.efSearch = 64
        t0 = time.time(); h.add(sub); _, Ih = h.search(sub, K + 1); th_ = time.time() - t0
        # recall against the exact GPU neighbours (self removed)
        rec = np.mean([len(set(Ih[i][Ih[i] != i][:K]) & set(ex[i])) / K for i in range(0, N, max(1, N // 5000))])
        log(path, dict(N=N, method="CPU approximate (FAISS HNSW)", threads=16, seconds=th_, recall=float(rec)))
    # strong scaling of the exact CPU search with the number of threads
    sub = X[rng.choice(len(X), 160000, replace=False)]
    for th in [1, 2, 4, 8, 16]:
        faiss.omp_set_num_threads(th); idx = faiss.IndexFlatL2(sub.shape[1]); idx.add(sub)
        t0 = time.time(); idx.search(sub, K + 1); log(os.path.join(ODIR, "thread_scaling.jsonl"), dict(N=160000, threads=th, seconds=time.time() - t0))

def stage_throughput():
    # end-to-end scoring of new flows: k-NN query against the reference set + forward pass
    D = split_stationary("X-IIoTID", 0); C = len(D["classes"])
    gtr, nb = make_graph(D, D["itr"]); gva, _ = make_graph(D, D["itr"], D["iva"], nbr_ref=nb)
    m = build_model("AGT", D, HPS["AGT"]); m, _ = train_net(m, gtr, gva, D["y"][D["iva"]], epochs=150, lr=HPS["AGT"]["lr"], C=C)
    path = os.path.join(ODIR, "throughput.jsonl")
    big = np.concatenate([D["ite"]] * 7)                                 # up to ~100k queries
    for dev in ["cuda", "cpu"]:
        md = copy.deepcopy(m).to(dev).eval()
        for B in [1000, 10000, 100000]:
            if dev == "cpu" and B > 10000: continue
            q = big[:B]; times = []
            for rep in range(3):
                t0 = time.time()
                if dev == "cuda": g, _ = make_graph(D, D["itr"], q, nbr_ref=nb); tk = time.time() - t0
                else:
                    faiss.omp_set_num_threads(16); idx = faiss.IndexFlatL2(D["flat"].shape[1]); idx.add(D["flat"][D["itr"]])
                    _, I = idx.search(D["flat"][q], K); tk = time.time() - t0
                    g, _ = make_graph(D, D["itr"], q, nbr_ref=nb)
                g = g.to(dev)
                with torch.no_grad():
                    t1 = time.time(); md(g)
                    if dev == "cuda": torch.cuda.synchronize()
                    tf = time.time() - t1
                times.append((tk, tf))
            tk, tf = np.median([t[0] for t in times]), np.median([t[1] for t in times])
            log(path, dict(device=dev, batch=B, ref=len(D["itr"]), knn_s=tk, forward_s=tf, total_s=tk + tf, flows_per_s=B / (tk + tf)))

def stage_retrain():
    # cost of keeping a model current: full XGBoost refit vs. constant-size rehearsal update
    D = full_data(); path = os.path.join(ODIR, "retrain_growth.jsonl"); hp = HPS["XGBoost"]; C = len(D["classes"])
    rng = np.random.default_rng(0)
    for N in [50000, 100000, 200000, 400000, len(D["itr"])]:
        sel = rng.choice(D["itr"], N, replace=False); Xs, ys = D["flat"][sel], D["y"][sel]
        hold = rng.choice(D["ite"], 20000, replace=False)
        for dev in ["cpu", "cuda"]:
            # scikit-learn interface: its GPU path was checked to agree with the CPU model
            m = XGBClassifier(n_estimators=600, max_depth=hp["max_depth"], learning_rate=hp["eta"], subsample=0.9,
                              colsample_bytree=0.9, tree_method="hist", device=dev, n_jobs=16, random_state=0, verbosity=0)
            t0 = time.time(); m.fit(Xs, ys); tt = time.time() - t0
            f1h = f1_score(D["y"][hold], m.predict(D["flat"][hold]), average="macro")
            log(path, dict(model="XGBoost full refit", device=dev, N=N, seconds=tt, check_macroF1=f1h))
    # AGT rehearsal update (graph over 2,050 labelled + 2,500 buffered flows, 15 steps)
    Ds = split_stationary("X-IIoTID", 0); C = len(Ds["classes"])
    m = build_model("AGT", Ds, HPS["AGT"]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1.2e-3)
    for rep in range(5):
        sel = rng.choice(Ds["itr"], 4550, replace=False)
        t0 = time.time(); g, _ = make_graph(Ds, sel); g = g.to(DEV); tg = time.time() - t0
        w = class_weights(g.y.cpu().numpy(), C); t1 = time.time()
        for _ in range(15):
            opt.zero_grad(); F.cross_entropy(m(g), g.y, weight=w).backward(); opt.step()
        torch.cuda.synchronize(); log(path, dict(model="AGT rehearsal update", device="cuda", N=4550, graph_s=tg, steps_s=time.time() - t1, seconds=time.time() - t0))

def stage_memory():
    # peak GPU memory of AGT training/inference at 80k records, split by phase
    D = split_stationary("X-IIoTID", 0); C = len(D["classes"]); path = os.path.join(ODIR, "memory.jsonl")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    gtr, nb = make_graph(D, D["itr"]); torch.cuda.synchronize()
    log(path, dict(phase="k-NN graph construction (56k nodes)", peak_alloc_MB=torch.cuda.max_memory_allocated() / 1e6,
                   reserved_MB=torch.cuda.memory_reserved() / 1e6))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build_model("AGT", D, HPS["AGT"]).to(DEV); g = gtr.to(DEV); w = class_weights(D["y"][D["itr"]], C)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3); base = torch.cuda.memory_allocated()
    for _ in range(3):
        opt.zero_grad(); F.cross_entropy(m(g), g.y, weight=w).backward(); opt.step()
    torch.cuda.synchronize()
    log(path, dict(phase="training step (full batch, 56k nodes)", peak_alloc_MB=torch.cuda.max_memory_allocated() / 1e6,
                   static_MB=base / 1e6, reserved_MB=torch.cuda.memory_reserved() / 1e6, params=nparams(m)))
    gte, _ = make_graph(D, D["itr"], D["ite"], nbr_ref=nb); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): m.eval(); m(gte.to(DEV))
    torch.cuda.synchronize()
    log(path, dict(phase="inference (56k reference + 16k queries)", peak_alloc_MB=torch.cuda.max_memory_allocated() / 1e6,
                   reserved_MB=torch.cuda.memory_reserved() / 1e6))

if __name__ == "__main__":
    {"arrival": stage_arrival, "knn": stage_knn, "throughput": stage_throughput,
     "retrain": stage_retrain, "memory": stage_memory}[sys.argv[1]]()
