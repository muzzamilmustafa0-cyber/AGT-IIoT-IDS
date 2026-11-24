# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Shared code for the revision experiments (inductive protocol, train-only preprocessing)
import os, time, copy, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, GCNConv, GATConv, SAGEConv, GINConv, GPSConv

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paths.json")))["data_root"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "X-IIoTID": dict(csv=ROOT + r"\xiiotid\X-IIoTID dataset.csv", label="class2",
                     drop=["class1", "class3", "Date", "Timestamp", "Scr_IP", "Des_IP"]),
    "Edge-IIoTset": dict(csv=ROOT + r"\datasets\edgeiiot\Edge-IIoTset dataset\Selected dataset for ML and DL\DNN-EdgeIIoT-dataset.csv",
                         label="Attack_type",
                         drop=["Attack_label", "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                               "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                               "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
                               "udp.port", "mqtt.msg"]),
    "WUSTL-IIoT-2021": dict(csv=ROOT + r"\datasets\wustl\wustl_iiot_2021.csv", label="Traffic",
                            drop=["StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId", "Target"]),
    "CICIoT2023": dict(csv=ROOT + r"\datasets\edge\CICIoT2023_8class.csv", label="label", drop=[]),
}

_cache = {}
def load_raw(name, keep=()):
    # returns dataframe with features + column "y_str"; columns in keep are retained as meta
    key = (name, tuple(keep))
    if key in _cache: return _cache[key].copy()
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv"], low_memory=False)
    df = df.rename(columns={cfg["label"]: "y_str"})
    df["y_str"] = df["y_str"].astype(str).str.strip()
    meta = {c: df[c].copy() for c in keep if c in df.columns}
    df = df.drop(columns=[c for c in cfg["drop"] if c in df.columns])
    for c, v in meta.items(): df["__" + c] = v
    vc = df["y_str"].value_counts(); df = df[df["y_str"].isin(vc[vc >= 10].index)]
    df = df.reset_index(drop=True)
    _cache[key] = df
    return df.copy()


class Prep:
    # all decisions and statistics come from the training partition only
    def fit(self, tr):
        self.num, self.cat, self.freq, self.vocab, self.med = [], [], {}, {}, {}
        cols = [c for c in tr.columns if c != "y_str" and not c.startswith("__")]
        for c in cols:
            col = tr[c]
            if col.dtype == bool or col.dtype.kind in "biuf":
                self.num.append(c); continue
            co = pd.to_numeric(col.astype(str).str.replace(",", ""), errors="coerce")
            if co.notna().mean() > 0.98: self.num.append(c)
            elif col.nunique() <= 64: self.cat.append(c)
            else:
                self.freq[c] = col.astype(str).value_counts(); self.num.append(c)
        X = self._numeric(tr, fit=True)
        self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd < 1e-9] = 1.0
        for c in self.cat:
            u = tr[c].fillna("NA").astype(str).unique()
            self.vocab[c] = {v: i for i, v in enumerate(u)}
        self.cards = [len(self.vocab[c]) + 1 for c in self.cat]   # last index = unseen category
        return self

    def _numeric(self, df, fit=False):
        out = np.zeros((len(df), len(self.num)), dtype=np.float64)
        for j, c in enumerate(self.num):
            if c in self.freq: v = df[c].astype(str).map(self.freq[c]).fillna(0).astype(float).values
            else: v = pd.to_numeric(df[c].astype(str).str.replace(",", "") if df[c].dtype == object else df[c], errors="coerce").astype(float).values
            v[~np.isfinite(v)] = np.nan
            if fit: self.med[c] = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            v = np.where(np.isnan(v), self.med[c], v)
            out[:, j] = v
        return out

    def transform(self, df):
        num = ((self._numeric(df) - self.mu) / self.sd).astype(np.float32)
        cat = np.zeros((len(df), len(self.cat)), dtype=np.int64)
        for j, c in enumerate(self.cat):
            cat[:, j] = df[c].fillna("NA").astype(str).map(self.vocab[c]).fillna(len(self.vocab[c])).astype(int).values
        return num, cat

    def onehot(self, cat):
        parts = [np.eye(k, dtype=np.float32)[cat[:, j]] for j, k in enumerate(self.cards)]
        return np.hstack(parts) if parts else np.zeros((len(cat), 0), np.float32)

    def flat(self, num, cat):
        # numeric + one-hot categorical: used for k-NN distances and tabular baselines
        return np.hstack([num, self.onehot(cat)]).astype(np.float32)


def split_stationary(name, seed, sample=80000, val=0.1, test=0.2, keep=()):
    df = load_raw(name, keep)
    if sample and len(df) > sample:
        df, _ = train_test_split(df, train_size=sample, random_state=seed, stratify=df["y_str"])
        df = df.reset_index(drop=True)
    classes = sorted(df["y_str"].unique())
    y = df["y_str"].map({c: i for i, c in enumerate(classes)}).values
    idx = np.arange(len(df))
    itv, ite = train_test_split(idx, test_size=test, random_state=seed, stratify=y)
    itr, iva = train_test_split(itv, test_size=val / (1 - test), random_state=seed, stratify=y[itv])
    pr = Prep().fit(df.iloc[itr])
    num, cat = pr.transform(df)
    flat = pr.flat(num, cat)
    return dict(df=df, y=y, classes=classes, itr=itr, iva=iva, ite=ite, prep=pr, num=num, cat=cat, flat=flat)


# ---------------- k-NN (exact, GPU, chunked) ----------------
@torch.no_grad()
def knn(ref, qry, k, exclude_self=False, mem=2.5e8):
    R = torch.from_numpy(np.ascontiguousarray(ref)).to(DEV)
    rn = (R * R).sum(1)
    bs = max(64, int(mem // max(1, len(ref))))
    out = np.empty((len(qry), k), dtype=np.int64)
    for s in range(0, len(qry), bs):
        Q = torch.from_numpy(np.ascontiguousarray(qry[s:s + bs])).to(DEV)
        d = (Q * Q).sum(1, keepdim=True) + rn[None, :] - 2.0 * (Q @ R.T)
        if exclude_self:
            r = torch.arange(len(Q), device=DEV); d[r, r + s] = float("inf")
        out[s:s + len(Q)] = d.topk(k, dim=1, largest=False).indices.cpu().numpy()
    if DEV.type == "cuda": torch.cuda.synchronize()
    return out


def make_graph(D, ref, qry=None, k=10, nbr_ref=None, feats=None):
    # nodes = ref (+ qry); every node receives edges from its k nearest REFERENCE nodes
    F_ = D["flat"] if feats is None else feats
    if nbr_ref is None: nbr_ref = knn(F_[ref], F_[ref], k, exclude_self=True)
    n_ref = len(ref)
    src = [nbr_ref.reshape(-1)]; dst = [np.repeat(np.arange(n_ref), k)]
    nodes = ref
    if qry is not None:
        nq = knn(F_[ref], F_[qry], k)
        src.append(nq.reshape(-1)); dst.append(n_ref + np.repeat(np.arange(len(qry)), k))
        nodes = np.concatenate([ref, qry])
    g = Data(edge_index=torch.tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long))
    g.x_num = torch.tensor(D["num"][nodes]); g.x_cat = torch.tensor(D["cat"][nodes])
    g.y = torch.tensor(D["y"][nodes], dtype=torch.long)
    ctx = torch.zeros(len(nodes), dtype=torch.bool); ctx[:n_ref] = True
    g.ctx = ctx; g.n_ref = n_ref; g.num_nodes = len(nodes)
    return g, nbr_ref


# ---------------- models ----------------
class TabEmbed(nn.Module):
    # entity embeddings for categorical fields + layer-normed numeric block -> hidden width
    def __init__(s, cards, n_num, d, mode="embed", cap=16):
        super().__init__(); s.mode = mode
        if mode == "embed":
            s.embs = nn.ModuleList([nn.Embedding(c, min(cap, max(3, int(np.ceil(np.sqrt(c)))))) for c in cards])
            ed = sum(e.embedding_dim for e in s.embs)
        else:
            s.cards = cards; ed = len(cards)          # ordinal codes, standardised
        s.ln = nn.LayerNorm(n_num)
        s.proj = nn.Sequential(nn.Linear(ed + n_num, d), nn.LayerNorm(d), nn.ReLU())
    def forward(s, xc, xn):
        if s.mode == "embed": e = [emb(xc[:, i]) for i, emb in enumerate(s.embs)]
        else:
            c = torch.tensor(s.cards, device=xc.device, dtype=torch.float)
            e = [(xc.float() / c - 0.5) * 3.46]       # ordinal code scaled to unit variance
        return s.proj(torch.cat(e + [s.ln(xn)], 1))


class LandmarkISAB(nn.Module):
    # two-stage landmark attention: landmarks pool the context nodes, nodes read the pooled summary
    def __init__(s, d, heads, L, dr):
        super().__init__()
        s.P = nn.Parameter(torch.randn(L, d) * 0.02)
        s.pool = nn.MultiheadAttention(d, heads, dropout=dr, batch_first=True)
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        s.read = nn.MultiheadAttention(d, heads, dropout=dr, batch_first=True)
    def summary(s, xc):
        S, _ = s.pool(s.P[None], xc[None], xc[None], need_weights=False)
        S = s.ln1(S + s.P[None]); return s.ln2(S + s.ff(S))
    def forward(s, x, ctx):
        S = s.summary(x[ctx])
        g, _ = s.read(x[None], S, S, need_weights=False)
        return g[0]


class LandmarkMemory(nn.Module):
    # original (submitted) module: nodes attend to fixed learned tokens only (no node-to-node exchange)
    def __init__(s, d, heads, L, dr):
        super().__init__(); s.P = nn.Parameter(torch.randn(L, d) * 0.02)
        s.read = nn.MultiheadAttention(d, heads, dropout=dr, batch_first=True)
    def forward(s, x, ctx):
        g, _ = s.read(x[None], s.P[None], s.P[None], need_weights=False); return g[0]


class FFNBlock(nn.Module):
    # parameter-matched per-node feed-forward block (capacity control)
    def __init__(s, d, width):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, d))
    def forward(s, x, ctx): return s.f(x)


def nparams(m): return sum(p.numel() for p in m.parameters())


class GraphNet(nn.Module):
    # generic residual encoder: embed -> local -> [global] -> local -> head
    # conv in {transformer, gcn, gat, sage, none}; glob in {isab, memory, ffn, none}
    def __init__(s, cards, n_num, n_cls, conv="transformer", glob="isab", hid=128, heads=8, L=64,
                 drop=0.2, layers=2, embed="embed", ffn_width=None):
        super().__init__()
        s.conv, s.glob, s.layers = conv, glob, layers
        s.embed = TabEmbed(cards, n_num, hid, embed)
        def mk():
            if conv == "transformer": return TransformerConv(hid, hid // heads, heads=heads, dropout=drop)
            if conv == "gat": return GATConv(hid, hid // heads, heads=heads, dropout=drop)
            if conv == "gcn": return GCNConv(hid, hid)
            if conv == "sage": return SAGEConv(hid, hid)
            return nn.Linear(hid, hid)
        s.locs = nn.ModuleList([mk() for _ in range(layers)])
        s.lns = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        if glob == "isab": s.g = LandmarkISAB(hid, heads, L, drop)
        elif glob == "memory": s.g = LandmarkMemory(hid, heads, L, drop)
        elif glob == "ffn": s.g = FFNBlock(hid, ffn_width)
        else: s.g = None
        s.lng = nn.LayerNorm(hid)
        s.dr = nn.Dropout(drop)
        s.head = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Dropout(drop), nn.Linear(hid, n_cls))
    def local(s, i, x, ei):
        return s.locs[i](x) if s.conv == "none" else s.locs[i](x, ei)
    def forward(s, g):
        x = s.embed(g.x_cat, g.x_num); ei = g.edge_index
        for i in range(s.layers):
            x = s.lns[i](x + s.dr(F.elu(s.local(i, x, ei))))
            if i == 0 and s.g is not None: x = s.lng(x + s.g(x, g.ctx))
        return s.head(x)


class MLPNet(nn.Module):
    # tabular MLP with the same entity-embedding front end
    def __init__(s, cards, n_num, n_cls, hid=256, layers=2, drop=0.2):
        super().__init__(); s.embed = TabEmbed(cards, n_num, hid)
        blocks = []
        for _ in range(layers - 1): blocks += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
        s.body = nn.Sequential(*blocks); s.out = nn.Linear(hid, n_cls)
    def forward(s, g): return s.out(s.body(s.embed(g.x_cat, g.x_num)))


class GPSNet(nn.Module):
    # GraphGPS layer stack (GIN local + Performer global attention), same front end
    def __init__(s, cards, n_num, n_cls, hid=128, heads=4, layers=2, drop=0.2):
        super().__init__(); s.embed = TabEmbed(cards, n_num, hid)
        s.convs = nn.ModuleList([GPSConv(hid, GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))),
                                         heads=heads, dropout=drop, attn_type="performer",
                                         attn_kwargs={"dropout": drop}) for _ in range(layers)])
        s.head = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Dropout(drop), nn.Linear(hid, n_cls))
    def forward(s, g):
        x = s.embed(g.x_cat, g.x_num)
        for c in s.convs: x = c(x, g.edge_index)
        return s.head(x)


def build_model(kind, D, hp):
    cards, n_num, C = D["prep"].cards, D["num"].shape[1], len(D["classes"])
    if kind == "MLP": return MLPNet(cards, n_num, C, hp.get("hid", 256), hp.get("layers", 2), hp.get("drop", 0.2))
    if kind == "GPS": return GPSNet(cards, n_num, C, hp.get("hid", 128), hp.get("heads", 4), hp.get("layers", 2), hp.get("drop", 0.2))
    conv = {"AGT": "transformer", "TransformerConv": "transformer", "GCN": "gcn", "GAT": "gat", "GraphSAGE": "sage"}[kind]
    glob = hp.get("glob", "isab" if kind == "AGT" else "none")
    return GraphNet(cards, n_num, C, conv=conv, glob=glob, hid=hp.get("hid", 128), heads=hp.get("heads", 8),
                    L=hp.get("L", 64), drop=hp.get("drop", 0.2), layers=hp.get("layers", 2),
                    embed=hp.get("embed", "embed"), ffn_width=hp.get("ffn_width"))


# ---------------- training / evaluation ----------------
def class_weights(y, C):
    c = np.bincount(y, minlength=C); return torch.tensor(len(y) / (C * np.maximum(c, 1)), dtype=torch.float, device=DEV)


@torch.no_grad()
def predict(model, g, chunk=None):
    model.eval(); g = g.to(DEV)
    lo = model(g)[g.n_ref:] if g.num_nodes > g.n_ref else model(g)
    return F.softmax(lo, 1).cpu().numpy()


def train_net(model, gtr, gva=None, yva=None, epochs=150, lr=3e-3, wd=5e-4, cw=True, C=None, eval_every=5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEV); gtr = gtr.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(gtr.y.cpu().numpy(), C) if cw else None
    best, best_state, hist = -1, None, dict(loss=[], acc=[], val=[])
    t0 = time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(gtr); loss = F.cross_entropy(out, gtr.y, weight=w)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sch.step()
        hist["loss"].append(float(loss)); hist["acc"].append(float((out.argmax(1) == gtr.y).float().mean()))
        if gva is not None and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
            p = predict(model, gva).argmax(1); v = f1_score(yva, p, average="macro", zero_division=0)
            hist["val"].append((ep + 1, v))
            if v > best: best, best_state = v, copy.deepcopy(model.state_dict())
    if DEV.type == "cuda": torch.cuda.synchronize()
    ttime = time.time() - t0
    if best_state is not None: model.load_state_dict(best_state)
    return model, dict(train_s=ttime, best_val=best, hist=hist)


def metrics(yt, proba, C):
    yp = proba.argmax(1)
    d = dict(Accuracy=accuracy_score(yt, yp),
             Precision=precision_score(yt, yp, average="weighted", zero_division=0),
             Recall=recall_score(yt, yp, average="weighted", zero_division=0),
             F1=f1_score(yt, yp, average="weighted", zero_division=0),
             MacroF1=f1_score(yt, yp, average="macro", zero_division=0),
             MCC=matthews_corrcoef(yt, yp))
    try:
        pr = proba / proba.sum(1, keepdims=True)
        d["ROC_AUC"] = roc_auc_score(yt, pr, multi_class="ovr", average="weighted", labels=list(range(C)))
    except Exception: d["ROC_AUC"] = float("nan")
    return d


def perclass_f1(yt, yp, C): return f1_score(yt, yp, average=None, labels=list(range(C)), zero_division=0)


def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
