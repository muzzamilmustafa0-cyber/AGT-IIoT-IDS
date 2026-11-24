# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# E-GraphSAGE baseline (Lo et al. 2022): hosts (IP:port) are nodes, flows are edges, edge classification.
# Uses the same subsample / splits / train-only preprocessing as the main study.
import sys
from common import *

ODIR = os.path.join(OUT, "stationary", "X-IIoTID"); os.makedirs(ODIR, exist_ok=True)

class ESAGE(nn.Module):
    def __init__(s, de, hid, C, drop):
        super().__init__()
        s.l1 = nn.Linear(de + de, hid); s.l2 = nn.Linear(hid + de, hid)
        s.cls = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, C))
        s.dr = nn.Dropout(drop); s.de = de
    def agg(s, ef, src, dst, n):
        # mean of the features of all flows incident to each endpoint
        a = torch.zeros(n, ef.size(1), device=ef.device).index_add_(0, src, ef).index_add_(0, dst, ef)
        deg = torch.zeros(n, device=ef.device).index_add_(0, src, torch.ones_like(src, dtype=torch.float)).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        return a / deg.clamp(min=1)[:, None]
    def forward(s, ef, src, dst, n):
        h = torch.ones(n, s.de, device=ef.device)
        a = s.agg(ef, src, dst, n)
        h = s.dr(F.relu(s.l1(torch.cat([h, a], 1))))
        h = s.dr(F.relu(s.l2(torch.cat([h, a], 1))))
        return s.cls(torch.cat([h[src], h[dst]], 1))

def flow_graph(D, sel):
    df = D["df"].iloc[sel]
    a = (df["__Scr_IP"].astype(str) + ":" + df["Scr_port"].astype(str)).values
    b = (df["__Des_IP"].astype(str) + ":" + df["Des_port"].astype(str)).values
    codes, uniq = pd.factorize(np.concatenate([a, b]))
    n = len(sel)
    return (torch.tensor(D["flat"][sel], device=DEV), torch.tensor(codes[:n], device=DEV),
            torch.tensor(codes[n:], device=DEV), len(uniq), torch.tensor(D["y"][sel], device=DEV))

def fit(D, hp, seed, epochs=150):
    torch.manual_seed(seed); C = len(D["classes"])
    tr = flow_graph(D, D["itr"]); va = flow_graph(D, D["iva"])
    m = ESAGE(D["flat"].shape[1], hp["hid"], C, hp["drop"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(D["y"][D["itr"]], C); best, bs = -1, None; t0 = time.time()
    for ep in range(epochs):
        m.train(); opt.zero_grad(); out = m(*tr[:4]); loss = F.cross_entropy(out, tr[4], weight=w)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step(); sch.step()
        if (ep + 1) % 5 == 0:
            m.eval()
            with torch.no_grad(): p = m(*va[:4]).argmax(1).cpu().numpy()
            v = f1_score(D["y"][D["iva"]], p, average="macro", zero_division=0)
            if v > best: best, bs = v, copy.deepcopy(m.state_dict())
    tt = time.time() - t0; m.load_state_dict(bs); return m, best, tt

def evaluate(m, D):
    te = flow_graph(D, D["ite"]); m.eval()
    with torch.no_grad(): return F.softmax(m(*te[:4]), 1).cpu().numpy()

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "tune":
        D = split_stationary("X-IIoTID", 0, keep=("Scr_IP", "Des_IP")); rows = []
        for lr in [1e-3, 3e-3]:
            for drop in [0.1, 0.2]:
                for hid in [128, 256]:
                    hp = dict(lr=lr, drop=drop, hid=hid); m, v, tt = fit(D, hp, 0)
                    rows.append(dict(hp=hp, val_macroF1=v, train_s=tt)); print(hp, round(v, 4), flush=True)
        best = max(rows, key=lambda r: r["val_macroF1"])
        save_json(dict(best=best["hp"], best_val=best["val_macroF1"], all=rows), os.path.join(ODIR, "tuning_egraphsage.json"))
    else:
        hp = json.load(open(os.path.join(ODIR, "tuning_egraphsage.json")))["best"]
        path = os.path.join(ODIR, "main_runs.jsonl")
        for seed in [int(s) for s in sys.argv[2].split(",")]:
            D = split_stationary("X-IIoTID", seed, keep=("Scr_IP", "Des_IP")); C = len(D["classes"]); yte = D["y"][D["ite"]]
            m, v, tt = fit(D, hp, seed); pr = evaluate(m, D)
            r = dict(seed=seed, model="E-GraphSAGE", hp=hp, train_s=tt, **metrics(yte, pr, C),
                     perclass=perclass_f1(yte, pr.argmax(1), C).tolist(), support=np.bincount(yte, minlength=C).tolist(),
                     classes=D["classes"], pred=pr.argmax(1).tolist(), ytrue=yte.tolist(), params=nparams(m), best_val=v)
            with open(path, "a") as f: f.write(json.dumps(r, default=float) + "\n")
            print(seed, {k: round(r[k], 4) for k in ["Accuracy", "F1", "MacroF1", "MCC"]}, flush=True)
