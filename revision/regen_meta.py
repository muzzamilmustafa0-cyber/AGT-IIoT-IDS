import json, numpy as np, pandas as pd
from common import load_raw, save_json
raw = load_raw("X-IIoTID", keep=("Timestamp",)); raw["__ts"] = pd.to_numeric(raw["__Timestamp"], errors="coerce")
raw = raw.dropna(subset=["__ts"]).sort_values("__ts", kind="stable").reset_index(drop=True).iloc[::4].reset_index(drop=True)
CL = sorted(raw["y_str"].unique()); ym = raw["y_str"].map({c: i for i, c in enumerate(CL)}).values
n = len(raw); cut = int(n * 0.25); W = np.array_split(np.arange(cut, n), 15); ts = raw["__ts"].values
rows = [dict(window=t, start=float(ts[w].min()), end=float(ts[w].max()), n=len(w), counts=np.bincount(ym[w], minlength=len(CL)).tolist()) for t, w in enumerate(W, 1)]
save_json(dict(classes=CL, init_counts=np.bincount(ym[:cut], minlength=len(CL)).tolist(), init_start=float(ts[0]), init_end=float(ts[cut-1]), windows=rows), os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "drift", "windows_meta.json"))
print(n, cut, [r["n"] for r in rows][:3], pd.to_datetime([ts[0], ts[cut-1]], unit="s"))
