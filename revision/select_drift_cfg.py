# choose update settings per method from the pseudo-stream (initial period only)
import json, glob, os, re, numpy as np
rows = {}
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "drift", "pseudo_multi_*_seed*.json")):
    d = json.load(open(f)); c = d["cfg"]; key = (c["policy"], c["M"], c["S"], c["eta"])
    for r in d["rows"]:
        for m in ["AGT-rehearsal", "MLP-rehearsal", "XGB-sliding"]:
            rows.setdefault(m, {}).setdefault(key, []).append(r[m]["MacroF1"])
best = {}
for m, kv in rows.items():
    ranked = sorted(kv.items(), key=lambda t: -np.mean(t[1]))
    print(m); [print("  ", k, round(np.mean(v), 4), len(v)) for k, v in ranked[:16]]
    k = ranked[0][0]; best[m] = dict(policy=k[0], M=k[1], S=k[2], eta=k[3], pseudo_macroF1=float(np.mean(ranked[0][1])))
json.dump(best, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "drift", "tuning_choice.json"), "w"), indent=2)
json.dump(best["AGT-rehearsal"], open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_cfg.json"), "w"), indent=2)
for m in best: print(m, best[m])
