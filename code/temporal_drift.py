# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""
REALISTIC temporal concept-drift test on real X-IIoTID.
Train on EARLIER traffic, test on LATER traffic (chronological), binary
attack-vs-normal. Compare to a random (no-drift) split to isolate the drift
effect. Question: does XGBoost's clean-data dominance survive real drift?
Train-only preprocessing (no leakage). AGT config identical to main run.
"""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.neighbors import kneighbors_graph

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID,HEADS,DROP,NLAND,EPOCHS,LR,K=96,4,0.2,64,100,3e-3,10
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"
NTR,NTE=50000,20000

# ---- load + chronological sort ----
df=pd.read_csv(CSV, low_memory=False)
df["ts"]=pd.to_numeric(df["Timestamp"], errors="coerce")
df=df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
y_bin=(df["class2"].astype(str).str.strip()!="Normal").astype(int).values
drop=[c for c in ["class1","class2","class3","Date","Timestamp","ts",
                  "Scr_IP","Des_IP"] if c in df.columns]
X=df.drop(columns=drop)
cat_cols=[c for c in X.columns if X[c].dtype=="object"]
num_cols=[c for c in X.columns if c not in cat_cols]
n=len(X); print(f"rows {n} | attack rate {y_bin.mean():.3f} | cat {len(cat_cols)} num {len(num_cols)}")

def preprocess(Xtr_raw, Xte_raw):
    Xtr=Xtr_raw.copy(); Xte=Xte_raw.copy()
    # impute from TRAIN
    for c in num_cols:
        Xtr[c]=pd.to_numeric(Xtr[c],errors="coerce"); Xte[c]=pd.to_numeric(Xte[c],errors="coerce")
        m=Xtr[c].replace([np.inf,-np.inf],np.nan).median()
        Xtr[c]=Xtr[c].replace([np.inf,-np.inf],np.nan).fillna(m)
        Xte[c]=Xte[c].replace([np.inf,-np.inf],np.nan).fillna(m)
    # encode categoricals with TRAIN vocab; unseen -> -1
    for c in cat_cols:
        vocab={v:i for i,v in enumerate(Xtr[c].astype(str).unique())}
        Xtr[c]=Xtr[c].astype(str).map(vocab).fillna(-1).astype(int)
        Xte[c]=Xte[c].astype(str).map(vocab).fillna(-1).astype(int)
    sc=StandardScaler().fit(Xtr.values)
    return sc.transform(Xtr.values), sc.transform(Xte.values)

# ---- AGT (identical) ----
class LGA(nn.Module):
    def __init__(s,d,h=4,L=64,dr=0.2):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h; s.Lp=nn.Parameter(torch.randn(L,d)*0.02)
        s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x):
        N=x.size(0); L=s.Lp.size(0)
        q=s.q(x).view(N,s.h,s.hd); k=s.k(s.Lp).view(L,s.h,s.hd); v=s.v(s.Lp).view(L,s.h,s.hd)
        a=s.dr(F.softmax(torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5),-1))
        return s.o(torch.einsum("hnl,lhd->nhd",a,v).reshape(N,s.d))
class AGT(nn.Module):
    def __init__(s,ic,h,oc,heads=4,dr=0.2,L=64):
        super().__init__(); s.ip=nn.Linear(ic,h)
        s.c1=TransformerConv(h,h//heads,heads=heads,dropout=dr)
        s.c2=TransformerConv(h,h//heads,heads=heads,dropout=dr)
        s.g=LGA(h,heads,L,dr); s.n1,s.n2,s.n3=nn.LayerNorm(h),nn.LayerNorm(h),nn.LayerNorm(h)
        s.f1=nn.Linear(h,h//2); s.f2=nn.Linear(h//2,oc); s.dr=nn.Dropout(dr)
    def forward(s,data):
        x,ei=data.x,data.edge_index; x=F.relu(s.ip(x))
        x=s.n1(x+s.dr(F.elu(s.c1(x,ei)))); x=s.n2(x+s.g(x)); x=s.n3(x+s.dr(F.elu(s.c2(x,ei))))
        return s.f2(s.dr(F.elu(s.f1(x))))
def graph(Xd,yd,k=K):
    A=kneighbors_graph(Xd,k,mode="connectivity",include_self=False); A=A.maximum(A.T)
    return Data(x=torch.tensor(Xd,dtype=torch.float),
        edge_index=torch.tensor(np.array(A.nonzero()),dtype=torch.long),
        y=torch.tensor(yd,dtype=torch.long))
def cw(yt,nc=2):
    c=np.bincount(yt,minlength=nc); return torch.tensor(len(yt)/(nc*np.maximum(c,1)),dtype=torch.float).to(DEVICE)
def train_agt(Xtr,ytr):
    g=graph(Xtr,ytr).to(DEVICE); m=AGT(Xtr.shape[1],HID,2,HEADS,DROP,NLAND).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); crit=nn.CrossEntropyLoss(weight=cw(ytr))
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad(); out=m(g); loss=crit(out,g.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step(); sch.step()
    return m
@torch.no_grad()
def eval_agt(m,Xte,yte):
    g=graph(Xte,yte).to(DEVICE); m.eval(); p=m(g).argmax(1).cpu().numpy()
    return p

def metrics(yt,yp):
    return dict(F1=round(f1_score(yt,yp,average="weighted",zero_division=0),4),
                MacroF1=round(f1_score(yt,yp,average="macro",zero_division=0),4),
                AttackRecall=round(recall_score(yt,yp,pos_label=1,zero_division=0),4),
                Acc=round(accuracy_score(yt,yp),4))

def run_split(name, tr_idx, te_idx):
    rng=np.random.default_rng(SEED)
    tr=rng.choice(tr_idx,min(NTR,len(tr_idx)),replace=False)
    te=rng.choice(te_idx,min(NTE,len(te_idx)),replace=False)
    Xtr,Xte=preprocess(X.iloc[tr], X.iloc[te]); ytr,yte=y_bin[tr],y_bin[te]
    out={}
    xgb=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.9,
        eval_metric="logloss",random_state=SEED,verbosity=0).fit(Xtr,ytr)
    out["XGBoost"]=metrics(yte,xgb.predict(Xte))
    mlp=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=200,random_state=SEED).fit(Xtr,ytr)
    out["MLP"]=metrics(yte,mlp.predict(Xte))
    m=train_agt(Xtr,ytr); out["AGT"]=metrics(yte,eval_agt(m,Xte,yte))
    print(f"\n[{name}] train {len(tr)} test {len(te)} | test attack rate {yte.mean():.3f}")
    for k,v in out.items(): print(f"  {k:9s} {v}")
    return out

# chronological split: first 70% train, last 30% test
cut=int(n*0.70)
tr_idx=np.arange(0,cut); te_idx=np.arange(cut,n)
res_temporal=run_split("TEMPORAL (train past -> test future)", tr_idx, te_idx)

# random reference (no drift): shuffle then same split sizes
rng=np.random.default_rng(1)
perm=rng.permutation(n); tr_r=perm[:cut]; te_r=perm[cut:]
res_random=run_split("RANDOM (no drift, reference)", tr_r, te_r)

# summary
print("\n==== F1: RANDOM vs TEMPORAL (drop = drift cost) ====")
rows=[]
for mdl in ["XGBoost","MLP","AGT"]:
    r=res_random[mdl]["F1"]; t=res_temporal[mdl]["F1"]
    print(f"  {mdl:9s} random {r:.4f} -> temporal {t:.4f}  (drop {r-t:+.4f}) | attack recall temporal {res_temporal[mdl]['AttackRecall']:.4f}")
    rows.append(dict(Model=mdl, F1_random=r, F1_temporal=t, F1_drop=round(r-t,4),
                     AttackRecall_temporal=res_temporal[mdl]["AttackRecall"],
                     MacroF1_temporal=res_temporal[mdl]["MacroF1"]))
os.makedirs("temporal_out",exist_ok=True)
pd.DataFrame(rows).to_csv("temporal_out/temporal_drift.csv",index=False)

plt.figure(figsize=(7,4.5)); x=np.arange(3); w=0.35
plt.bar(x-w/2,[res_random[m]["F1"] for m in ["XGBoost","MLP","AGT"]],w,label="Random split (no drift)")
plt.bar(x+w/2,[res_temporal[m]["F1"] for m in ["XGBoost","MLP","AGT"]],w,label="Temporal split (real drift)")
plt.xticks(x,["XGBoost","MLP","AGT"]); plt.ylabel("Weighted F1"); plt.ylim(0,1.05)
plt.title("Real temporal concept drift (X-IIoTID): attack detection")
plt.legend(); plt.tight_layout(); plt.savefig("temporal_out/temporal_drift.png",dpi=300,bbox_inches="tight")
print("\nSaved temporal_out/temporal_drift.csv + .png")
