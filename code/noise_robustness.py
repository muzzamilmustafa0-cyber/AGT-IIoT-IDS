# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""
DRIFT-ROBUSTNESS PROBE (real X-IIoTID).
Question: under progressive concept drift, does the strongest STATIC baseline
(XGBoost) degrade while the ADAPTIVE AGT (self-evolving) stays robust?
Same data / preprocessing / AGT config as the main run. Diagnostic only.
"""
import os, time, copy, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID,HEADS,DROP,NLAND,EPOCHS,LR,K=96,4,0.2,64,100,3e-3,10
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"
SAMPLE=int(os.environ.get("AGT_SAMPLE","80000"))

# ---- load real data (identical preprocessing) ----
df=pd.read_csv(CSV, low_memory=False)
drop=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in df.columns]
df=df.drop(columns=drop); df=df.rename(columns={"class2":"class"})
df["class"]=df["class"].astype(str).str.strip()
if len(df)>SAMPLE:
    df,_=train_test_split(df, train_size=SAMPLE, random_state=SEED, stratify=df["class"])
df=df.reset_index(drop=True)
df=df.replace([np.inf,-np.inf],np.nan)
for c in df.columns:
    if df[c].dtype.kind in "biufc": df[c]=df[c].fillna(df[c].median())
    elif c!="class": df[c]=df[c].fillna(df[c].mode()[0])
for c in df.select_dtypes(include=["object"]).columns:
    if c!="class": df[c]=LabelEncoder().fit_transform(df[c].astype(str))
X=df.drop(columns=["class"]); le=LabelEncoder(); y=le.fit_transform(df["class"])
class_names=[str(c) for c in le.classes_]; n_cls=len(class_names)
Xs=StandardScaler().fit_transform(X)
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=SEED,stratify=y)
n_feat=Xtr.shape[1]
print(f"Data: train {Xtr.shape} test {Xte.shape} | {n_cls} classes | device {DEVICE}")

# ---- AGT (identical) ----
class LGA(nn.Module):
    def __init__(s,d,h=4,L=64,dr=0.2):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h
        s.L=nn.Parameter(torch.randn(L,d)*0.02)
        s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x):
        N=x.size(0); L=s.L.size(0)
        q=s.q(x).view(N,s.h,s.hd); k=s.k(s.L).view(L,s.h,s.hd); v=s.v(s.L).view(L,s.h,s.hd)
        a=torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5); a=s.dr(F.softmax(a,-1))
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
def cw(yt):
    c=np.bincount(yt,minlength=n_cls); return torch.tensor(len(yt)/(n_cls*np.maximum(c,1)),dtype=torch.float).to(DEVICE)
def train(model,g,epochs=EPOCHS,lr=LR,w=None):
    model=model.to(DEVICE); g=g.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); crit=nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        model.train(); opt.zero_grad(); out=model(g); loss=crit(out,g.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); sch.step()
    return model
@torch.no_grad()
def agt_f1(model,Xe,ye):
    g=graph(Xe,ye).to(DEVICE); model.eval()
    p=model(g).argmax(1).cpu().numpy(); return f1_score(ye,p,average="weighted",zero_division=0)

# ---- train static models ----
print("Training AGT ..."); t=time.time()
gtr=graph(Xtr,ytr); agt=train(AGT(n_feat,HID,n_cls,HEADS,DROP,NLAND),gtr,w=cw(ytr))
print(f"  AGT trained ({time.time()-t:.0f}s)")
print("Training XGBoost ..."); xgb=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
    subsample=0.9,eval_metric="mlogloss",random_state=SEED,verbosity=0).fit(Xtr,ytr)
print("Training MLP ..."); mlp=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=200,random_state=SEED).fit(Xtr,ytr)

# adaptive AGT setup (rehearsal)
rng=np.random.default_rng(SEED)
adapt=copy.deepcopy(agt); aopt=torch.optim.AdamW(adapt.parameters(),lr=1.2e-3,weight_decay=1e-4)
acrit=nn.CrossEntropyLoss(weight=cw(ytr))
bidx=rng.choice(len(Xtr),800,replace=False); bX,bY=Xtr[bidx].copy(),ytr[bidx].copy()

# ---- staged drift ----
nS=6; cX=np.array_split(Xte,nS); cY=np.array_split(yte,nS)
rows=[]
for s,(Xc,yc) in enumerate(zip(cX,cY),1):
    sig=0.10*(s-1); Xd=Xc+rng.normal(0,sig,Xc.shape)
    perm=rng.permutation(len(Xd)); half=len(Xd)//2
    Xa,ya=Xd[perm[:half]],yc[perm[:half]]; Xe,ye=Xd[perm[half:]],yc[perm[half:]]
    # static baselines on eval split
    f_xgb=f1_score(ye,xgb.predict(Xe),average="weighted",zero_division=0)
    f_mlp=f1_score(ye,mlp.predict(Xe),average="weighted",zero_division=0)
    f_agt_static=agt_f1(agt,Xe,ye)
    # adaptive AGT: rehearsal update on adapt split + buffer, then eval
    Xu=np.vstack([Xa,bX]); yu=np.concatenate([ya,bY]); gu=graph(Xu,yu,k=8).to(DEVICE)
    adapt.train()
    for _ in range(8):
        aopt.zero_grad(); out=adapt(gu); loss=acrit(out,gu.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapt.parameters(),2.0); aopt.step()
    f_agt_adapt=agt_f1(adapt,Xe,ye)
    bX=np.vstack([bX,Xa])[-800:]; bY=np.concatenate([bY,ya])[-800:]  # refresh buffer
    rows.append(dict(Stage=s,Drift_sigma=round(sig,2),XGBoost_static=round(f_xgb,4),
                     MLP_static=round(f_mlp,4),AGT_static=round(f_agt_static,4),
                     AGT_adaptive=round(f_agt_adapt,4)))
    print(f"  stage {s} σ={sig:.2f} | XGB {f_xgb:.4f} | MLP {f_mlp:.4f} | AGT static {f_agt_static:.4f} | AGT adapt {f_agt_adapt:.4f}")

res=pd.DataFrame(rows)
os.makedirs("drift_probe_out",exist_ok=True)
res.to_csv("drift_probe_out/drift_robustness.csv",index=False)

plt.figure(figsize=(8,5))
for col,style in [("XGBoost_static","o-"),("MLP_static","s--"),("AGT_static","^:"),("AGT_adaptive","D-")]:
    plt.plot(res["Drift_sigma"],res[col],style,lw=2,ms=7,label=col.replace("_"," "))
plt.xlabel("Concept-drift magnitude (σ)"); plt.ylabel("Weighted F1")
plt.title("Robustness to concept drift (real X-IIoTID)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("drift_probe_out/drift_robustness.png",dpi=300,bbox_inches="tight")

print("\n=== DRIFT ROBUSTNESS (weighted F1) ===")
print(res.to_string(index=False))
print("\nDegradation from clean (σ=0) to σ=0.5:")
for col in ["XGBoost_static","MLP_static","AGT_static","AGT_adaptive"]:
    d=res[col].iloc[0]-res[col].iloc[-1]; print(f"  {col:16s}: {res[col].iloc[0]:.4f} -> {res[col].iloc[-1]:.4f}  (drop {d:+.4f})")
