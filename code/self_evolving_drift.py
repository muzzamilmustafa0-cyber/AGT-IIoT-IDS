# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""
DECISIVE TEST: self-evolving AGT under REAL temporal concept drift (X-IIoTID).
Prequential protocol: train on the past, stream the chronological future in
blocks; for each block evaluate FIRST (predict before labels), then adapt on a
disjoint labelled portion. Compare static vs adaptive AGT and static vs
retrained XGBoost. Binary attack-vs-normal. Preprocessing fit on PAST only.
"""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.neighbors import kneighbors_graph

SEED=int(os.environ.get("AGT_SEED","42")); np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"
TOTAL=int(os.environ.get("AGT_TOTAL","120000"))   # chronological subsample
PAST_FRAC=0.55; NBLOCKS=8
HID=128; HEADS=8; DROP=0.2; NLAND=64; INIT_EPOCHS=120
ADAPT_STEPS=int(os.environ.get("AGT_STEPS","15")); LR=3e-3; K=10
BUF0=int(os.environ.get("AGT_BUF","1500")); BUFMAX=int(os.environ.get("AGT_BUFMAX","2500"))
rng=np.random.default_rng(SEED)

# ---- load + chronological order ----
df=pd.read_csv(CSV, low_memory=False)
df["ts"]=pd.to_numeric(df["Timestamp"], errors="coerce")
df=df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
ybin=(df["class2"].astype(str).str.strip()!="Normal").astype(int).values
df=df.drop(columns=[c for c in ["class1","class2","class3","Date","Timestamp","ts","Scr_IP","Des_IP"] if c in df.columns])
# systematic subsample preserving temporal order
if len(df)>TOTAL:
    keep=np.linspace(0,len(df)-1,TOTAL).astype(int)
    df=df.iloc[keep].reset_index(drop=True); ybin=ybin[keep]
n=len(df); cut=int(n*PAST_FRAC)
print(f"rows {n} | past {cut} future {n-cut} | future attack rate {ybin[cut:].mean():.3f}")

# ---- typing (decide on PAST), fit transforms on PAST only ----
num_cols=[]; cat_cols=[]; freq_maps={}
for c in df.columns:
    col=df[c]
    if col.dtype!="object": num_cols.append(c); continue
    co=pd.to_numeric(col.iloc[:cut].astype(str).str.replace(",",""),errors="coerce")
    if co.notna().mean()>0.98:
        df[c]=pd.to_numeric(col.astype(str).str.replace(",",""),errors="coerce"); num_cols.append(c)
    elif col.iloc[:cut].nunique()<=64: cat_cols.append(c)
    else:
        vc=col.iloc[:cut].value_counts(); freq_maps[c]=vc
        df[c]=col.map(vc).astype(float); num_cols.append(c)
# impute (past medians/modes)
for c in num_cols:
    df[c]=pd.to_numeric(df[c],errors="coerce"); m=df[c].iloc[:cut].median(); df[c]=df[c].fillna(m)
cat_vocab={}; cat_mat=np.zeros((n,len(cat_cols)),dtype=np.int64); cards=[]
for j,c in enumerate(cat_cols):
    vals=df[c].iloc[:cut].fillna("NA").astype(str)
    vocab={v:i for i,v in enumerate(vals.unique())}; cat_vocab[c]=vocab; cards.append(len(vocab)+1)
    cat_mat[:,j]=df[c].fillna("NA").astype(str).map(vocab).fillna(len(vocab)).astype(int)
sc=StandardScaler().fit(df[num_cols].iloc[:cut].values)
num_mat=sc.transform(df[num_cols].values).astype(np.float32)
allf=np.hstack([num_mat,cat_mat.astype(np.float32)]).astype(np.float32)
allf=StandardScaler().fit(allf[:cut]).transform(allf).astype(np.float32)
print(f"cats {len(cat_cols)} num {len(num_cols)}")

# ---- AGT (embeddings + weighted CE) ----
class TabEmbed(nn.Module):
    def __init__(s,cards,n_num,d,cap=16):
        super().__init__(); s.embs=nn.ModuleList([nn.Embedding(K_,min(cap,max(3,int(np.ceil(np.sqrt(K_)))))) for K_ in cards])
        ed=sum(e.embedding_dim for e in s.embs) if cards else 0; s.ln=nn.LayerNorm(n_num)
        s.proj=nn.Sequential(nn.Linear(ed+n_num,d),nn.LayerNorm(d),nn.ReLU())
    def forward(s,xc,xn):
        e=[emb(xc[:,i].clamp(max=emb.num_embeddings-1)) for i,emb in enumerate(s.embs)]
        return s.proj(torch.cat(e+[s.ln(xn)],1) if e else s.ln(xn))
class LGA(nn.Module):
    def __init__(s,d,h,L,dr):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h; s.Lp=nn.Parameter(torch.randn(L,d)*0.02)
        s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x):
        N=x.size(0);L=s.Lp.size(0)
        q=s.q(x).view(N,s.h,s.hd);k=s.k(s.Lp).view(L,s.h,s.hd);v=s.v(s.Lp).view(L,s.h,s.hd)
        a=s.dr(F.softmax(torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5),-1))
        return s.o(torch.einsum("hnl,lhd->nhd",a,v).reshape(N,s.d))
class AGT(nn.Module):
    def __init__(s,cards,n_num,h,oc,heads,dr,L):
        super().__init__(); s.embed=TabEmbed(cards,n_num,h)
        s.c1=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.c2=TransformerConv(h,h//heads,heads=heads,dropout=dr)
        s.g=LGA(h,heads,L,dr); s.n1,s.n2,s.n3=nn.LayerNorm(h),nn.LayerNorm(h),nn.LayerNorm(h)
        s.head=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Dropout(dr),nn.Linear(h,oc))
    def forward(s,d):
        x=s.embed(d.x_cat,d.x_num); ei=d.edge_index
        x=s.n1(x+F.elu(s.c1(x,ei))); x=s.n2(x+s.g(x)); x=s.n3(x+F.elu(s.c2(x,ei)))
        return s.head(x)
def graph(sel):
    A=kneighbors_graph(allf[sel],min(K,len(sel)-1),mode="connectivity",include_self=False); A=A.maximum(A.T)
    d=Data(edge_index=torch.tensor(np.array(A.nonzero()),dtype=torch.long),y=torch.tensor(ybin[sel],dtype=torch.long))
    d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long); d.x_num=torch.tensor(num_mat[sel],dtype=torch.float)
    d.num_nodes=len(sel); return d
def wts(yv):
    c=np.bincount(yv,minlength=2); return torch.tensor(len(yv)/(2*np.maximum(c,1)),dtype=torch.float).to(DEV)
@torch.no_grad()
def agt_predict(model,sel):
    g=graph(sel).to(DEV); model.eval(); return model(g).argmax(1).cpu().numpy()
def agt_update(model,opt,sel_lab,buf_idx,steps,w):
    sel=np.concatenate([sel_lab,buf_idx]); g=graph(sel).to(DEV); model.train()
    crit=nn.CrossEntropyLoss(weight=w)
    for _ in range(steps):
        opt.zero_grad(); out=model(g); loss=crit(out,g.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()

# ---- initial training on PAST ----
past=np.arange(cut)
print("Training initial AGT on past ...")
torch.manual_seed(SEED)
agt=AGT(cards,len(num_cols),HID,2,HEADS,DROP,NLAND).to(DEV)
gp=graph(past).to(DEV); opt=torch.optim.AdamW(agt.parameters(),lr=LR,weight_decay=5e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=INIT_EPOCHS); w0=wts(ybin[past])
t=time.time()
for ep in range(INIT_EPOCHS):
    agt.train(); opt.zero_grad(); out=agt(gp); loss=F.cross_entropy(out,gp.y,weight=w0)
    loss.backward(); torch.nn.utils.clip_grad_norm_(agt.parameters(),2.0); opt.step(); sch.step()
print(f"  AGT trained ({time.time()-t:.0f}s)")
import copy
agt_static=copy.deepcopy(agt)            # frozen reference
aopt=torch.optim.AdamW(agt.parameters(),lr=1.2e-3,weight_decay=1e-4)  # adapter optimiser

def newxgb(): return XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.1,subsample=0.9,
    eval_metric="logloss",random_state=SEED,verbosity=0,tree_method="hist")
xgb_static=newxgb().fit(allf[past],ybin[past])
xgb_re=copy.copy(xgb_static)
online_mlp=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=200,random_state=SEED).fit(allf[past],ybin[past])  # online incremental baseline

# replay buffer (past) for AGT rehearsal
buf=rng.choice(past,BUF0,replace=False)
cum_adapt=[]                              # accumulated revealed future labels for XGB retrain

# ---- prequential streaming over future blocks ----
future=np.arange(cut,n); blocks=np.array_split(future,NBLOCKS)
rows=[]
def f1(sel,yp): return f1_score(ybin[sel],yp,average="weighted",zero_division=0)
def rec(sel,yp): return recall_score(ybin[sel],yp,pos_label=1,zero_division=0)
for bi,blk in enumerate(blocks,1):
    perm=rng.permutation(len(blk)); h=len(blk)//2
    ev=blk[perm[:h]]; ad=blk[perm[h:]]                     # disjoint eval / adapt-labels
    # evaluate BEFORE adapting (predict before labels)
    r={"Block":bi,"attack_rate":round(float(ybin[ev].mean()),3)}
    r["AGT_static"]=round(f1(ev,agt_predict(agt_static,ev)),4)
    r["AGT_selfevolve"]=round(f1(ev,agt_predict(agt,ev)),4)
    r["XGB_static"]=round(f1(ev,xgb_static.predict(allf[ev])),4)
    r["XGB_retrained"]=round(f1(ev,xgb_re.predict(allf[ev])),4)
    r["AGT_SE_recall"]=round(rec(ev,agt_predict(agt,ev)),4)
    r["XGB_re_recall"]=round(rec(ev,xgb_re.predict(allf[ev])),4)
    r["OnlineMLP"]=round(f1(ev,online_mlp.predict(allf[ev])),4)
    rows.append(r)
    print(f"  block {bi}: AGTstatic {r['AGT_static']:.3f} | AGT-SE {r['AGT_selfevolve']:.3f} | "
          f"XGBstatic {r['XGB_static']:.3f} | XGBretrain {r['XGB_retrained']:.3f}")
    # adapt on disjoint labelled portion
    agt_update(agt,aopt,ad,buf,ADAPT_STEPS,wts(ybin[ad]))
    buf=np.concatenate([buf,ad])[-BUFMAX:]
    cum_adapt.append(ad)
    online_mlp.partial_fit(allf[ad],ybin[ad])  # incremental online update
    xgb_re=newxgb().fit(np.vstack([allf[past]]+[allf[a] for a in cum_adapt]),
                        np.concatenate([ybin[past]]+[ybin[a] for a in cum_adapt]))

res=pd.DataFrame(rows)
os.makedirs("adaptive_out",exist_ok=True); res.to_csv("adaptive_out/adaptive_drift.csv",index=False)
print("\n==== PREQUENTIAL F1 UNDER REAL TEMPORAL DRIFT ===="); print(res.to_string(index=False))
print("\nMean over future blocks:")
for c in ["AGT_static","AGT_selfevolve","OnlineMLP","XGB_static","XGB_retrained"]:
    print(f"  {c:16s}: {res[c].mean():.4f}")
print(f"\nSelf-evolving recovery vs static AGT: {res['AGT_selfevolve'].mean()-res['AGT_static'].mean():+.4f}")
print(f"Self-evolving AGT vs retrained XGBoost: {res['AGT_selfevolve'].mean()-res['XGB_retrained'].mean():+.4f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
for c,st in [("AGT_static","^:"),("AGT_selfevolve","D-"),("OnlineMLP","v--"),("XGB_static","s:"),("XGB_retrained","o-")]:
    plt.plot(res["Block"],res[c],st,lw=2,ms=7,label=c.replace("_"," "))
plt.xlabel("Future block (chronological)"); plt.ylabel("Weighted F1"); plt.ylim(0,1.02)
plt.title("Self-evolving AGT vs static/retrained baselines under real drift"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("adaptive_out/adaptive_drift.png",dpi=300,bbox_inches="tight")
print("saved adaptive_out/adaptive_drift.csv + .png")
