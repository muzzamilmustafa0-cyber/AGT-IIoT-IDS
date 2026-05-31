# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""Runtime / scalability / adaptation-cost analysis (concerns 5 and 6).
Measures: k-NN graph build time vs N; AGT train/inference latency and GPU memory;
self-evolving update wall-clock vs full XGBoost retrain. Real X-IIoTID."""
import os, time, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"
HID=128;HEADS=8;DROP=0.2;NLAND=64;EPOCHS=150;LR=3e-3;K=10
full=pd.read_csv(CSV,low_memory=False)
full=full.drop(columns=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in full.columns]).rename(columns={"class2":"class"})
full["class"]=full["class"].astype(str).str.strip()
def prep(df):
    df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan); le=LabelEncoder(); y=le.fit_transform(df["class"]); Xdf=df.drop(columns=["class"]); num_cols=[];cat_cols=[]
    for c in Xdf.columns:
        col=Xdf[c]
        if col.dtype!="object": num_cols.append(c); continue
        co=pd.to_numeric(col.astype(str).str.replace(",",""),errors="coerce")
        if co.notna().mean()>0.98: Xdf[c]=co; num_cols.append(c)
        elif col.nunique()<=64: cat_cols.append(c)
        else: Xdf[c]=col.map(col.value_counts()).astype(float); num_cols.append(c)
    for c in num_cols: Xdf[c]=pd.to_numeric(Xdf[c],errors="coerce").fillna(pd.to_numeric(Xdf[c],errors="coerce").median())
    for c in cat_cols: Xdf[c]=Xdf[c].fillna(Xdf[c].mode()[0]).astype(str)
    cat=np.zeros((len(Xdf),len(cat_cols)),dtype=np.int64); cards=[]
    for j,c in enumerate(cat_cols): cd,uq=pd.factorize(Xdf[c]); cat[:,j]=cd; cards.append(len(uq))
    num=StandardScaler().fit_transform(Xdf[num_cols].values).astype(np.float32)
    allf=StandardScaler().fit_transform(np.hstack([num,cat.astype(np.float32)])).astype(np.float32)
    return y,len(le.classes_),num,cat,cards,allf
class TabEmbed(nn.Module):
    def __init__(s,cards,n_num,d,cap=16):
        super().__init__(); s.embs=nn.ModuleList([nn.Embedding(K_+1,min(cap,max(3,int(np.ceil(np.sqrt(K_)))))) for K_ in cards]); ed=sum(e.embedding_dim for e in s.embs); s.ln=nn.LayerNorm(n_num); s.proj=nn.Sequential(nn.Linear(ed+n_num,d),nn.LayerNorm(d),nn.ReLU())
    def forward(s,xc,xn): e=[emb(xc[:,i].clamp(max=emb.num_embeddings-1)) for i,emb in enumerate(s.embs)]; return s.proj(torch.cat(e+[s.ln(xn)],1))
class LGA(nn.Module):
    def __init__(s,d,h,L,dr):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h; s.Lp=nn.Parameter(torch.randn(L,d)*0.02); s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x): N=x.size(0);L=s.Lp.size(0); q=s.q(x).view(N,s.h,s.hd);k=s.k(s.Lp).view(L,s.h,s.hd);v=s.v(s.Lp).view(L,s.h,s.hd); a=s.dr(F.softmax(torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5),-1)); return s.o(torch.einsum("hnl,lhd->nhd",a,v).reshape(N,s.d))
class AGT(nn.Module):
    def __init__(s,cards,n_num,h,oc,heads,dr,L):
        super().__init__(); s.embed=TabEmbed(cards,n_num,h); s.c1=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.c2=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.g=LGA(h,heads,L,dr); s.n1,s.n2,s.n3=nn.LayerNorm(h),nn.LayerNorm(h),nn.LayerNorm(h); s.head=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Dropout(dr),nn.Linear(h,oc))
    def forward(s,d): x=s.embed(d.x_cat,d.x_num); ei=d.edge_index; x=s.n1(x+F.elu(s.c1(x,ei))); x=s.n2(x+s.g(x)); x=s.n3(x+F.elu(s.c2(x,ei))); return s.head(x)
def mkgraph(num,cat,allf,y,sel):
    Aa=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); Aa=Aa.maximum(Aa.T)
    d=Data(edge_index=torch.tensor(np.array(Aa.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long)); d.x_cat=torch.tensor(cat[sel],dtype=torch.long); d.x_num=torch.tensor(num[sel],dtype=torch.float); d.num_nodes=len(sel); return d

# 1) graph build time vs N
print("=== graph-construction scalability ==="); rows=[]
for N in [10000,20000,40000,80000,160000]:
    sub,_=train_test_split(full,train_size=N,random_state=SEED,stratify=full["class"]); y,ncl,num,cat,cards,allf=prep(sub)
    t=time.time(); Aa=kneighbors_graph(allf,K,mode="connectivity",include_self=False); Aa=Aa.maximum(Aa.T); gt=time.time()-t
    rows.append({"N":N,"edges":int(Aa.nnz),"graph_build_s":round(gt,2)}); print(rows[-1])
gscale=pd.DataFrame(rows)

# 2) AGT train/inference latency + GPU memory + adaptation cost (at N=80k)
sub,_=train_test_split(full,train_size=80000,random_state=SEED,stratify=full["class"]); y,ncl,num,cat,cards,allf=prep(sub)
itr,ite=train_test_split(np.arange(len(sub)),test_size=0.2,random_state=SEED,stratify=y)
gtr=mkgraph(num,cat,allf,y,itr).to(DEV); gte=mkgraph(num,cat,allf,y,ite).to(DEV)
cnt=np.bincount(y[itr],minlength=ncl); w=torch.tensor(len(itr)/(ncl*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
torch.manual_seed(SEED); m=AGT(cards,num.shape[1],HID,ncl,HEADS,DROP,NLAND).to(DEV)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
if DEV.type=="cuda": torch.cuda.reset_peak_memory_stats()
t=time.time()
for ep in range(EPOCHS): m.train();opt.zero_grad();out=m(gtr);loss=F.cross_entropy(out,gtr.y,weight=w);loss.backward();opt.step();sch.step()
train_s=time.time()-t
m.eval()
with torch.no_grad():
    for _ in range(3): m(gte)  # warmup
    if DEV.type=="cuda": torch.cuda.synchronize()
    t=time.time();
    for _ in range(10): m(gte)
    if DEV.type=="cuda": torch.cuda.synchronize()
    inf_ms=(time.time()-t)/10*1000
peak_mb=torch.cuda.max_memory_allocated()/1e6 if DEV.type=="cuda" else float("nan")

# 3) adaptation cost: self-evolving update (15 steps on chunk+buffer ~2300 nodes) vs full XGBoost retrain on train set
chunk=np.random.default_rng(SEED).choice(itr,1500,replace=False); buf=np.random.default_rng(1).choice(itr,800,replace=False)
gse=mkgraph(num,cat,allf,y,np.concatenate([chunk,buf])).to(DEV)
aopt=torch.optim.AdamW(m.parameters(),lr=1.2e-3)
if DEV.type=="cuda": torch.cuda.synchronize()
t=time.time()
for _ in range(15): m.train();aopt.zero_grad();F.cross_entropy(m(gse),gse.y,weight=w).backward();aopt.step()
if DEV.type=="cuda": torch.cuda.synchronize()
se_update_s=time.time()-t
t=time.time(); XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.1,subsample=0.9,eval_metric="mlogloss",random_state=SEED,verbosity=0,tree_method="hist").fit(allf[itr],y[itr]); xgb_retrain_s=time.time()-t

summary={"agt_train_s_80k":round(train_s,2),"agt_inference_ms_16k_nodes":round(inf_ms,1),
         "agt_peak_gpu_mem_MB":round(peak_mb,1) if peak_mb==peak_mb else None,
         "selfevolve_update_s":round(se_update_s,3),"xgboost_full_retrain_s":round(xgb_retrain_s,3),
         "adaptation_speedup_x":round(xgb_retrain_s/max(se_update_s,1e-6),1)}
os.makedirs("runtime_out",exist_ok=True)
gscale.to_csv("runtime_out/graph_scalability.csv",index=False)
json.dump(summary,open("runtime_out/runtime_summary.json","w"),indent=2)
print("\n=== GRAPH SCALABILITY ===\n"+gscale.to_string(index=False))
print("\n=== RUNTIME SUMMARY ===\n"+json.dumps(summary,indent=2))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(7,4.3)); plt.plot(gscale["N"],gscale["graph_build_s"],"o-",color="#3b6fb6")
plt.xlabel("Number of nodes N"); plt.ylabel("k-NN graph build time (s)"); plt.title("k-NN graph construction scalability (X-IIoTID)",weight="bold"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("runtime_out/graph_scalability.png",dpi=300,bbox_inches="tight"); print("saved runtime_out/")
