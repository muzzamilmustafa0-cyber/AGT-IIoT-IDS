# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""Per-class behaviour under drift (concern 8): multi-class AGT, per-class F1 as
Gaussian covariate-shift magnitude sigma grows. Focus on minority attack classes."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"; SAMPLE=80000
HID=128;HEADS=8;DROP=0.2;NLAND=64;EPOCHS=150;LR=3e-3;K=10
df=pd.read_csv(CSV,low_memory=False)
df=df.drop(columns=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in df.columns]).rename(columns={"class2":"class"})
df["class"]=df["class"].astype(str).str.strip()
df,_=train_test_split(df,train_size=SAMPLE,random_state=SEED,stratify=df["class"]); df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
le=LabelEncoder(); y=le.fit_transform(df["class"]); class_names=[str(c) for c in le.classes_]; n_cls=len(class_names); Xdf=df.drop(columns=["class"])
num_cols=[];cat_cols=[]
for c in Xdf.columns:
    col=Xdf[c]
    if col.dtype!="object": num_cols.append(c); continue
    co=pd.to_numeric(col.astype(str).str.replace(",",""),errors="coerce")
    if co.notna().mean()>0.98: Xdf[c]=co; num_cols.append(c)
    elif col.nunique()<=64: cat_cols.append(c)
    else: Xdf[c]=col.map(col.value_counts()).astype(float); num_cols.append(c)
for c in num_cols: Xdf[c]=pd.to_numeric(Xdf[c],errors="coerce").fillna(pd.to_numeric(Xdf[c],errors="coerce").median())
for c in cat_cols: Xdf[c]=Xdf[c].fillna(Xdf[c].mode()[0]).astype(str)
cat_mat=np.zeros((len(Xdf),len(cat_cols)),dtype=np.int64); cards=[]
for j,c in enumerate(cat_cols): cd,uq=pd.factorize(Xdf[c]); cat_mat[:,j]=cd; cards.append(len(uq))
num_mat=StandardScaler().fit_transform(Xdf[num_cols].values).astype(np.float32)
allf=StandardScaler().fit_transform(np.hstack([num_mat,cat_mat.astype(np.float32)])).astype(np.float32)
idx=np.arange(len(Xdf)); itr,ite=train_test_split(idx,test_size=0.2,random_state=SEED,stratify=y)
def graph_idx(sel,xnum=None):
    Aa=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); Aa=Aa.maximum(Aa.T)
    d=Data(edge_index=torch.tensor(np.array(Aa.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long)); d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long)
    d.x_num=torch.tensor(num_mat[sel] if xnum is None else xnum,dtype=torch.float); d.num_nodes=len(sel); return d
gtr=graph_idx(itr)
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
cnt=np.bincount(y[itr],minlength=n_cls); w=torch.tensor(len(itr)/(n_cls*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
torch.manual_seed(SEED); m=AGT(cards,len(num_cols),HID,n_cls,HEADS,DROP,NLAND).to(DEV); g=gtr.to(DEV)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
for ep in range(EPOCHS): m.train();opt.zero_grad();out=m(g);loss=F.cross_entropy(out,g.y,weight=w);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step();sch.step()
m.eval(); rng=np.random.default_rng(SEED)
rows=[]
for sig in [0.0,0.1,0.2,0.3,0.4,0.5]:
    xn=num_mat[ite]+rng.normal(0,sig,num_mat[ite].shape).astype(np.float32)
    d=graph_idx(ite,xnum=xn).to(DEV)
    with torch.no_grad(): pred=m(d).argmax(1).cpu().numpy()
    per=f1_score(y[ite],pred,average=None,labels=range(n_cls),zero_division=0)
    rows.append(dict(sigma=sig,**{class_names[i]:round(float(per[i]),3) for i in range(n_cls)}))
res=pd.DataFrame(rows); os.makedirs("perclass_out",exist_ok=True); res.to_csv("perclass_out/perclass_drift.csv",index=False)
print(res.to_string(index=False))
plt.figure(figsize=(8.5,5))
focus=["Normal","RDOS","Reconnaissance","Lateral _movement","C&C","Exploitation","crypto-ransomware"]
focus=[c for c in focus if c in class_names]
for c in focus: plt.plot(res["sigma"],res[c],"o-",label=c)
plt.xlabel("Covariate-shift magnitude sigma"); plt.ylabel("Per-class F1"); plt.title("Per-class robustness under drift (AGT, X-IIoTID)",weight="bold")
plt.legend(fontsize=8,loc="lower left"); plt.grid(alpha=0.3); plt.ylim(0,1.02); plt.tight_layout()
plt.savefig("perclass_out/perclass_drift.png",dpi=300,bbox_inches="tight"); print("saved perclass_out/")
