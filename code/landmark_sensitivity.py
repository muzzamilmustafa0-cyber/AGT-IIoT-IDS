# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""Sensitivity of AGT to the number of landmark tokens L (concern 7),
plus a no-global-attention reference. Real X-IIoTID, 80k."""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"; SAMPLE=80000
HID=128;HEADS=8;DROP=0.2;EPOCHS=150;LR=3e-3;K=10

df=pd.read_csv(CSV,low_memory=False)
df=df.drop(columns=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in df.columns]).rename(columns={"class2":"class"})
df["class"]=df["class"].astype(str).str.strip()
df,_=train_test_split(df,train_size=SAMPLE,random_state=SEED,stratify=df["class"]); df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
le=LabelEncoder(); y=le.fit_transform(df["class"]); n_cls=len(le.classes_); Xdf=df.drop(columns=["class"])
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
def graph(sel):
    Aa=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); Aa=Aa.maximum(Aa.T)
    d=Data(edge_index=torch.tensor(np.array(Aa.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long)); d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long); d.x_num=torch.tensor(num_mat[sel],dtype=torch.float); d.num_nodes=len(sel); return d
gtr=graph(itr); gte=graph(ite)
class TabEmbed(nn.Module):
    def __init__(s,cards,n_num,d,cap=16):
        super().__init__(); s.embs=nn.ModuleList([nn.Embedding(K_+1,min(cap,max(3,int(np.ceil(np.sqrt(K_)))))) for K_ in cards]); ed=sum(e.embedding_dim for e in s.embs); s.ln=nn.LayerNorm(n_num); s.proj=nn.Sequential(nn.Linear(ed+n_num,d),nn.LayerNorm(d),nn.ReLU())
    def forward(s,xc,xn): e=[emb(xc[:,i].clamp(max=emb.num_embeddings-1)) for i,emb in enumerate(s.embs)]; return s.proj(torch.cat(e+[s.ln(xn)],1))
class LGA(nn.Module):
    def __init__(s,d,h,L,dr):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h; s.Lp=nn.Parameter(torch.randn(L,d)*0.02); s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x): N=x.size(0);L=s.Lp.size(0); q=s.q(x).view(N,s.h,s.hd);k=s.k(s.Lp).view(L,s.h,s.hd);v=s.v(s.Lp).view(L,s.h,s.hd); a=s.dr(F.softmax(torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5),-1)); return s.o(torch.einsum("hnl,lhd->nhd",a,v).reshape(N,s.d))
class AGT(nn.Module):
    def __init__(s,cards,n_num,h,oc,heads,dr,L,use_global=True):
        super().__init__(); s.embed=TabEmbed(cards,n_num,h); s.c1=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.c2=TransformerConv(h,h//heads,heads=heads,dropout=dr)
        s.use_global=use_global; s.g=LGA(h,heads,max(L,1),dr) if use_global else None; s.n1,s.n2,s.n3=nn.LayerNorm(h),nn.LayerNorm(h),nn.LayerNorm(h); s.head=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Dropout(dr),nn.Linear(h,oc))
    def forward(s,d):
        x=s.embed(d.x_cat,d.x_num); ei=d.edge_index; x=s.n1(x+F.elu(s.c1(x,ei)))
        if s.use_global: x=s.n2(x+s.g(x))
        x=s.n3(x+F.elu(s.c2(x,ei))); return s.head(x)
cnt=np.bincount(y[itr],minlength=n_cls); w=torch.tensor(len(itr)/(n_cls*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
def run(L,use_global=True):
    torch.manual_seed(SEED); m=AGT(cards,len(num_cols),HID,n_cls,HEADS,DROP,L,use_global).to(DEV); g=gtr.to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); t=time.time()
    for ep in range(EPOCHS): m.train();opt.zero_grad();out=m(g);loss=F.cross_entropy(out,g.y,weight=w);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step();sch.step()
    m.eval()
    with torch.no_grad(): pred=m(gte.to(DEV)).argmax(1).cpu().numpy()
    yt=y[ite]; return dict(L=(L if use_global else 0),F1=round(f1_score(yt,pred,average="weighted",zero_division=0),4),MacroF1=round(f1_score(yt,pred,average="macro",zero_division=0),4),Acc=round(accuracy_score(yt,pred),4),train_s=round(time.time()-t,1))
rows=[run(0,use_global=False)]
for L in [16,32,64,128,256]: rows.append(run(L,True)); print(rows[-1])
res=pd.DataFrame(rows); os.makedirs("landmark_out",exist_ok=True); res.to_csv("landmark_out/landmark_sensitivity.csv",index=False)
print("\n=== LANDMARK SENSITIVITY (L=0 means no global attention) ===\n"+res.to_string(index=False))
plt.figure(figsize=(7,4.3)); g2=res[res.L>0]
plt.plot(g2["L"],g2["MacroF1"],"o-",label="Macro-F1"); plt.plot(g2["L"],g2["F1"],"s--",label="Weighted F1")
plt.axhline(res[res.L==0]["MacroF1"].iloc[0],color="grey",ls=":",label="no global attn (macro-F1)")
plt.xscale("log",base=2); plt.xlabel("Number of landmark tokens L"); plt.ylabel("Score"); plt.title("AGT sensitivity to landmark count L",weight="bold"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("landmark_out/landmark_sensitivity.png",dpi=300,bbox_inches="tight")
print("saved landmark_out/")
