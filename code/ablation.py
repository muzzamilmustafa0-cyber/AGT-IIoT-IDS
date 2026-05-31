# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""Ablation of the CURRENT AGT (entity embeddings + local graph attn + landmark
global attn + 2 layers + class-weighted loss). Real X-IIoTID, same split."""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"; SAMPLE=80000
HID=128; HEADS=8; DROP=0.2; NLAND=64; EPOCHS=150; LR=3e-3; K=10

df=pd.read_csv(CSV, low_memory=False)
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
for j,c in enumerate(cat_cols):
    cd,uq=pd.factorize(Xdf[c]); cat_mat[:,j]=cd; cards.append(len(uq))
num_mat=StandardScaler().fit_transform(Xdf[num_cols].values).astype(np.float32)
allf=StandardScaler().fit_transform(np.hstack([num_mat,cat_mat.astype(np.float32)])).astype(np.float32)
idx=np.arange(len(Xdf)); itr,ite=train_test_split(idx,test_size=0.2,random_state=SEED,stratify=y)
print(f"cats {len(cat_cols)} num {len(num_cols)} | {n_cls} classes")

def graph(sel):
    A=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); A=A.maximum(A.T)
    d=Data(edge_index=torch.tensor(np.array(A.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long))
    d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long); d.x_num=torch.tensor(num_mat[sel],dtype=torch.float)
    d.x_flat=torch.tensor(allf[sel],dtype=torch.float); d.num_nodes=len(sel); return d
gtr=graph(itr); gte=graph(ite)

class TabEmbed(nn.Module):
    def __init__(s,cards,n_num,d,cap=16):
        super().__init__(); s.embs=nn.ModuleList([nn.Embedding(K_+1,min(cap,max(3,int(np.ceil(np.sqrt(K_)))))) for K_ in cards])
        ed=sum(e.embedding_dim for e in s.embs); s.ln=nn.LayerNorm(n_num)
        s.proj=nn.Sequential(nn.Linear(ed+n_num,d),nn.LayerNorm(d),nn.ReLU())
    def forward(s,xc,xn):
        e=[emb(xc[:,i].clamp(max=emb.num_embeddings-1)) for i,emb in enumerate(s.embs)]
        return s.proj(torch.cat(e+[s.ln(xn)],1))
class LGA(nn.Module):
    def __init__(s,d,h,L,dr):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h; s.Lp=nn.Parameter(torch.randn(L,d)*0.02)
        s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x):
        N=x.size(0);L=s.Lp.size(0)
        q=s.q(x).view(N,s.h,s.hd);k=s.k(s.Lp).view(L,s.h,s.hd);v=s.v(s.Lp).view(L,s.h,s.hd)
        a=s.dr(F.softmax(torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5),-1))
        return s.o(torch.einsum("hnl,lhd->nhd",a,v).reshape(N,s.d))
class Cfg(nn.Module):
    def __init__(s,use_embed,use_global,two_layer,use_graph,n_flat):
        super().__init__(); s.use_embed,s.use_global,s.two,s.use_graph=use_embed,use_global,two_layer,use_graph
        if use_embed: s.embed=TabEmbed(cards,len(num_cols),HID)
        else: s.embed=nn.Sequential(nn.Linear(n_flat,HID),nn.LayerNorm(HID),nn.ReLU())
        if use_graph:
            s.c1=TransformerConv(HID,HID//HEADS,heads=HEADS,dropout=DROP)
            if two_layer: s.c2=TransformerConv(HID,HID//HEADS,heads=HEADS,dropout=DROP)
        else:
            s.c1=nn.Linear(HID,HID)
            if two_layer: s.c2=nn.Linear(HID,HID)
        if use_global: s.g=LGA(HID,HEADS,NLAND,DROP)
        s.n1,s.n2,s.n3=nn.LayerNorm(HID),nn.LayerNorm(HID),nn.LayerNorm(HID)
        s.head=nn.Sequential(nn.Linear(HID,HID),nn.ELU(),nn.Dropout(DROP),nn.Linear(HID,n_cls))
    def conv(s,layer,x,ei): return layer(x,ei) if s.use_graph else layer(x)
    def forward(s,d):
        x=s.embed(d.x_cat,d.x_num) if s.use_embed else s.embed(d.x_flat); ei=d.edge_index
        x=s.n1(x+F.elu(s.conv(s.c1,x,ei)))
        if s.use_global: x=s.n2(x+s.g(x))
        if s.two: x=s.n3(x+F.elu(s.conv(s.c2,x,ei)))
        return s.head(x)

cnt=np.bincount(y[itr],minlength=n_cls); w=torch.tensor(len(itr)/(n_cls*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
def run(name,**kw):
    torch.manual_seed(SEED); m=Cfg(n_flat=allf.shape[1],**kw).to(DEV); g=gtr.to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    for _ in range(EPOCHS):
        m.train();opt.zero_grad();loss=F.cross_entropy(m(g),g.y,weight=w);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step();sch.step()
    m.eval()
    with torch.no_grad(): p=m(gte.to(DEV)).argmax(1).cpu().numpy()
    yt=y[ite]
    r=dict(Variant=name,Accuracy=round(accuracy_score(yt,p),4),F1=round(f1_score(yt,p,average="weighted",zero_division=0),4),
           MacroF1=round(f1_score(yt,p,average="macro",zero_division=0),4),MCC=round(matthews_corrcoef(yt,p),4))
    print(f"  {name:42s} F1={r['F1']:.4f} macroF1={r['MacroF1']:.4f}"); return r

print("Running ablation ...")
rows=[
 run("Full AGT (embed+local+global, 2-layer)",use_embed=1,use_global=1,two_layer=1,use_graph=1),
 run("(-) landmark global attention",        use_embed=1,use_global=0,two_layer=1,use_graph=1),
 run("(-) entity embeddings",                use_embed=0,use_global=1,two_layer=1,use_graph=1),
 run("(-) second graph layer (shallow)",     use_embed=1,use_global=1,two_layer=0,use_graph=1),
 run("(-) graph topology (MLP only)",        use_embed=1,use_global=0,two_layer=1,use_graph=0),
]
res=pd.DataFrame(rows)
os.makedirs("ablation_out",exist_ok=True); res.to_csv("ablation_out/ablation_new_agt.csv",index=False)
print("\n==== ABLATION (current AGT, real X-IIoTID) ===="); print(res.to_string(index=False))
