# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
# Large-scale consistency check: AGT on a larger subsample + XGBoost on FULL data.
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"; HID=128;HEADS=8;DROP=0.2;NLAND=64;EPOCHS=150;LR=3e-3;K=10

full=pd.read_csv(CSV,low_memory=False)
full=full.drop(columns=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in full.columns]).rename(columns={"class2":"class"})
full["class"]=full["class"].astype(str).str.strip()

def prep(df):
    df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
    le=LabelEncoder(); y=le.fit_transform(df["class"]); Xdf=df.drop(columns=["class"])
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
    cat=np.zeros((len(Xdf),len(cat_cols)),dtype=np.int64); cards=[]
    for j,c in enumerate(cat_cols): cd,uq=pd.factorize(Xdf[c]); cat[:,j]=cd; cards.append(len(uq))
    num=StandardScaler().fit_transform(Xdf[num_cols].values).astype(np.float32)
    allf=StandardScaler().fit_transform(np.hstack([num,cat.astype(np.float32)])).astype(np.float32)
    return y,len(le.classes_),num,cat,cards,allf

# --- XGBoost on FULL data ---
print("XGBoost on FULL data (",len(full),"records )...")
y,ncl,num,cat,cards,allf=prep(full)
itr,ite=train_test_split(np.arange(len(full)),test_size=0.2,random_state=SEED,stratify=y)
xgb=XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.1,subsample=0.9,eval_metric="mlogloss",random_state=SEED,verbosity=0,tree_method="hist").fit(allf[itr],y[itr])
yp=xgb.predict(allf[ite])
print(f"  XGBoost FULL: acc={accuracy_score(y[ite],yp):.4f} F1={f1_score(y[ite],yp,average='weighted'):.4f} macroF1={f1_score(y[ite],yp,average='macro'):.4f}")

# --- AGT on larger subsample ---
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

for SZ in [200000,150000,120000]:
    try:
        torch.cuda.empty_cache()
        sub,_=train_test_split(full,train_size=SZ,random_state=SEED,stratify=full["class"])
        y,ncl,num,cat,cards,allf=prep(sub)
        itr,ite=train_test_split(np.arange(len(sub)),test_size=0.2,random_state=SEED,stratify=y)
        def graph(sel):
            A=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); A=A.maximum(A.T)
            d=Data(edge_index=torch.tensor(np.array(A.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long)); d.x_cat=torch.tensor(cat[sel],dtype=torch.long); d.x_num=torch.tensor(num[sel],dtype=torch.float); d.num_nodes=len(sel); return d
        gtr=graph(itr).to(DEV); gte=graph(ite).to(DEV)
        cnt=np.bincount(y[itr],minlength=ncl); w=torch.tensor(len(itr)/(ncl*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
        torch.manual_seed(SEED); m=AGT(cards,num.shape[1],HID,ncl,HEADS,DROP,NLAND).to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        print(f"AGT on {SZ} ( {gtr.edge_index.shape[1]} edges )...")
        t=time.time()
        for ep in range(EPOCHS):
            m.train();opt.zero_grad();out=m(gtr);loss=F.cross_entropy(out,gtr.y,weight=w);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step();sch.step()
        m.eval()
        with torch.no_grad(): pred=m(gte).argmax(1).cpu().numpy()
        yt=gte.y.cpu().numpy()
        print(f"  AGT {SZ}: acc={accuracy_score(yt,pred):.4f} F1={f1_score(yt,pred,average='weighted'):.4f} macroF1={f1_score(yt,pred,average='macro'):.4f} ({time.time()-t:.0f}s)")
        break
    except RuntimeError as e:
        print(f"  AGT {SZ} FAILED: {str(e)[:80]}"); torch.cuda.empty_cache(); continue
