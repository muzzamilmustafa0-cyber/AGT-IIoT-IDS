# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""Generalised AGT-vs-baselines runner for any tabular IDS dataset.
Usage: python run_dataset.py --csv PATH --label COL --drop c1,c2 --name NAME --sample 80000
Outputs comparison + per-class + run_summary + figures into multi_out/<NAME>/."""
import os, time, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, GCNConv

ap=argparse.ArgumentParser()
ap.add_argument("--csv",required=True); ap.add_argument("--label",required=True)
ap.add_argument("--drop",default=""); ap.add_argument("--name",required=True)
ap.add_argument("--sample",type=int,default=80000); ap.add_argument("--epochs",type=int,default=150)
A=ap.parse_args()
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HID=128;HEADS=8;DROP=0.2;NLAND=64;LR=3e-3;K=10
OUT=os.path.join("multi_out",A.name); os.makedirs(OUT,exist_ok=True)
log=[]
def L(*a): s=" ".join(str(x) for x in a); print(s); log.append(s)

L(f"=== DATASET {A.name} ===  device {DEV}")
df=pd.read_csv(A.csv,low_memory=False)
drop=[c.strip() for c in A.drop.split(",") if c.strip() and c.strip() in df.columns]
df=df.drop(columns=drop)
df=df.dropna(subset=[A.label]); df[A.label]=df[A.label].astype(str).str.strip()
# drop ultra-rare classes (<10 samples) that break stratification
vc=df[A.label].value_counts(); keep=vc[vc>=10].index; df=df[df[A.label].isin(keep)]
if len(df)>A.sample:
    df,_=train_test_split(df,train_size=A.sample,random_state=SEED,stratify=df[A.label])
df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
le=LabelEncoder(); y=le.fit_transform(df[A.label]); class_names=[str(c) for c in le.classes_]; n_cls=len(class_names)
Xdf=df.drop(columns=[A.label])
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
num_mat=StandardScaler().fit_transform(Xdf[num_cols].values).astype(np.float32) if num_cols else np.zeros((len(Xdf),1),dtype=np.float32)
allf=StandardScaler().fit_transform(np.hstack([num_mat,cat_mat.astype(np.float32)])).astype(np.float32)
idx=np.arange(len(Xdf)); itr,ite=train_test_split(idx,test_size=0.2,random_state=SEED,stratify=y)
L(f"records {len(Xdf)} | classes {n_cls} {class_names} | cat {len(cat_cols)} num {len(num_cols)}")

t0=time.time()
def graph(sel):
    A_=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); A_=A_.maximum(A_.T)
    d=Data(edge_index=torch.tensor(np.array(A_.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long))
    d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long); d.x_num=torch.tensor(num_mat[sel],dtype=torch.float)
    d.x_flat=torch.tensor(allf[sel],dtype=torch.float); d.num_nodes=len(sel); return d
gtr=graph(itr); gte=graph(ite); graph_time=time.time()-t0
L(f"graph build time: {graph_time:.1f}s | avg degree {gtr.edge_index.shape[1]/gtr.num_nodes:.1f}")

class TabEmbed(nn.Module):
    def __init__(s,cards,n_num,d,cap=16):
        super().__init__(); s.embs=nn.ModuleList([nn.Embedding(K_+1,min(cap,max(3,int(np.ceil(np.sqrt(K_)))))) for K_ in cards]); ed=sum(e.embedding_dim for e in s.embs); s.ln=nn.LayerNorm(n_num); s.proj=nn.Sequential(nn.Linear(ed+n_num,d),nn.LayerNorm(d),nn.ReLU())
    def forward(s,xc,xn):
        e=[emb(xc[:,i].clamp(max=emb.num_embeddings-1)) for i,emb in enumerate(s.embs)]; z=torch.cat(e+[s.ln(xn)],1) if e else s.ln(xn); return s.proj(z)
class LGA(nn.Module):
    def __init__(s,d,h,Ln,dr):
        super().__init__(); s.d,s.h,s.hd=d,h,d//h; s.Lp=nn.Parameter(torch.randn(Ln,d)*0.02); s.q,s.k,s.v,s.o=[nn.Linear(d,d) for _ in range(4)]; s.dr=nn.Dropout(dr)
    def forward(s,x): N=x.size(0);Ln=s.Lp.size(0); q=s.q(x).view(N,s.h,s.hd);k=s.k(s.Lp).view(Ln,s.h,s.hd);v=s.v(s.Lp).view(Ln,s.h,s.hd); a=s.dr(F.softmax(torch.einsum("nhd,lhd->hnl",q,k)/(s.hd**0.5),-1)); return s.o(torch.einsum("hnl,lhd->nhd",a,v).reshape(N,s.d))
class AGT(nn.Module):
    def __init__(s,cards,n_num,h,oc,heads,dr,Ln):
        super().__init__(); s.embed=TabEmbed(cards,n_num,h); s.c1=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.c2=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.g=LGA(h,heads,Ln,dr); s.n1,s.n2,s.n3=nn.LayerNorm(h),nn.LayerNorm(h),nn.LayerNorm(h); s.head=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Dropout(dr),nn.Linear(h,oc))
    def forward(s,d): x=s.embed(d.x_cat,d.x_num); ei=d.edge_index; x=s.n1(x+F.elu(s.c1(x,ei))); x=s.n2(x+s.g(x)); x=s.n3(x+F.elu(s.c2(x,ei))); return s.head(x)
class GCN(nn.Module):
    def __init__(s,ic,h,oc,dr): super().__init__(); s.c1=GCNConv(ic,h); s.c2=GCNConv(h,h); s.fc=nn.Linear(h,oc); s.dr=nn.Dropout(dr)
    def forward(s,d): x,ei=d.x_flat,d.edge_index; x=s.dr(F.relu(s.c1(x,ei))); x=s.dr(F.relu(s.c2(x,ei))); return s.fc(x)
cnt=np.bincount(y[itr],minlength=n_cls); w=torch.tensor(len(itr)/(n_cls*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
def train(model):
    model=model.to(DEV); g=gtr.to(DEV); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=A.epochs); t=time.time()
    for ep in range(A.epochs):
        model.train();opt.zero_grad();out=model(g);loss=F.cross_entropy(out,g.y,weight=w);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step();sch.step()
    return model,time.time()-t
def evaluate(model):
    model.eval()
    with torch.no_grad(): t=time.time(); lo=model(gte.to(DEV)); inf=time.time()-t; pr=F.softmax(lo,1).cpu().numpy(); pdc=lo.argmax(1).cpu().numpy()
    return pdc,pr,inf
def row(name,yt,yp,proba=None):
    d={"Model":name,"Accuracy":accuracy_score(yt,yp),"F1-Score":f1_score(yt,yp,average="weighted",zero_division=0),"Macro-F1":f1_score(yt,yp,average="macro",zero_division=0),"MCC":matthews_corrcoef(yt,yp)}
    try: d["ROC-AUC"]=roc_auc_score(yt,proba,multi_class="ovr",average="weighted")
    except Exception: d["ROC-AUC"]=float("nan")
    return d
yt=y[ite]; rows=[]
Xtr,Xte,ytr=allf[itr],allf[ite],y[itr]
for nm,clf in [("Logistic Regression",LogisticRegression(max_iter=400)),
               ("Random Forest",RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=-1)),
               ("XGBoost",XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.1,subsample=0.9,eval_metric="mlogloss",random_state=SEED,verbosity=0,tree_method="hist")),
               ("MLP (Deep NN)",MLPClassifier(hidden_layer_sizes=(256,128),max_iter=250,random_state=SEED))]:
    clf.fit(Xtr,ytr); yp=clf.predict(Xte)
    try: prb=clf.predict_proba(Xte)
    except Exception: prb=None
    rows.append(row(nm,yt,yp,prb)); L(f"  {nm:20s} F1={rows[-1]['F1-Score']:.4f} macroF1={rows[-1]['Macro-F1']:.4f}")
torch.manual_seed(SEED); gcn,_=train(GCN(allf.shape[1],HID,n_cls,DROP)); gp,gpr,_=evaluate(gcn); rows.append(row("GCN",yt,gp,gpr)); L(f"  GCN F1={rows[-1]['F1-Score']:.4f}")
torch.manual_seed(SEED); agt,tt=train(AGT(cards,num_mat.shape[1],HID,n_cls,HEADS,DROP,NLAND)); pred,proba,inf=evaluate(agt); rows.append(row("AGT (proposed)",yt,pred,proba))
npar=sum(p.numel() for p in agt.parameters())
L(f"  AGT F1={rows[-1]['F1-Score']:.4f} macroF1={rows[-1]['Macro-F1']:.4f} | params {npar:,} train {tt:.1f}s inf {inf*1000:.0f}ms")
comp=pd.DataFrame(rows)[["Model","Accuracy","F1-Score","Macro-F1","MCC","ROC-AUC"]].round(4)
comp.to_csv(os.path.join(OUT,"comparison.csv"),index=False)
per=pd.DataFrame(classification_report(yt,pred,target_names=class_names,output_dict=True,zero_division=0)).T
per.loc[class_names][["precision","recall","f1-score","support"]].round(4).to_csv(os.path.join(OUT,"agt_per_class.csv"))
json.dump({"dataset":A.name,"records":int(len(Xdf)),"classes":n_cls,"class_names":class_names,
           "features":int(allf.shape[1]),"cat":len(cat_cols),"num":len(num_cols),
           "graph_build_sec":round(graph_time,2),"agt_train_sec":round(tt,2),"agt_infer_ms":round(inf*1000,1),
           "agt_params":int(npar),"avg_degree":round(gtr.edge_index.shape[1]/gtr.num_nodes,2),
           "agt":comp[comp.Model=="AGT (proposed)"].iloc[0].to_dict()},open(os.path.join(OUT,"summary.json"),"w"),indent=2)
open(os.path.join(OUT,"run.log"),"w",encoding="utf-8").write("\n".join(log))
L("\n"+comp.to_string(index=False)); L("saved ->",OUT)
