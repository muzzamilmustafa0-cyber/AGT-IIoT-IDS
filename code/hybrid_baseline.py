# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""
GRAPH-REFINED GRADIENT BOOSTING (original hybrid).
XGBoost provides per-node class priors; the AGT graph encoder refines them via
relational message passing (neighbourhood error-correction). Out-of-fold priors
on train (no leakage). Goal: beat XGBoost itself. Real X-IIoTID, same split.
"""
import os, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"
SAMPLE=int(os.environ.get("AGT_SAMPLE","80000")); HID=128; HEADS=8; DROP=0.2; NLAND=64
EPOCHS=int(os.environ.get("AGT_EPOCHS","150")); LR=3e-3; K=10

df=pd.read_csv(CSV, low_memory=False)
df=df.drop(columns=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in df.columns])
df=df.rename(columns={"class2":"class"}); df["class"]=df["class"].astype(str).str.strip()
if len(df)>SAMPLE: df,_=train_test_split(df,train_size=SAMPLE,random_state=SEED,stratify=df["class"])
df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
le=LabelEncoder(); y=le.fit_transform(df["class"]); n_cls=len(le.classes_)
Xdf=df.drop(columns=["class"])
# proper typing
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
# also a pure-numeric matrix for XGBoost (codes for cats)
xgb_feat=np.hstack([num_mat,cat_mat.astype(np.float32)]).astype(np.float32)
idx=np.arange(len(Xdf)); itr,ite=train_test_split(idx,test_size=0.2,random_state=SEED,stratify=y)
print(f"cats {len(cat_cols)} num {len(num_cols)} | {n_cls} classes | device {DEV}")

def newxgb(): return XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.1,subsample=0.9,
    eval_metric="mlogloss",random_state=SEED,verbosity=0,tree_method="hist")

# ---- OOF XGBoost priors (no leakage) ----
print("Building OOF XGBoost priors ...")
prior=np.zeros((len(Xdf),n_cls),dtype=np.float32)
skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
for tr,va in skf.split(xgb_feat[itr],y[itr]):
    tri,vai=itr[tr],itr[va]
    m=newxgb().fit(xgb_feat[tri],y[tri]); prior[vai]=m.predict_proba(xgb_feat[vai])
xgb_full=newxgb().fit(xgb_feat[itr],y[itr])
prior[ite]=xgb_full.predict_proba(xgb_feat[ite])
xgb_pred=xgb_full.predict(xgb_feat[ite])
# standalone xgb metrics
def rep(name,yt,yp,pr=None):
    d={"Model":name,"Accuracy":accuracy_score(yt,yp),
       "F1-Score":f1_score(yt,yp,average="weighted",zero_division=0),
       "Macro-F1":f1_score(yt,yp,average="macro",zero_division=0),
       "MCC":matthews_corrcoef(yt,yp)}
    try:d["ROC-AUC"]=roc_auc_score(yt,pr,multi_class="ovr",average="weighted")
    except Exception:d["ROC-AUC"]=float("nan")
    return d
xgb_row=rep("XGBoost (standalone)",y[ite],xgb_pred,prior[ite])
print(f"  XGBoost standalone: F1={xgb_row['F1-Score']:.4f} macroF1={xgb_row['Macro-F1']:.4f}")

# ---- graph on original features ----
def graph(sel):
    A=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); A=A.maximum(A.T)
    d=Data(edge_index=torch.tensor(np.array(A.nonzero()),dtype=torch.long), y=torch.tensor(y[sel],dtype=torch.long))
    d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long)
    d.x_num=torch.tensor(np.hstack([num_mat[sel],prior[sel]]),dtype=torch.float)  # + XGB priors
    d.num_nodes=len(sel); return d
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
class Hybrid(nn.Module):
    def __init__(s,cards,n_num,h,oc,heads,dr,L):
        super().__init__(); s.embed=TabEmbed(cards,n_num,h)
        s.c1=TransformerConv(h,h//heads,heads=heads,dropout=dr); s.c2=TransformerConv(h,h//heads,heads=heads,dropout=dr)
        s.g=LGA(h,heads,L,dr); s.n1,s.n2,s.n3=nn.LayerNorm(h),nn.LayerNorm(h),nn.LayerNorm(h)
        s.head=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Dropout(dr),nn.Linear(h,oc))
        s.skip=nn.Linear(oc,oc,bias=False)  # learn to keep/adjust XGB prior
    def forward(s,d,prior_logit):
        x=s.embed(d.x_cat,d.x_num); ei=d.edge_index
        x=s.n1(x+F.elu(s.c1(x,ei))); x=s.n2(x+s.g(x)); x=s.n3(x+F.elu(s.c2(x,ei)))
        return s.head(x)+s.skip(prior_logit)   # residual over XGB prior
n_num_in=num_mat.shape[1]+n_cls
cnt=np.bincount(y[itr],minlength=n_cls); w=torch.tensor(len(itr)/(n_cls*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
pri_tr=torch.tensor(np.log(np.clip(prior[itr],1e-6,1)),dtype=torch.float).to(DEV)
pri_te=torch.tensor(np.log(np.clip(prior[ite],1e-6,1)),dtype=torch.float).to(DEV)
torch.manual_seed(SEED)
m=Hybrid(cards,n_num_in,HID,n_cls,HEADS,DROP,NLAND).to(DEV); g=gtr.to(DEV); gt=gte.to(DEV)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
print("Training hybrid (graph-refined XGBoost) ..."); t=time.time()
for ep in range(EPOCHS):
    m.train();opt.zero_grad();out=m(g,pri_tr);loss=F.cross_entropy(out,g.y,weight=w)
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.0);opt.step();sch.step()
m.eval()
with torch.no_grad():
    logit=m(gt,pri_te);prob=F.softmax(logit,1).cpu().numpy();pred=logit.argmax(1).cpu().numpy()
print(f"  hybrid params {sum(p.numel() for p in m.parameters()):,} | train {time.time()-t:.0f}s")
hy_row=rep("AGT-Hybrid (graph-refined GBDT)",y[ite],pred,prob)

res=pd.DataFrame([hy_row,xgb_row])[["Model","Accuracy","F1-Score","Macro-F1","MCC","ROC-AUC"]].round(4)
os.makedirs("hybrid_out",exist_ok=True); res.to_csv("hybrid_out/hybrid_comparison.csv",index=False)
print("\n==== GRAPH-REFINED HYBRID vs XGBoost ===="); print(res.to_string(index=False))
print(f"\nDelta (hybrid - xgb): weightedF1 {hy_row['F1-Score']-xgb_row['F1-Score']:+.4f} | "
      f"macroF1 {hy_row['Macro-F1']-xgb_row['Macro-F1']:+.4f}")
