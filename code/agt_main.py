# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
"""
AGT — definitive main pipeline (real X-IIoTID, current AGT design:
entity embeddings + local graph attention + landmark global attention +
class-weighted loss). Produces the consistent set of paper tables & figures.
"""
import os, time, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style="whitegrid", context="paper"); PAL="viridis"
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

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV=r"C:\agtwork\xiiotid\X-IIoTID dataset.csv"; SAMPLE=int(os.environ.get("AGT_SAMPLE","80000"))
HID=128; HEADS=8; DROP=0.2; NLAND=64; EPOCHS=int(os.environ.get("AGT_EPOCHS","150")); LR=3e-3; K=10
OUT=os.environ.get("AGT_OUT","main_artifacts"); FIG=os.path.join(OUT,"figures"); os.makedirs(FIG,exist_ok=True)

# ---------- data ----------
df=pd.read_csv(CSV, low_memory=False)
df=df.drop(columns=[c for c in ["class1","class3","Date","Timestamp","Scr_IP","Des_IP"] if c in df.columns]).rename(columns={"class2":"class"})
df["class"]=df["class"].astype(str).str.strip()
df,_=train_test_split(df,train_size=SAMPLE,random_state=SEED,stratify=df["class"]); df=df.reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
le=LabelEncoder(); y=le.fit_transform(df["class"]); class_names=[str(c) for c in le.classes_]; n_cls=len(class_names)
Xdf=df.drop(columns=["class"])
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
print(f"records {len(Xdf)} | {n_cls} classes | cats {len(cat_cols)} num {len(num_cols)} | device {DEV}")

# class distribution table (Table II)
cdist=df["class"].value_counts().reset_index(); cdist.columns=["Class","Count"]
cdist["Percentage"]=(cdist["Count"]/cdist["Count"].sum()*100).round(2)
cdist.to_csv(os.path.join(OUT,"table2_class_distribution.csv"),index=False)

# ---------- graph ----------
def graph(sel):
    A=kneighbors_graph(allf[sel],K,mode="connectivity",include_self=False); A=A.maximum(A.T)
    d=Data(edge_index=torch.tensor(np.array(A.nonzero()),dtype=torch.long),y=torch.tensor(y[sel],dtype=torch.long))
    d.x_cat=torch.tensor(cat_mat[sel],dtype=torch.long); d.x_num=torch.tensor(num_mat[sel],dtype=torch.float)
    d.x_flat=torch.tensor(allf[sel],dtype=torch.float); d.num_nodes=len(sel); return d
gtr=graph(itr); gte=graph(ite); avg_deg=gtr.edge_index.shape[1]/gtr.num_nodes

# ---------- models ----------
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
class GCN(nn.Module):
    def __init__(s,ic,h,oc,dr):
        super().__init__(); s.c1=GCNConv(ic,h); s.c2=GCNConv(h,h); s.fc=nn.Linear(h,oc); s.dr=nn.Dropout(dr)
    def forward(s,d):
        x,ei=d.x_flat,d.edge_index; x=s.dr(F.relu(s.c1(x,ei))); x=s.dr(F.relu(s.c2(x,ei))); return s.fc(x)

cnt=np.bincount(y[itr],minlength=n_cls); w=torch.tensor(len(itr)/(n_cls*np.maximum(cnt,1)),dtype=torch.float).to(DEV)
def train(model,epochs=EPOCHS,log=True):
    model=model.to(DEV); g=gtr.to(DEV); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); hist={"loss":[],"acc":[]}; t=time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad(); out=model(g); loss=F.cross_entropy(out,g.y,weight=w)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); sch.step()
        with torch.no_grad(): hist["loss"].append(float(loss)); hist["acc"].append(accuracy_score(g.y.cpu(),out.argmax(1).cpu()))
    return model,hist,time.time()-t
def evaluate(model):
    model.eval()
    with torch.no_grad(): lo=model(gte.to(DEV)); pr=F.softmax(lo,1).cpu().numpy(); pd_=lo.argmax(1).cpu().numpy()
    return pd_,pr
def mrow(name,yt,yp,proba=None):
    d={"Model":name,"Accuracy":accuracy_score(yt,yp),"Precision":precision_score(yt,yp,average="weighted",zero_division=0),
       "Recall":recall_score(yt,yp,average="weighted",zero_division=0),"F1-Score":f1_score(yt,yp,average="weighted",zero_division=0),
       "Macro-F1":f1_score(yt,yp,average="macro",zero_division=0),"MCC":matthews_corrcoef(yt,yp)}
    try: d["ROC-AUC"]=roc_auc_score(yt,proba,multi_class="ovr",average="weighted")
    except Exception: d["ROC-AUC"]=float("nan")
    return d

print("Training AGT ..."); torch.manual_seed(SEED)
agt=AGT(cards,len(num_cols),HID,n_cls,HEADS,DROP,NLAND); agt,hist,ttime=train(agt)
n_params=sum(p.numel() for p in agt.parameters()); print(f"  AGT params {n_params:,} | {ttime:.0f}s")
yt=y[ite]; pred,proba=evaluate(agt); agt_row=mrow("AGT (Proposed)",yt,pred,proba)

# training curves (Fig)
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(hist["loss"],color="#440154",lw=1.8); ax[0].set_title("(a) Training loss",weight="bold"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy")
ax[1].plot(hist["acc"],color="#21918c",lw=1.8); ax[1].set_title("(b) Training accuracy",weight="bold"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_training_curves.png"),dpi=300,bbox_inches="tight"); plt.close()

# per-class table (Table IV) + confusion + per-class F1
rep=classification_report(yt,pred,target_names=class_names,output_dict=True,zero_division=0)
per=pd.DataFrame(rep).T.loc[class_names][["precision","recall","f1-score","support"]].round(4)
per.to_csv(os.path.join(OUT,"table4_per_class.csv"))
cm=confusion_matrix(yt,pred)
plt.figure(figsize=(8.4,6.8)); sns.heatmap(cm,annot=True,fmt="d",cmap="viridis",xticklabels=class_names,yticklabels=class_names,linewidths=.4,linecolor="white",cbar_kws={"label":"Count"})
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix — AGT",weight="bold"); plt.xticks(rotation=40,ha="right",fontsize=8); plt.yticks(rotation=0,fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig4_confusion_matrix.png"),dpi=300,bbox_inches="tight"); plt.close()
pc=per.reset_index().rename(columns={"index":"Class"})
plt.figure(figsize=(9,4.2)); sns.barplot(data=pc,x="Class",y="f1-score",palette=PAL,hue="Class",legend=False)
plt.ylim(0,1.08); plt.ylabel("F1-score"); plt.xlabel(""); plt.xticks(rotation=40,ha="right")
for i,v in enumerate(pc["f1-score"]): plt.text(i,v+0.02,f"{v:.3f}",ha="center",fontsize=7.5)
plt.title("Per-class detection performance — AGT",weight="bold"); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig5_per_class_f1.png"),dpi=300,bbox_inches="tight"); plt.close()

# baselines
print("Training baselines ..."); rows=[]
Xtr,Xte,ytr=allf[itr],allf[ite],y[itr]
for name,clf in [("Logistic Regression",LogisticRegression(max_iter=500)),
                 ("Random Forest",RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=-1)),
                 ("XGBoost",XGBClassifier(n_estimators=300,max_depth=7,learning_rate=0.1,subsample=0.9,eval_metric="mlogloss",random_state=SEED,verbosity=0,tree_method="hist")),
                 ("MLP (Deep NN)",MLPClassifier(hidden_layer_sizes=(256,128),max_iter=250,random_state=SEED))]:
    clf.fit(Xtr,ytr); yp=clf.predict(Xte)
    try: prb=clf.predict_proba(Xte)
    except Exception: prb=None
    rows.append(mrow(name,yt,yp,prb)); print(f"  {name:20s} F1={rows[-1]['F1-Score']:.4f}")
    if name=="XGBoost":  # feature importance (Table V + Fig)
        imp=pd.DataFrame({"Feature":list(Xdf.columns),"Importance":clf.feature_importances_}).sort_values("Importance",ascending=False)
        imp.head(10).to_csv(os.path.join(OUT,"table5_feature_importance.csv"),index=False)
        plt.figure(figsize=(8,5)); top=imp.head(15)[::-1]; plt.barh(top["Feature"],top["Importance"],color=plt.cm.viridis(np.linspace(0.1,0.9,15)))
        plt.xlabel("Gradient-boosting gain"); plt.title("Top-15 discriminative features",weight="bold"); plt.tight_layout()
        plt.savefig(os.path.join(FIG,"fig8_feature_importance.png"),dpi=300,bbox_inches="tight"); plt.close()
print("Training GCN ..."); torch.manual_seed(SEED); gcn,_,_=train(GCN(allf.shape[1],HID,n_cls,DROP),log=False)
gp,gpr=evaluate(gcn); rows.append(mrow("GCN",yt,gp,gpr)); print(f"  GCN F1={rows[-1]['F1-Score']:.4f}")

comp=pd.DataFrame(rows+[agt_row])[["Model","Accuracy","Precision","Recall","F1-Score","Macro-F1","MCC","ROC-AUC"]].round(4)
comp.to_csv(os.path.join(OUT,"table3_model_comparison.csv"),index=False)
melt=comp.melt(id_vars="Model",value_vars=["Accuracy","Precision","Recall","F1-Score","Macro-F1"],var_name="Metric",value_name="Score")
plt.figure(figsize=(12,5)); sns.barplot(data=melt,x="Model",y="Score",hue="Metric",palette=PAL)
plt.ylim(0,1.05); plt.xticks(rotation=20,ha="right"); plt.title("Performance comparison: baselines vs AGT",weight="bold")
plt.legend(fontsize=8,ncol=5,loc="lower center",bbox_to_anchor=(0.5,-0.34)); plt.tight_layout()
plt.savefig(os.path.join(FIG,"fig7_model_comparison.png"),dpi=300,bbox_inches="tight"); plt.close()

# correlation heatmap (numeric, first 30)
corr=pd.DataFrame(num_mat[:, :30]).corr()
plt.figure(figsize=(9,7.5)); sns.heatmap(corr,cmap="RdBu_r",center=0,square=True,cbar_kws={"shrink":0.7,"label":"Pearson r"})
plt.xticks(fontsize=6,rotation=90); plt.yticks(fontsize=6); plt.title("Feature correlation (first 30 numeric attributes)",weight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_correlation.png"),dpi=300,bbox_inches="tight"); plt.close()

summary={"data_source":"REAL X-IIoTID (Kaggle: munaalhawawreh/xiiotid-iiot-intrusion-dataset)",
    "records":int(len(Xdf)),"sample_of_total":820834,"n_classes":n_cls,"class_names":class_names,
    "n_features":int(allf.shape[1]),"categorical":len(cat_cols),"numeric":len(num_cols),
    "avg_graph_degree":round(float(avg_deg),2),"agt_params":int(n_params),"agt_train_sec":round(ttime,1),
    "agt_metrics":{k:round(float(v),4) for k,v in agt_row.items() if k!="Model"}}
json.dump(summary,open(os.path.join(OUT,"run_summary.json"),"w"),indent=2)
print("\n==== MODEL COMPARISON (current AGT, real X-IIoTID) ===="); print(comp.to_string(index=False))
print("\nArtifacts in",OUT)
