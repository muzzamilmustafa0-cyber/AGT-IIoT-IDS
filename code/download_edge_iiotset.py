# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
import os, glob, pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
D=r"C:\agtwork\datasets\edgeiiot"; os.makedirs(D,exist_ok=True)
if not glob.glob(D+"/**/*.csv",recursive=True):
    print("downloading real Edge-IIoTset (ML CSV) ...")
    for ref in ["mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
                "sibasispradhan/edge-iiotset-dataset"]:
        try:
            api.dataset_download_files(ref,path=D,unzip=True,quiet=True); print("got",ref); break
        except Exception as e: print("fail",ref,str(e)[:60])
csvs=sorted(glob.glob(D+"/**/*.csv",recursive=True),key=lambda f:-os.path.getsize(f))
print("CSV files (top by size):")
for c in csvs[:10]: print("  ",c.replace(D,""),os.path.getsize(c))
# prefer the DNN/ML merged CSV
ml=[c for c in csvs if any(k in os.path.basename(c) for k in ["DNN","ML","Edge"]) ] or csvs
main=ml[0]
df=pd.read_csv(main,nrows=5000,low_memory=False)
print("\nMAIN:",os.path.basename(main),"SHAPE:",df.shape,"NCOLS:",df.shape[1])
print("COLUMNS:",list(df.columns)[:80])
for c in df.columns:
    if any(k in c.lower() for k in ["attack","label","class","type"]):
        try: print(f"LABEL-LIKE {c} ({df[c].nunique()} uniq): {list(df[c].unique())[:18]}")
        except: pass
