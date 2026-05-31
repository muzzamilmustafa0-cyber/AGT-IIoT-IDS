# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
import os, glob, pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
D=r"C:\agtwork\datasets\wustl"; os.makedirs(D,exist_ok=True)
if not glob.glob(D+"/**/*.csv",recursive=True):
    print("downloading WUSTL-IIoT-2021 ...")
    api.dataset_download_files("annaamalaiu/wustl-iiot-2021-dataset",path=D,unzip=True,quiet=True)
csvs=sorted(glob.glob(D+"/**/*.csv",recursive=True),key=lambda f:-os.path.getsize(f))
print("CSV files:")
for c in csvs: print("  ",os.path.basename(c),os.path.getsize(c))
main=csvs[0]
df=pd.read_csv(main,nrows=5000,low_memory=False)
print("\nSHAPE(sample):",df.shape)
print("COLUMNS:",list(df.columns))
# guess label columns
for c in df.columns:
    if df[c].dtype=="object" or df[c].nunique()<25:
        if any(k in c.lower() for k in ["label","traffic","target","attack","class","category"]):
            print(f"LABEL-LIKE {c}: {df[c].value_counts().head(12).to_dict()}")
