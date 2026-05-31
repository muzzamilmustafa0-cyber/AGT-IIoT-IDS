# -*- coding: utf-8 -*-
# Author: Muzzamil Mustafa
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv(r"C:\agtwork\datasets\edge\IoT_Intrusion.csv",low_memory=False)
def grp(l):
    l=str(l)
    if l.startswith("DDoS"): return "DDoS"
    if l.startswith("DoS"): return "DoS"
    if l.startswith("Mirai"): return "Mirai"
    if l.startswith("Recon") or l=="VulnerabilityScan": return "Recon"
    if "Spoof" in l or l=="MITM-ArpSpoofing": return "Spoofing"
    if l=="BenignTraffic": return "Benign"
    if "BruteForce" in l: return "BruteForce"
    return "Web"
df["label"]=df["label"].map(grp)
print("8-class distribution:\n", df["label"].value_counts())
sub,_=train_test_split(df,train_size=120000,random_state=42,stratify=df["label"])
sub.to_csv(r"C:\agtwork\datasets\edge\CICIoT2023_8class.csv",index=False)
print("saved 8-class sample:",sub.shape)
