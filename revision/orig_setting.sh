#!/bin/bash
# binary task with the submitted paper's drift settings (55% initial, 8 windows, 50% labels, M=2500, S=15, eta=1.2e-3)
cd "$(dirname "$0")"; mkdir -p logs; export PYTHONIOENCODING=utf-8
STRIDE=7 INIT_FRAC=0.55 NWIN=8 NO_CFG=1 TAGX=origsetting python exp_drift.py binary 0,1,2,3,4 0.5 2500 0.0 AGT-static,AGT-rehearsal,MLP-static,MLP-rehearsal,XGB-static,XGB-retrain > logs/orig_setting.log 2>&1
echo ORIGDONE >> logs/orig_setting_done.log
