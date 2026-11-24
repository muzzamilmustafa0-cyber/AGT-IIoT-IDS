#!/bin/bash
cd "$(dirname "$0")"; mkdir -p logs; export PYTHONIOENCODING=utf-8
until grep -q TUNEDONE logs/tune_drift_done.log 2>/dev/null; do sleep 60; done
python select_drift_cfg.py > logs/select_drift.log 2>&1
cfg() { python -c "import json;c=json.load(open('out/drift/tuning_choice.json'))['$1'];print(c['policy'],c['M'],c['S'],c['eta'])"; }
read pol m s eta <<< "$(cfg AGT-rehearsal)"; TAGX=tuned NO_CFG=1 POLICY=$pol S_STEPS=$s ETA=$eta python exp_drift.py multi 0,1,2,3,4 0.2 $m 0.0 AGT-rehearsal > logs/final_agt.log 2>&1
read pol m s eta <<< "$(cfg MLP-rehearsal)"; TAGX=tunedmlp NO_CFG=1 POLICY=$pol S_STEPS=$s ETA=$eta python exp_drift.py multi 0,1,2,3,4 0.2 $m 0.0 MLP-rehearsal > logs/final_mlp.log 2>&1
read pol m s eta <<< "$(cfg XGB-sliding)"; TAGX=tunedxgb NO_CFG=1 POLICY=$pol S_STEPS=$s ETA=$eta python exp_drift.py multi 0,1,2,3,4 0.2 $m 0.0 XGB-sliding,XGB-continued > logs/final_xgb.log 2>&1
echo FINALDONE >> logs/tune_drift_done.log
