#!/bin/bash
# leakage-free tuning of the update settings on a pseudo-stream from the initial period
cd "$(dirname "$0")"; mkdir -p logs; export PYTHONIOENCODING=utf-8 PSEUDO=1 NO_CFG=1
for pol in recent balanced; do for m in 2500 5000; do for s in 15 30; do for eta in 0.0012 0.003; do
  f="out/drift/pseudo_multi_rho0.2_M${m}_eps0.0"; [ "$pol" != recent ] || [ "$s" != 15 ] || [ "$eta" != 0.0012 ] && f="${f}_${pol}_S${s}_eta${eta}"
  if [ -f "${f}_seed2.json" ]; then continue; fi
  POLICY=$pol S_STEPS=$s ETA=$eta python exp_drift.py multi 0,1,2 0.2 $m 0.0 AGT-rehearsal,MLP-rehearsal,XGB-sliding >> logs/tune_drift.log 2>&1
  echo "$pol $m $s $eta $?" >> logs/tune_drift_done.log
done; done; done; done
echo TUNEDONE >> logs/tune_drift_done.log
