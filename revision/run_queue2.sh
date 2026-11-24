#!/bin/bash
# resumable queue: a step listed in logs/queue.log as finished with exit 0 is skipped
cd "$(dirname "$0")"; mkdir -p logs
export PYTHONIOENCODING=utf-8
S10=0,1,2,3,4,5,6,7,8,9
step() { name="$1"; shift
  if grep -q "^$name 0$" logs/queue.log 2>/dev/null; then return; fi
  echo "$(date +%H:%M) start $name" >> logs/queue_times.log
  "$@" > "logs/$name.log" 2>&1; echo "$name $?" >> logs/queue.log
  echo "$(date +%H:%M) end $name" >> logs/queue_times.log; }
step main python exp_stationary.py main $S10
step esage python exp_egraphsage.py main $S10
step drift_multi python exp_drift.py multi 0,1,2,3,4
step ablation python exp_stationary.py ablation $S10
step noise python exp_stationary.py noise 0,1,2,3,4
step sens python exp_stationary.py sens 0,1,2,3,4
step importance python exp_stationary.py importance
step drift_binary python exp_drift.py binary 0,1,2,3,4 0.2 2500 0.0 AGT-static,AGT-rehearsal,MLP-static,MLP-rehearsal,XGB-static,XGB-continued,XGB-sliding,XGB-retrain,INSOMNIA-style
for r in 0.05 0.1 0.5; do step drift_rho$r python exp_drift.py multi 0,1,2 $r 2500 0.0 AGT-rehearsal,MLP-rehearsal,XGB-sliding,XGB-retrain; done
for m in 1000 5000 10000; do step drift_M$m python exp_drift.py multi 0,1,2 0.2 $m 0.0 AGT-rehearsal,MLP-rehearsal,XGB-sliding; done
for e in 0.1 0.2 0.3; do step drift_eps$e python exp_drift.py multi 0,1,2 0.2 2500 $e AGT-rehearsal,MLP-rehearsal,XGB-sliding,XGB-retrain; done
for d in Edge-IIoTset WUSTL-IIoT-2021 CICIoT2023; do step ds_$d env DS=$d python exp_stationary.py main 0,1,2,3,4; done
for s in memory throughput retrain knn; do step hpc_$s python exp_hpc.py $s; done
step fullcorpus python exp_fullcorpus.py 0,1,2
step aggregate python aggregate.py main main:Edge-IIoTset main:WUSTL-IIoT-2021 main:CICIoT2023 ablation sens noise
echo ALLDONE >> logs/queue.log
