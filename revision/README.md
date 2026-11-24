# Revision experiments (Journal of Supercomputing, revision 1)

This folder reproduces every table and figure of the revised manuscript. The code in `../code` is the
version of the initial submission and is kept for reference.

## Setup

1. Install the requirements of the repository (`../requirements.txt`); PyTorch with CUDA and
   PyTorch Geometric are needed for the graph models, `faiss-cpu` for the CPU neighbor-search comparison.
2. Download the datasets (see `../code/download_*.py` and `../code/prepare_ciciot2023.py`) and set
   `data_root` in `paths.json`. The expected file layout below `data_root` is listed in `paths.json`.
3. Run the scripts from this folder. Results are written to `out/`, logs to `logs/`.

The per-run outputs of the runs reported in the paper are already in `out/`, so tables and figures can be
regenerated without re-running the experiments (`python aggregate.py ...`, `python gen_tables.py`,
`python make_figs.py`, `python make_figs2.py ...`).

## Protocol in brief

- Preprocessing (type decisions, imputation, categorical vocabularies, frequency tables, scaling) is fitted
  on the training partition only (`common.Prep`).
- Graphs are inductive: training flows form the reference set; every other flow receives edges from its
  k nearest reference flows (exact GPU search, `common.knn`, `common.make_graph`).
- Hyperparameters of all models are selected on the validation partition of seed 0 (`exp_stationary.py tune`).
- Note: XGBoost runs on the CPU in the drift, full-corpus and runtime scripts. On our GPU (RTX 5050) the
  native XGBoost GPU interface returned a constant class; the scikit-learn interface (used for the
  stationary tables) agrees with the CPU model.

## Script-to-result map

| Paper item | Command | Output |
|---|---|---|
| Table 4 (hyperparameters) | `python exp_stationary.py tune`; `python exp_egraphsage.py tune` | `out/stationary/X-IIoTID/tuning*.json` |
| Table 6, Fig. 4 (10 seeds) | `python exp_stationary.py main 0,...,9`; `python exp_egraphsage.py main 0,...,9`; `python aggregate.py main` | `out/stationary/X-IIoTID/main_runs.jsonl`, `out/tables/main_X-IIoTID.csv` |
| Fig. 3 (training curves) | written by the main run of seed 0 | `out/stationary/X-IIoTID/agt_hist_seed0.json` |
| Table 7 (full corpus) | `python exp_fullcorpus.py 0,1,2` | `out/fullcorpus/runs.jsonl` |
| Table 8, Fig. 5 (four datasets) | `DS=<name> python exp_stationary.py main 0,1,2,3,4`; `python aggregate.py main:<name>` | `out/stationary/<name>/`, `out/tables/main_<name>.csv` |
| Table 9, Figs. 6--7 (per class) | `python aggregate.py main` | `out/tables/perclass_X-IIoTID.json` |
| Table 10, Fig. 8 (importance) | `python exp_stationary.py importance` | `out/stationary/X-IIoTID/importance.json` |
| Table 11 (ablation) | `python exp_stationary.py ablation 0,...,9`; `python aggregate.py ablation` | `out/tables/ablation.csv` |
| Table 12 (L and k) | `python exp_stationary.py sens 0,1,2,3,4`; `python aggregate.py sens` | `out/tables/sensitivity.csv` |
| Table 5 (drift protocol), Fig. 9 | `python regen_meta.py` | `out/drift/windows_meta.json` |
| Update-setting selection (pseudo-stream) | `./tune_drift.sh`; `python select_drift_cfg.py` | `out/drift/pseudo_*`, `out/drift/tuning_choice.json`, `drift_cfg.json` |
| Table 13, Fig. 10, Fig. 15 (ten-class drift) | `NO_CFG=1 python exp_drift.py multi 0,1,2,3,4`; `./final_drift.sh` (tuned AGT, MLP, XGBoost) | `out/drift/multi_rho0.2_M2500_eps0.0_*` |
| Table 14 (binary task) | `python exp_drift.py binary 0,1,2,3,4 0.2 2500 0.0 <methods>`; `./orig_setting.sh` | `out/drift/binary_*` |
| Table 15 (label budget, buffer) | `python exp_drift.py multi 0,1,2 <rho> <M> 0.0 <methods>` (see `run_queue2.sh`) | `out/drift/multi_rho*_M*` |
| Table 16, Fig. 11 (poisoning) | `python exp_drift.py multi 0,1,2 0.2 2500 <eps> <methods>`; reference: `TAGX=epsref python exp_drift.py multi 0,1,2 0.2 2500 0.0 MLP-rehearsal,AGT-rehearsal` | `out/drift/*eps0.[123]*`, `*epsref*`, `out/tables/poisoning.csv` |
| Table 17, Figs. 12--13 (noise) | `python exp_stationary.py noise 0,1,2,3,4`; `python aggregate.py noise` | `out/tables/noise*.csv` |
| Table 18, Fig. 14a (graph construction) | `python exp_hpc.py knn` | `out/hpc/knn_scaling.jsonl`, `thread_scaling.jsonl` |
| Fig. 14b (throughput), arrival rates | `python exp_hpc.py throughput`; `python exp_hpc.py arrival` | `out/hpc/throughput.jsonl`, `arrival.json` |
| Fig. 14c, Table 19 (cost, memory) | `python exp_hpc.py retrain`; `python exp_hpc.py memory` | `out/hpc/retrain_growth.jsonl`, `memory.jsonl` |
| All tables in LaTeX | `AGT_MANUSCRIPT=<folder> python gen_tables.py` | `<folder>/tables/*.tex`, `values.json` |
| All figures | `python make_figs.py`; `python make_figs2.py fig_hpc fig_perclass fig_confusion fig_importance fig_noise_perclass fig_poison` | `out/figures/` |
| Figure 1 | `python fig1_r1.py` | `Figure1_AGT_R1.pptx` |

`run_queue2.sh` runs the whole sequence (about 12 hours on one RTX 5050 laptop GPU).
