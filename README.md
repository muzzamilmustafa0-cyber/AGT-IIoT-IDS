# AGT — Attention-Augmented Graph Transformer for Drift-Resilient IIoT Intrusion Detection

**Author:** Muzzamil Mustafa
**Contact:** muzzamil.mustafa@graduate.univaq.it

Reproducible code, results, logs and figures for the paper *"An Attention-Augmented
Graph Transformer with Self-Evolving Adaptation for Drift-Resilient Intrusion
Detection in Cyber-Physical IoT Networks."* All results are on **real** public
datasets and are fully reproducible from this repository.

## Method
Traffic instances are nodes of a k-nearest-neighbor similarity graph. Categorical
attributes use learned **entity embeddings**; a residual encoder couples **local
graph attention** (graph-transformer convolution) with a **linear-complexity
landmark global-attention** module. A **self-evolving** module performs
rehearsal-based incremental updates to track concept drift. Training uses
class-weighted cross-entropy for the heavy class imbalance.

## Datasets (4, all real/public)
| Dataset | Domain | Classes |
|---|---|---|
| X-IIoTID | Industrial IoT, multi-stage | 10 |
| Edge-IIoTset | Industrial IoT (Modbus/MQTT) | 15 |
| WUSTL-IIoT-2021 | SCADA testbed | 5 |
| CICIoT2023 | IoT (recent, large-scale) | 8 (super-categories) |

## Repository layout
```
code/      experiment scripts (table below)
results/   xiiotid/ + multi_dataset/ + self_evolving/ + runtime/ + logs/  (tables, json, logs)
figures/   all 16 paper figures
data/      dataset download scripts + a small X-IIoTID sample
```

### Scripts (`code/`)
| Script | Purpose |
|---|---|
| `agt_main.py` | Main X-IIoTID pipeline: AGT vs baselines, per-class, confusion, training curves |
| `run_dataset.py` | Generalized runner for any dataset (used for the 4-dataset study) |
| `ablation.py` | Ablation of AGT components |
| `landmark_sensitivity.py` | Sensitivity to the number of landmark tokens L |
| `runtime_analysis.py` | Graph-build scalability, latency, memory, adaptation-cost (self-evolve vs retrain) |
| `feature_importance.py` | AGT's own permutation feature importance |
| `self_evolving_drift.py` | Prequential self-evolving AGT vs static/retrained/online baselines under real temporal drift |
| `noise_robustness.py` | Robustness to feature-noise drift |
| `temporal_drift.py` | Train-past / test-future drift characterization |
| `perclass_drift.py` | Per-class F1 under drift |
| `hybrid_baseline.py` | Graph-refined gradient-boosting hybrid (analysis) |
| `scale_check.py` | Larger-scale / full-data consistency check |
| `download_data.py`, `download_wustl.py`, `download_edge_iiotset.py`, `prepare_ciciot2023.py` | Dataset acquisition |

## Reproduce
```bash
pip install -r requirements.txt
python code/download_data.py            # X-IIoTID (Kaggle); see data/README.md
python code/agt_main.py                 # main tables/figures
python code/ablation.py
python code/self_evolving_drift.py      # the drift study (multi-seed)
python code/run_dataset.py --csv <path> --label <col> --name <DATASET>   # other datasets
```
GPU recommended (results produced on an RTX 5050, CUDA 12.8). Default stratified
80,000-record subsample (`--sample` / `AGT_SAMPLE`).

## Key results
**Cross-dataset (weighted-F1 / macro-F1):** AGT beats the GCN on every dataset
(macro gaps +0.21/+0.12/+0.20/+0.07) and is competitive on multi-stage IIoT;
gradient-boosted trees lead on stationary accuracy (reported transparently).

**Real temporal concept drift (prequential, 4 seeds, mean F1):**
XGBoost-static 0.70 (collapses) · online-MLP 0.80 · AGT-static 0.84 ·
**AGT self-evolving 0.89** · XGBoost-retrained 0.91. The self-evolving AGT
matches a fully retrained model at **59x lower adaptation cost** (0.19 s vs 11.4 s
per update).

## Honest scope
AGT does not beat gradient-boosted trees on stationary accuracy (a known property
of tabular data) and is weaker on volumetric-flood IoT traffic (CICIoT2023); its
contribution is **drift-robust, cost-efficient adaptive detection** for
coordinated multi-stage intrusions. All numbers, logs and figures are included.

---
(c) 2026 Muzzamil Mustafa. All rights reserved. All code, results and figures herein are the work of the author.
