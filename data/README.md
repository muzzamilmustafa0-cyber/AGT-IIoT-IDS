# Datasets

All four datasets are public. Raw CSVs are **not committed** (they exceed
GitHub's 100 MB limit); download them with the provided scripts. A small
`X-IIoTID_sample.csv` (~12k rows) is included so the pipeline can be run
immediately without any download.

| Dataset | Source | Download |
|---|---|---|
| X-IIoTID | Kaggle `munaalhawawreh/xiiotid-iiot-intrusion-dataset` | `python code/download_data.py` |
| Edge-IIoTset | Kaggle `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot` | `python code/download_edge_iiotset.py` |
| WUSTL-IIoT-2021 | Kaggle `annaamalaiu/wustl-iiot-2021-dataset` | `python code/download_wustl.py` |
| CICIoT2023 | Kaggle (merged CSV) | `python code/prepare_ciciot2023.py` |

A Kaggle API token at `~/.kaggle/kaggle.json` is required (see
https://www.kaggle.com/docs/api). Each runner reads its dataset via `--csv`.

## Citations
- X-IIoTID: Al-Hawawreh, Sitnikova, Aboutorab, *IEEE IoT Journal*, 2022.
- Edge-IIoTset: Ferrag et al., *IEEE Access*, 2022.
- WUSTL-IIoT-2021: Zolanvari et al., *IEEE IoT Journal*, 2021.
- CICIoT2023: Neto et al., *Sensors*, 2023.
