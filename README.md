# FTQNet: Fractional-Order Surrogate Gradient Learning for Biomedical Time Series Forecasting

**FTQNet** is a research implementation of the *Fractional Temporal Query Network* for multivariate time-series forecasting. It combines Temporal Query (TQ) aggregation with **Fractional Query Projection (FQ Projection)** and **Group-Representative Enhanced Fractional Learning (GREFL)**.

> **Paper:** *FTQNet: fractional-order surrogate gradient learning for biomedical time series forecasting* (manuscript supplied with this code release).  
> **Scope:** Biomedical forecasting on infectious-disease, healthcare, wearable, and longitudinal microbiome datasets, plus 12 general-purpose forecasting benchmarks.  
> **Important:** This repository contains research code, dataset-specific processing, experiment scripts, and saved experiment artifacts. See [Reproducibility notes](#reproducibility-notes) before comparing with reported paper results.

## 1. Overview

FTQNet retains the Temporal Query mechanism for cross-variable dependency learning while replacing four task-related affine projections with FQ Projections:

1. **FQ Input Projection** — maps the historical sequence into hidden features.
2. **FQ MLP-1** and **FQ MLP-2** — process hidden features with GELU activations and a residual path.
3. **FQ Forecast Projection** — maps hidden features to the prediction horizon.

The forward computation remains a standard affine projection. During training, GREFL replaces its **weight gradient** with a fractional-order surrogate response based on:

- Output-channel grouping and group-representative parameter sharing.
- An input-driven and a remainder-driven fractional response.
- Sign–magnitude decomposition for real-valued weights.
- Group-wise RMS magnitude control on the remainder branch.

Input and bias gradients follow their ordinary affine rules. At the integer-order endpoint (`alpha = 1`, within the implemented tolerance), the weight gradient recovers the standard first-order gradient.

**Core implementation:** [`models/FTQNet.py`](models/FTQNet.py) and [`layers/FLinear.py`](layers/FLinear.py).

## 2. Repository structure

```text
FTQNet/
├── run.py                       # Main train/test entry point
├── run_epicast_fixed.sh         # Runs all 12 biomedical dataset scripts
├── run_main.sh                  # Runs 12 general-purpose benchmark scripts
├── environment-epicast.yml     # Pinned CUDA-oriented Conda environment
├── requirements-epicast.txt    # Pinned dependencies for that environment
├── requirements.txt            # Minimal dependency list (see note below)
├── models/
│   └── FTQNet.py               # Temporal Query and forecasting backbone
├── layers/
│   └── FLinear.py              # Fractional-order surrogate-gradient projection
├── data_provider/
│   ├── data_factory.py
│   ├── data_epicast.py         # Five weekly infectious-disease datasets
│   ├── data_health.py          # ILI and COVID-19
│   ├── data_wearable.py        # Wearable, MixBench-style protocol
│   ├── data_microbiome.py      # Four longitudinal microbiome datasets
│   └── data_loader.py          # General-purpose benchmarks
├── exp/                         # Experiment runners and evaluation
├── scripts/
│   ├── FTQNet/                # Dataset-specific shell scripts
│   └── wearable/
│       └── prepare_wearable.py
├── dataset/                     # Extract dataset.zip here (see Section 4)
├── checkpoints/                 # Saved model checkpoints
├── results/                     # Dataset-specific result summaries
├── test_results/                # Forecast visualizations (PDF)
└── LICENSE
```

**Note:** The data files are **not** stored in this Git repository. All datasets used in the paper (12 biomedical + 12 general-purpose benchmarks) are packaged in a single archive hosted on Google Drive — see [Section 4](#4-datasets-and-download-links) for the download link and extraction instructions. Code, checkpoints, results, and test PDFs are included in the repository.

## 3. Installation

Run commands from the project root. Linux, WSL, or a shell with Bash support is recommended.

### Option A — supplied pinned environment (CUDA 12.1)

```bash
conda env create -f environment-epicast.yml
conda activate ftqnet-epicast
```

This environment specification targets Python 3.8.20, PyTorch 2.4.1 with CUDA 12.1, NumPy 1.24.4, and pandas 2.0.3. GPU / CUDA compatibility depends on your system; this README does **not** claim that the environment has been tested on every device.

### Option B — set up dependencies manually

```bash
conda create -n ftqnet python=3.10 -y
conda activate ftqnet
# Install the appropriate PyTorch build for your own CPU/GPU first.
pip install -r requirements.txt
pip install scipy
```

The minimal `requirements.txt` does not list `scipy`, but `layers/FLinear.py` imports it. For exact environment replication, prefer Option A.

On a machine without a usable CUDA device, `run.py` automatically disables GPU use. For a CPU-only run you can also use:

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/FTQNet/chickenpox_hungary.sh 4
```

## 4. Datasets and download links

### 4.1 Dataset archive (download required)

The datasets are **not** committed to this repository because of their size. Everything needed to run the scripts is packaged in one archive:

> **`dataset.zip` — 180 MB — [⬇ Download from Google Drive](https://drive.google.com/file/d/1JtuuvK7nDRrFsfCbkwrmI4ZeMCeu1LDa/view?usp=sharing)**

- **Access:** the link is public — no Google account, no GitHub account, and no access request is required. Only the URL is needed.
- **Contents:** all 24 data files used in the paper — the 12 biomedical datasets (Section 4.2) and the 12 general-purpose benchmarks (Section 4.3).
- **How to use:** download the archive, then extract it **directly into `dataset/`** so that the files sit flush with the directory, not inside an extra nested folder:

```bash
# from the project root
unzip dataset.zip -d dataset/
ls dataset/            # expect: ETTh1.csv, ILI.csv, Covid-19.csv, PEMS03.npz, ...
```

The scripts read `dataset/<filename>` with no nested `ETT-small/`, `traffic/`, or `dataset/dataset/` paths, so a wrong extraction level is the most common cause of a "file not found" error.

- **Integrity:** if you need to confirm a complete download, the archive contains 24 entries; `Covid-19.csv` (≈51 MB), `traffic.csv` (≈130 MB), and `solar_AL.txt` (≈172 MB uncompressed) are the large ones.
- **Licensing:** third-party datasets keep their own licenses and attribution requirements (see Section 8, item 6). The archive is provided for research reproducibility only.

### 4.2 Biomedical datasets (included in `dataset.zip`)

**All 12 CSV files below are contained in `dataset.zip` and land directly under `dataset/` after extraction.** You do not need to download them individually to try the provided scripts. The sources below are for provenance, independent acquisition, and proper dataset citation.

**Infectious-disease datasets (EpiCastBench-related tasks)**

| `--data` | Required file under `dataset/` | Code-side series shape | Original data / download location |
|---|---|---:|---|
| `Chickenpox-Hungary` | `hungary_chickenpox.csv` | 522 weeks × 20 variables | [UCI Hungarian Chickenpox Cases](https://archive.ics.uci.edu/dataset/580/hungarian%2Bchickenpox%2Bcases) |
| `Dengue-Colombia` | `Dengue Colombia.csv` | 835 weeks × 33 variables | [OpenDengue Data Explorer](https://opendengue.org/data) |
| `Dengue-Argentina` | `Dengue Argentina.csv` | 92 weeks × 23 variables | [OpenDengue Data Explorer](https://opendengue.org/data) |
| `Influenza-US` | `Influenza US.csv` | 521 weeks × 53 variables | [CDC FluView / ILINet](https://www.cdc.gov/fluview/overview/fluview-interactive.html) |
| `Influenza-US-2` | `Influenza US 2.csv` | 1200 weeks × 10 HHS regions | [CDC FluView / ILINet](https://www.cdc.gov/fluview/overview/fluview-interactive.html) |

- **OpenDengue:** Obtain country-level records for Colombia/Argentina; the code expects the original record schema, and `data_epicast.py` selects weekly Admin2 records and aggregates them to Admin1.
- **CDC:** Export **`ILITOTAL` (visit counts)**, not ILI percentage. `Influenza-US` uses selected state/jurisdiction series; `Influenza-US-2` uses the 10 HHS region series. They are different datasets. Preserve the source table columns and filenames.
- **Chickenpox:** Use the weekly `hungary_chickenpox.csv` time-series file, not the county-adjacency file.
- **Do not substitute** a current updated export without checking dates, regions, reporting conventions, and preprocessing: the resulting matrix may differ from the packaged one.

**Other healthcare / wearable datasets**

| `--data` | Required file | Data size / model input | Download or original-source guidance |
|---|---|---|---|
| `ILI` | `ILI.csv` | 966 timestamps × 7 channels | [THUML illness benchmark CSV](https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/illness/national_illness.csv); original influenza surveillance: [CDC FluView](https://www.cdc.gov/fluview/overview/fluview-interactive.html) |
| `Covid-19` | `Covid-19.csv` | 1392 timestamps × 948 channels | **Use the packaged CSV for exact experiments.** For underlying public COVID-19 records, consult [Johns Hopkins CSSE COVID-19](https://github.com/CSSEGISandData/COVID-19) or [Google COVID-19 Open Data](https://github.com/GoogleCloudPlatform/covid-19-open-data); these are *not verified drop-in replacements* for this 948-channel benchmark. |
| `Wearable` | `wearable.csv` | 38,913 raw records; loader selects 19 modeling channels | [Original wearable-HRV dataset on Springer Nature / Figshare](https://springernature.figshare.com/articles/dataset/In-situ_wearable-based_dataset_of_continuous_heart_rate_variability_monitoring_accompanied_by_sleep_diaries/28509740) |

**Important format notes:**

- Packaged `ILI.csv` and `Covid-19.csv` use a **long CSV format** with columns `date,data,cols`. The public `national_illness.csv` is a **wide** table; `data_health.py` can read either its recognized wide layout or the supplied long format, but downloaded files must be renamed/converted and validated before use.
- For COVID-19, the official upstream projects linked above do **not** establish the exact processing and 948-channel selection used in this package.
- The provided wearable raw file contains `deviceId`, `ts_start`, and sensor fields. Its loader performs subject-aware processing, drops designated columns, and uses **19** model channels. The paper's dataset-characteristics table lists **24** variables; these counts refer to different representations and should not be treated as automatically identical.
- For wearable provenance, obtain the raw, 5-minute `sensor_hrv.csv` from the linked dataset release and validate its schema before using it. `scripts/wearable/prepare_wearable.py` can export the loader-selected channels if needed; the current experiment scripts already read `dataset/wearable.csv`.

**Longitudinal human microbiome datasets**

| `--data` | Required file | Samples × microbial variables | Original data / download location |
|---|---|---:|---|
| `A-stool` | `A_Stool_abun.csv` | 341 × 105 | [MicroSTNet author repository — data directory](https://github.com/pwner-web/MicroBioSTNet) |
| `F4-stool` | `F4_gut_abun.csv` | 130 × 225 | [MicroSTNet author repository — data directory](https://github.com/pwner-web/MicroBioSTNet) |
| `M3-stool` | `M3_gut_abun.csv` | 332 × 308 | [MicroSTNet author repository — data directory](https://github.com/pwner-web/MicroBioSTNet) |
| `A-saliva` | `A_Saliva_abun.csv` | 285 × 298 | [MicroSTNet author repository — data directory](https://github.com/pwner-web/MicroBioSTNet) |

The microbiome source studies are [David et al. (2014)](https://doi.org/10.1186/s13059-014-0089-3) for subject A and [Caporaso et al. (2011)](https://doi.org/10.1186/gb-2011-12-5-r50) for subjects F4/M3. For reproducing this code, use the exact CSVs in the supplied package or locate the corresponding abundance tables in the MicroSTNet repository's `data/` directory.

The microbiome CSVs are **taxa × samples** matrices: the first column identifies taxa and subsequent columns are successive samples. Do **not** transpose or normalize the CSVs on disk; the adapter handles this in memory.

### 4.3 General-purpose forecasting datasets (included in `dataset.zip`)

The code also includes scripts for **12 standard datasets / 48 forecasting tasks**. Their data files are included in `dataset.zip` as well (same `dataset/` extraction target), so no separate download is required for a full reproduction.

**Provenance / independent acquisition:**

- [TQNet official repository — dataset download instructions](https://github.com/ACAT-SCUT/TQNet) and its [public Google Drive dataset archive (`all_dataset.zip`)](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view).
- [THUML Time-Series-Library dataset collection](https://huggingface.co/datasets/thuml/Time-Series-Library) for ETT, Electricity, Traffic, Weather, and ILI.
- [Original multivariate time-series data repository](https://github.com/laiguokun/multivariate-time-series-data) for sources including Solar-Energy, Electricity, and Traffic.
- [ASTGCN traffic dataset repository](https://github.com/Davidham3/ASTGCN-2019-mxnet/tree/master/data) for PeMS source material (check that the downloaded files match the expected benchmark `.npz` format).

Place the following files **directly into `dataset/`** (the scripts do not use nested `ETT-small/`, `traffic/`, etc. paths):

| Benchmark group | Required filenames | Forecast horizons |
|---|---|---|
| ETT (4) | `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` | 96, 192, 336, 720 |
| Electricity, Solar, Traffic, Weather (4) | `electricity.csv`, `solar_AL.txt`, `traffic.csv`, `weather.csv` | 96, 192, 336, 720 |
| PEMS (4) | `PEMS03.npz`, `PEMS04.npz`, `PEMS07.npz`, `PEMS08.npz` | 12, 24, 48, 96 |

For example, after extraction the expected paths include `dataset/ETTh1.csv` and `dataset/PEMS03.npz`. Some upstream archives have different folders or filenames; ensure the actual contents match the selected benchmark before relocating files.

## 5. Quick start — train and evaluate

### One infectious-disease experiment

```bash
# Chickenpox-Hungary, 24-week input → 4-week forecast
bash scripts/FTQNet/chickenpox_hungary.sh 4

# Other weekly surveillance datasets (choose horizon(s) 4/8/12)
bash scripts/FTQNet/dengue_argentina.sh 4 8 12
bash scripts/FTQNet/dengue_colombia.sh 4
bash scripts/FTQNet/influenza_us.sh 8
bash scripts/FTQNet/influenza_us_2.sh 12
```

### Healthcare, wearable, and microbiome experiments

```bash
# ILI and COVID-19: H=24/36/48/60
bash scripts/FTQNet/ili.sh 24
bash scripts/FTQNet/covid-19.sh 24

# Wearable: H=12/48/144/288
bash scripts/FTQNet/wearable.sh 12

# Microbiome: 12 input samples → 1 or 8 output samples
bash scripts/FTQNet/a_stool.sh 1
bash scripts/FTQNet/f4_stool.sh 8
bash scripts/FTQNet/m3_stool.sh 1
bash scripts/FTQNet/a_saliva.sh 8
```

Omitting horizon arguments from these scripts normally runs all supported horizons for that dataset. Each horizon is trained separately with the hyperparameters specified in its shell script.

### Run the complete experiment sets

```bash
# All 12 biomedical datasets; can take substantial time
bash run_epicast_fixed.sh

# All 12 general-purpose benchmarks (download their datasets first)
bash run_main.sh

# A single general-purpose benchmark
bash scripts/FTQNet/etth1.sh
```

### Evaluate an existing checkpoint

`run.py` supports `--is_training 0` for test-only execution. Start from the **full command** in the matching shell script, change `--is_training 1` to `--is_training 0`, and keep the dataset, model identifier, forecasting horizon, fractional settings, and other experiment arguments unchanged. The model must find a compatible saved checkpoint.

For the special COVID-19 path, retain `--covid_calibrated`; its test-only execution also reconstructs validation-based calibration.

## 6. Main configuration arguments

| Argument | Meaning |
|---|---|
| `--seq_len`, `--pred_len` | Historical look-back length and prediction horizon |
| `--enc_in` | Number of input variables; microbiome adapters infer the original taxa count when set to `0` |
| `--cycle` | Period of the learnable Temporal Query representation |
| `--d_model` | Hidden feature dimension |
| `--use_revin` | Whether instance normalization is enabled |
| `--flinear_alpha` | Fractional order (`0 < alpha <= 1`) |
| `--flinear_c` | Positive group-wise RMS threshold for the remainder response |
| `--flinear_groups` | Number of output groups |
| `--loss` | Training objective (`mse` or `mae`) |
| `--is_training` | `1` = train and test; `0` = test only |

The actual model uses a four-head `MultiheadAttention` layer. For reproducibility, the dataset scripts are the source of truth for hyperparameter values; do not assume one universal configuration across all experiments.

## 7. Outputs

Experiment runs use the existing project locations:

```text
checkpoints/<experiment-setting>/checkpoint.pth
results/<dataset>.txt
result.txt
test_results/<experiment-setting>/*.pdf
```

Depending on the task, reported metrics include MSE, MAE, MASE and other dataset-specific measures. Results present in this repository are existing experiment artifacts, **not** newly reproduced or independently verified by this README.

## 8. Reproducibility notes

Please read these limitations before interpreting results:

1. **EpiCastBench dataset mismatch:** The supplied code reconstructs `Dengue-Colombia` as **835 × 33**; Table 1 in the supplied manuscript lists **101 × 28**. These should be considered different preprocessing / dataset definitions until checked against the exact benchmark release. Exact paper-number replication is therefore not assured.
2. **Fixed-origin infectious-disease testing:** The EpiCast adapters train on sliding windows and predict one final block of length `H`, with task-specific preprocessing and validation logic. They are not necessarily interchangeable with rolling-origin evaluations in external benchmarks.
3. **Microbiome evaluation is not an independent held-out test:** The adapter follows the [public MicroSTNet preprocessing/training code](https://github.com/pwner-web/MicroBioSTNet) with a global Z-score (using the full dataset), a sequential 70/30 partition, and partial reuse of validation windows in training. Its `test` mode evaluates the **same validation interval** used to choose checkpoints; results must **not** be described as unbiased held-out-test scores.
4. **Wearable protocol:** The loader uses a subject-based split and the source project's preprocessing choices, including fitting the scaler across all subjects. Interpret generalization results in light of this protocol.
5. **COVID-19 calibration:** COVID-19 scripts use `--covid_calibrated` and fit calibration using validation data before testing; it is part of the experiment protocol.
6. **Source files and data licenses:** The code carries an Apache-2.0 `LICENSE`. Third-party datasets have their **own** licenses, attribution requirements, and use conditions. Consult each original source before redistributing datasets.
7. **Citation details:** The manuscript copy supplied with this archive still uses placeholder author/contact/repository fields. Add the finalized author list, publication DOI, and repository URL here when available; do not cite placeholder bibliographic information.

## 9. Acknowledgments and references

FTQNet builds on the Temporal Query idea and draws on publicly released methods and datasets:

- [Temporal Query Network (TQNet), official implementation](https://github.com/ACAT-SCUT/TQNet).
- [MicroSTNet, author implementation and data](https://github.com/pwner-web/MicroBioSTNet).
- [EpiCastBench paper](https://arxiv.org/abs/2605.11598) for the infectious-disease forecasting benchmark.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/), [CDC FluView](https://www.cdc.gov/fluview/overview/fluview-interactive.html), and [OpenDengue](https://opendengue.org/data).
- [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [TQNet dataset instructions](https://github.com/ACAT-SCUT/TQNet) for general-purpose forecasting data.
- [Wearable HRV and sleep diary dataset](https://doi.org/10.6084/m9.figshare.28509740).

If you use this project, please cite the finalized FTQNet paper, the original datasets, and any reference implementations whose protocols you reproduce.

## 10. License

The repository includes an [Apache License, Version 2.0](LICENSE) file. That repository-level license does not override any separate license or restrictions on third-party data.
