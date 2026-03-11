# NewRO25

EEG, fNIRS, and EEG-fNIRS fusion training scripts for movement classification based on trial-wise windowing and channel selection.

## Overview

This repository contains three main training pipelines plus EEGBCI public-dataset evaluation scripts:

- `eeg.py`: EEG classification
- `fnirs.py`: fNIRS classification
- `fusion.py`: EEG-fNIRS fusion classification
- `eegbci_temporal_se.py`: EEGBCI quick test with the in-house `temporal_se` model
- `eegbci_eegnet.py`: EEGBCI quick test with published EEGNet
- `eegbci_deep4.py`: EEGBCI quick test with published Deep4Net
- `eegbci_shallow.py`: EEGBCI quick test with published ShallowFBCSPNet

The current scripts share the same core workflow:

- resolve project and output paths with `runtime_utils.py`
- load `.fif` data from a modality-specific directory
- build trial-level segments first
- split data by trial into train/valid/test
- generate overlapping windows inside each split only
- rank channels with Fisher-score-based selection
- train a Braindecode classifier and save metrics, confusion matrices, logs, and summaries

This trial-first split avoids leakage from overlapping windows crossing dataset splits.

## Main Files

- `eeg.py`: EEG pipeline with `tdpsd` or `bandpower` Fisher channel ranking
- `fnirs.py`: fNIRS pipeline with trial reconstruction from annotation `11`
- `fusion.py`: fusion pipeline using existing `Rest -> Elbow_Flexion -> Elbow_Extension` annotation triplets as trials
- `eegbci_common.py`: shared EEGBCI download, preprocessing, split, training, and evaluation logic
- `eegbci_temporal_se.py`: standalone EEGBCI wrapper for the in-house temporal SE model
- `eegbci_eegnet.py`: standalone EEGBCI wrapper for EEGNet
- `eegbci_deep4.py`: standalone EEGBCI wrapper for Deep4Net
- `eegbci_shallow.py`: standalone EEGBCI wrapper for ShallowFBCSPNet
- `model_factory.py`: model builder for `shallow` and `temporal_se`
- `runtime_utils.py`: shared runtime path, config, and output-directory helpers
- `fe_u/eeg_bandpower.py`: EEG bandpower feature scoring
- `fe_u/eeg_tdpsd.py`: EEG TDPSD feature scoring
- `test.ipynb`: notebook for inspecting relabeled EEG files and trial views

## Data Layout

Expected default directories:

- `PPEEG`: EEG `.fif` files
- `PPfNIRS`: fNIRS `.fif` files
- `FusionEEG-fNIRS`: fusion `.fif` files

Typical filenames:

- EEG: `N004_A1L_eeg_raw.fif`
- fNIRS: `N004_A1L_fnirs_raw.fif`
- Fusion: `N004_A1L_eeg_fnirs_raw.fif`

## Configuration

Default training settings are read from `config.yaml`:

```yaml
top_k: 32
window_size_samples: 200
window_stride_samples: 100
batch_size: 32
n_epochs: 100
lr: 0.000625
weight_decay: 0.001
seed: 20250713
```

Command-line arguments override config values where supported.

## Common Runtime Arguments

All three main scripts support these shared arguments from `runtime_utils.py`:

- `--project_root`
- `--output_root`
- `--config_path`
- `--top_k_step`
- `--min_top_k`
- `--files_limit`

They also support:

- `--top_k`: starting channel count for the top-k search

Additional modality-specific arguments:

- `eeg.py`
  - `--device`
  - `--model`
  - `--fisher_method`
  - `--bandpower_mode`
- `fnirs.py`
  - `--model`
- `fusion.py`
  - `--model`
- `eegbci_*`
  - `--task_type`
  - `--subjects`
  - `--runs`
  - `--device`
  - `--model`
  - `--fisher_method`
  - `--bandpower_mode`
  - `--early_stopping_patience`
  - `--window_size_samples`
  - `--window_stride_samples`

## Usage

Run EEG training:

```bash
python eeg.py --data_dir PPEEG --model temporal_se --fisher_method tdpsd --top_k 63
```

Run EEG training with bandpower ranking:

```bash
python eeg.py --data_dir PPEEG --fisher_method bandpower --bandpower_mode avg --top_k 63
```

Run fNIRS training:

```bash
python fnirs.py --data_dir PPfNIRS --model shallow --top_k 88
```

Run fusion training:

```bash
python fusion.py --data_dir FusionEEG-fNIRS --model shallow --top_k 151
```

Run the EEGBCI public-dataset quick test with EEGNet:

```bash
python eegbci_eegnet.py
```

Run the EEGBCI Deep4Net test on a few subjects only:

```bash
python eegbci_deep4.py --subjects 1 2 3
```

Run the EEGBCI Temporal-SE test for comparison:

```bash
python eegbci_temporal_se.py --subjects 1 2 3
```

Run the EEGBCI ShallowFBCSP test without channel selection:

```bash
python eegbci_shallow.py --subjects 1 2 3 --disable_channel_selection
```

Limit the number of files for quick checks:

```bash
python eeg.py --files_limit 1
python fnirs.py --files_limit 1
python fusion.py --files_limit 1
```

## Outputs

Scripts write outputs under the project root by default, or under `--output_root` if provided.

Common outputs include:

- `Logs/`: training logs
- `ResEEG/`: EEG confusion matrices, selected channels, and summaries
- `ResfNIRS/`: fNIRS confusion matrices and summaries
- `ResFusion/`: fusion confusion matrices and summaries
- `ResEEGBCI/`: EEGBCI quick-test confusion matrices, channel selections, and summaries

Generated files include:

- confusion matrix images
- confusion matrix CSV files
- selected-channel CSV files
- summary CSV files across `top_k` values

## Trial Handling

The scripts now use trial-first splitting:

- `eeg.py`: trials are reconstructed from `Stimulus/S  5`
- `fnirs.py`: trials are reconstructed from annotation `11`
- `fusion.py`: each `Rest -> Elbow_Flexion -> Elbow_Extension` triplet is treated as one trial
- `eegbci_*`: each EEGBCI annotation segment (`T0/T1/T2`) is treated as one trial before split/windowing

This means:

- overlapping windows remain inside a single split
- train/valid/test no longer share windows from the same trial
- channel selection can still be run on the full retained dataset if intentionally desired

## Models

Models are defined in `model_factory.py`:

- `shallow`: Braindecode `ShallowFBCSPNet`
- `temporal_se`: lightweight temporal Conv1D network with squeeze-and-excitation
- `eegnet`: published EEGNet for compact EEG decoding
- `deep4`: published Deep4Net from Schirrmeister et al.

## Notes

- `top_k` is now configurable from either `config.yaml` or `--top_k`
- `top_k_step` must be a positive integer
- scripts exit early if no input files are found
- Matplotlib is configured for headless execution in `fnirs.py` and `fusion.py`

## Notebook

Use `test.ipynb` to inspect relabeled EEG data:

- rebuild trial annotations
- preview relabeled annotation tables
- save relabeled EEG FIF files
- inspect a selected trial with MNE plotting

## EEGBCI Quick Test

Use the EEGBCI scripts to validate model behavior on a public EEG dataset.

- defaults to the EEGBCI imagery task with runs `4/8/12`
- uses three classes: `Rest`, `Left_Fist`, `Right_Fist`
- checks whether EEGBCI files already exist under `data/eegbci`
- downloads missing files automatically to `data/eegbci`
- supports separate wrappers for `temporal_se`, `eegnet`, `deep4`, and `shallow`
- supports `--disable_channel_selection` for a more standard public-dataset baseline
- applies EEGBCI preprocessing with `60 Hz` notch, `1-40 Hz` band-pass, and average EEG reference by default
- reports class counts, majority-class baselines, accuracy, balanced accuracy, and macro F1
- saves logs and result files under `Logs/` and `ResEEGBCI/`

## Requirements

The checked-in `requirements.txt` is incomplete for the full training stack. In practice, the project also expects packages such as:

- `mne`
- `torch`
- `braindecode`
- `skorch`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `pandas`
- `numpy`

Install the missing runtime dependencies in your environment before running the training scripts.
