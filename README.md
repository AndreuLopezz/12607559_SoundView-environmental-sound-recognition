# SoundView

SoundView is an Applied Deep Learning project about recognizing environmental sounds and showing visual alerts for selected categories. The planned experiment compares a small CNN trained on audio spectrograms with a pretrained audio model. The [Assignment 1 proposal](<Assignment 1.md>) describes the motivation, methods, and intended web demo in detail.

## Dataset

The project uses [ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 labelled five-second recordings across 50 sound classes and five predefined evaluation folds. Download it from the official repository into a `datasets` folder **beside** this project, so the audio files stay outside the project Git repository.

From the SoundView repository root, run these commands in PowerShell:

```powershell
New-Item -ItemType Directory -Force ..\datasets | Out-Null
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git ..\datasets\ESC-50
```

On Linux or macOS, use:

```bash
mkdir -p ../datasets
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git ../datasets/ESC-50
```

The expected metadata file is `../datasets/ESC-50/meta/esc50.csv`, and recordings are in `../datasets/ESC-50/audio/`. If ESC-50 is already present there, no second download is needed. The official folds will be kept separate during training and evaluation.

## Data preparation

Use Python 3.11 in a Conda environment for the project. From the repository root:

```powershell
conda create -n soundview python=3.11 -y
conda activate soundview
python -m pip install -r requirements.txt
python -m soundview.data --data-dir ..\datasets\ESC-50
python -m pytest -q
```

If `soundview` already exists, run `conda install -n soundview python=3.11 pip -y` in place of `conda create` before activating it. The dataset check reports clip, class, and fold counts and fails if metadata refers to missing audio. `soundview.data` also loads ESC-50's mono 16-bit PCM WAV files at their original 44.1 kHz rate and produces 64-band log-Mel spectrograms for the planned CNN.

## CNN baseline in Google Colab

Open [the CNN training notebook](notebooks/train_cnn_colab.ipynb) with a Colab GPU runtime after this code has been pushed. You can use the Colab website or connect to that runtime from VS Code. Edit the settings cell if needed, then run the cells in order. The notebook downloads this project and ESC-50 into separate folders and installs dependencies.

The default run uses folds 2, 3, and 4 for training and fold 1 for validation. Fold 5 remains reserved for later evaluation. The run writes `best_cnn.pt` and `training_summary.json` to `/content/runs/cnn_fold5`; the summary includes validation metrics, chosen settings, and elapsed time. Download or copy both files before the Colab runtime disconnects, because `/content` is temporary. If your VS Code connection cannot retrieve remote files, set `OUTPUT_DIR` to a mounted Drive folder instead.

The target set before training is a macro-F1 of at least 0.50 across the five official test folds. This first run reports validation results only; the achieved test value will be added after the five-fold evaluation.

## CNN test evaluation

After choosing a checkpoint from validation results, open [the evaluation notebook](notebooks/evaluate_cnn_colab.ipynb) in a Colab GPU runtime. Its default `RUN_DIR` points to the 40-epoch experiment in Google Drive; change that path if your checkpoint is elsewhere. Run the cells in order. The notebook scores only the reserved fold 5 and saves `test_evaluation.json` beside the checkpoint, with overall metrics, per-class scores, a confusion matrix, clip predictions, and model inference time. Keep this test result separate from validation when deciding on later experiments. The five-fold evaluation is still needed for the planned final comparison.

## Five-fold CNN evaluation

Once the fold-5 report is saved, open [the five-fold notebook](notebooks/fivefold_cnn_colab.ipynb) in a Colab GPU runtime. It keeps the selected training settings fixed, trains and tests folds 1–4, and reuses the existing fold-5 report. Check the two Drive paths in its settings cell, then run the cells in order. Results are saved to Drive under `SoundView/cnn_fivefold/`. If the runtime disconnects, completed folds can be reused when the same settings are retained. The final `fivefold_summary.json` reports the mean of the five test-fold macro-F1 scores and the pooled score across all 2,000 clips.

## Pretrained PANNs in Google Colab

Open [the PANNs notebook](notebooks/panns_colab.ipynb) in a Colab GPU runtime. It installs the additional packages from `requirements-panns.txt`, loads the pretrained CNN14 checkpoint from Google Drive, and checks its output on a generated tone. It then extracts one frozen 2,048-value representation per ESC-50 clip, trains a new 50-class head with the same five-fold protocol as the CNN, and saves fold reports plus `fivefold_summary.json` under `SoundView/panns_frozen_v1/` in Drive. Completed folds are reused when their saved settings match. PANNs expects 32 kHz audio, so its input uses model-specific resampling while the CNN continues to use ESC-50's original 44.1 kHz recordings. Keep checkpoints and feature caches outside Git. Partial fine-tuning can be added to this notebook after the frozen-head result is reviewed.

## Project status

The repository contains the Assignment 1 proposal, ESC-50 data preparation, and CNN baseline training and evaluation code. The CNN five-fold experiment is complete, and the frozen PANNs comparison is ready to run in Colab. Partial fine-tuning and the visual alert demo are planned for later stages.
