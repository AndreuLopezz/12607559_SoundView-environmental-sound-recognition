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

If `soundview` already exists, run `conda install -n soundview python=3.11 pip -y` in place of `conda create` before activating it. The dataset check reports clip, class, and fold counts and fails if metadata refers to missing audio. `soundview.data` also loads ESC-50's mono 16-bit PCM WAV files at their original 44.1 kHz rate and produces 64-band log-Mel spectrograms for the planned CNN. Model training dependencies and commands will be added with the training code.

## Project status

The repository contains the Assignment 1 proposal and the ESC-50 data preparation code. Training, evaluation, and demo commands will be documented here as they are added for Assignment 2.
