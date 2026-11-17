"""ESC-50 metadata, fold splits, and audio preprocessing."""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


# Keep the label and fold beside each filename so they cannot get out of sync
# when records are split for training and evaluation.
@dataclass(frozen=True)
class Esc50Record:
    filename: str
    fold: int
    target: int
    category: str


# The dataset check returns these counts so the command can show what was found.
@dataclass(frozen=True)
class DatasetAudit:
    clips: int
    categories: int
    fold_counts: dict[int, int]


# ESC-50's CSV defines both the class labels and the official folds. Reject bad
# entries here so a later training run cannot silently use the wrong labels.
def read_metadata(dataset_dir: Path) -> list[Esc50Record]:
    """Read the labels and official folds from ESC-50's metadata file."""
    metadata_path = Path(dataset_dir) / "meta" / "esc50.csv"
    records = []
    filenames = set()

    with metadata_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"filename", "fold", "target", "category"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {metadata_path}")

        for row in reader:
            filename = row["filename"]
            fold = int(row["fold"])
            target = int(row["target"])
            if fold not in range(1, 6):
                raise ValueError(f"Invalid ESC-50 fold: {fold}")
            if target not in range(50):
                raise ValueError(f"Invalid ESC-50 target: {target}")
            if not filename or Path(filename).name != filename or filename in filenames:
                raise ValueError(f"Invalid or duplicate audio filename: {filename}")
            if not row["category"]:
                raise ValueError(f"Missing category for {filename}")

            filenames.add(filename)
            records.append(Esc50Record(filename, fold, target, row["category"]))

    return records


# Across five runs, each fold is held out for testing once. The following fold
# provides validation data, leaving the other three for training.
def split_folds(
    records: list[Esc50Record], test_fold: int
) -> tuple[list[Esc50Record], list[Esc50Record], list[Esc50Record]]:
    """Use one official fold for test and the next for validation."""
    if test_fold not in range(1, 6):
        raise ValueError(f"Invalid test fold: {test_fold}")

    validation_fold = test_fold % 5 + 1
    train = [
        record for record in records if record.fold not in {test_fold, validation_fold}
    ]
    validation = [record for record in records if record.fold == validation_fold]
    test = [record for record in records if record.fold == test_fold]
    return train, validation, test


# Check the downloaded audio against the CSV before training. This catches a
# partial download or an inconsistent label mapping early.
def audit_dataset(dataset_dir: Path) -> DatasetAudit:
    """Check that metadata, labels, folds, and audio filenames agree."""
    dataset_dir = Path(dataset_dir)
    records = read_metadata(dataset_dir)
    if not records:
        raise ValueError("ESC-50 metadata is empty")

    audio_dir = dataset_dir / "audio"
    filenames = {record.filename for record in records}
    for filename in filenames:
        if not (audio_dir / filename).is_file():
            raise FileNotFoundError(
                f"Audio file listed in metadata is missing: {filename}"
            )

    extra_files = {path.name for path in audio_dir.glob("*.wav")} - filenames
    if extra_files:
        raise ValueError(
            f"Audio files missing from metadata: {sorted(extra_files)[:3]}"
        )

    categories_by_target = {}
    targets_by_category = {}
    for record in records:
        categories_by_target.setdefault(record.target, set()).add(record.category)
        targets_by_category.setdefault(record.category, set()).add(record.target)
    if any(len(categories) != 1 for categories in categories_by_target.values()) or any(
        len(targets) != 1 for targets in targets_by_category.values()
    ):
        raise ValueError("ESC-50 categories and numeric targets are inconsistent")

    return DatasetAudit(
        clips=len(records),
        categories=len(targets_by_category),
        fold_counts=dict(sorted(Counter(record.fold for record in records).items())),
    )


# ESC-50 contains mono 16-bit PCM WAV files at 44.1 kHz. Keep that sample rate
# and put the stored integer samples on a float scale for the CNN.
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an ESC-50 WAV without changing its sample rate."""
    sample_rate, samples = wavfile.read(path)
    audio = samples.astype(np.float32) / 32768.0
    return audio, sample_rate


# Turn the waveform into the time-frequency representation used by the CNN.
# At 44.1 kHz, each window spans about 46 ms and advances about 23 ms.
# The small floor before taking the log keeps silent sections finite.
def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 1024,
) -> np.ndarray:
    """Convert a waveform to a log-Mel power spectrogram for the CNN."""
    if audio.ndim != 1 or len(audio) < n_fft:
        raise ValueError(
            "Audio must be one-dimensional and at least one FFT window long"
        )
    if sample_rate <= 0 or n_mels <= 0 or n_fft <= 0 or not 0 < hop_length <= n_fft:
        raise ValueError("Invalid spectrogram settings")

    frequencies, _, spectrum = stft(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )
    power = np.abs(spectrum) ** 2

    # Spacing the filters on the Mel scale gives the CNN more detail at lower frequencies.
    max_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_edges = np.linspace(0, max_mel, n_mels + 2)
    hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
    filters = np.empty((n_mels, len(frequencies)), dtype=np.float32)
    for band in range(n_mels):
        left, center, right = hz_edges[band : band + 3]
        rising = (frequencies - left) / (center - left)
        falling = (right - frequencies) / (right - center)
        filters[band] = np.maximum(0, np.minimum(rising, falling))

    # Each triangular filter collects the energy in one Mel frequency band.
    mel_power = filters @ power
    return np.log(np.maximum(mel_power, 1e-10)).astype(np.float32)


# Running this module directly gives a quick check of a local ESC-50 download.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the local ESC-50 dataset")
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()

    summary = audit_dataset(arguments.data_dir)
    fold_text = ", ".join(
        f"{fold}={count}" for fold, count in summary.fold_counts.items()
    )
    print(f"Clips: {summary.clips}")
    print(f"Classes: {summary.categories}")
    print(f"Folds: {fold_text}")

    if (
        summary.clips != 2000
        or summary.categories != 50
        or summary.fold_counts != dict.fromkeys(range(1, 6), 400)
    ):
        print(
            "Expected 2000 ESC-50 clips, 50 classes, and 400 clips per fold",
            file=sys.stderr,
        )
        raise SystemExit(1)
