import csv
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile


# Build a small ESC-50 style CSV so each test can focus on one data problem.
def write_metadata(dataset_dir, rows):
    metadata_dir = dataset_dir / "meta"
    metadata_dir.mkdir(parents=True)
    with (metadata_dir / "esc50.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["filename", "fold", "target", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)


# A clip in the test fold must never also appear in training or validation.
def test_split_keeps_official_folds_separate(tmp_path):
    from soundview.data import read_metadata, split_folds

    rows = [
        {
            "filename": f"clip_{fold}.wav",
            "fold": fold,
            "target": fold - 1,
            "category": f"class_{fold}",
        }
        for fold in range(1, 6)
    ]
    write_metadata(tmp_path, rows)

    train, validation, test = split_folds(read_metadata(tmp_path), test_fold=2)

    assert {record.filename for record in train} == {
        "clip_1.wav",
        "clip_4.wav",
        "clip_5.wav",
    }
    assert [record.filename for record in validation] == ["clip_3.wav"]
    assert [record.filename for record in test] == ["clip_2.wav"]


# The validation choice wraps from fold five back to fold one.
def test_last_test_fold_uses_first_fold_for_validation(tmp_path):
    from soundview.data import read_metadata, split_folds

    write_metadata(
        tmp_path,
        [
            {
                "filename": f"clip_{fold}.wav",
                "fold": fold,
                "target": 0,
                "category": "siren",
            }
            for fold in range(1, 6)
        ],
    )

    _, validation, test = split_folds(read_metadata(tmp_path), test_fold=5)

    assert [record.fold for record in validation] == [1]
    assert [record.fold for record in test] == [5]


# An invalid fold number should fail before it reaches the training pipeline.
def test_metadata_rejects_unknown_fold(tmp_path):
    from soundview.data import read_metadata

    write_metadata(
        tmp_path,
        [{"filename": "clip.wav", "fold": 6, "target": 0, "category": "siren"}],
    )

    with pytest.raises(ValueError, match="fold"):
        read_metadata(tmp_path)


# A CSV entry without its WAV file means the dataset download is incomplete.
def test_dataset_audit_reports_missing_audio(tmp_path):
    from soundview.data import audit_dataset

    write_metadata(
        tmp_path,
        [{"filename": "missing.wav", "fold": 1, "target": 0, "category": "siren"}],
    )
    (tmp_path / "audio").mkdir()

    with pytest.raises(FileNotFoundError, match="missing.wav"):
        audit_dataset(tmp_path)


# The command should show the counts and reject a dataset with only a few clips.
def test_data_command_reports_dataset_inventory(tmp_path):
    rows = [
        {
            "filename": f"clip_{fold}.wav",
            "fold": fold,
            "target": fold - 1,
            "category": f"class_{fold}",
        }
        for fold in range(1, 6)
    ]
    write_metadata(tmp_path, rows)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for row in rows:
        wavfile.write(
            audio_dir / row["filename"], 44100, np.zeros(4410, dtype=np.int16)
        )

    result = subprocess.run(
        [sys.executable, "-m", "soundview.data", "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Clips: 5" in result.stdout
    assert "Folds: 1=1, 2=1, 3=1, 4=1, 5=1" in result.stdout
    assert "Expected 2000 ESC-50 clips" in result.stderr


# A known PCM value catches a wrong amplitude scale while keeping the source rate.
def test_audio_keeps_original_rate_and_scales_pcm(tmp_path):
    from soundview.data import load_audio

    path = tmp_path / "clip.wav"
    wavfile.write(path, 44100, np.full(44100, 16384, dtype=np.int16))

    audio, sample_rate = load_audio(path)

    assert sample_rate == 44100
    assert len(audio) == 44100
    assert audio.dtype == np.float32
    assert np.mean(audio[100:-100]) == pytest.approx(0.5, abs=0.01)


# Silence must stay finite, while a tone should produce visible spectral energy.
def test_log_mel_is_finite_and_responds_to_sound():
    from soundview.data import log_mel_spectrogram

    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * 440 * time)

    tone_spectrogram = log_mel_spectrogram(tone, sample_rate)
    silence_spectrogram = log_mel_spectrogram(np.zeros_like(tone), sample_rate)

    assert tone_spectrogram.shape[0] == 64
    assert tone_spectrogram.shape == silence_spectrogram.shape
    assert np.isfinite(tone_spectrogram).all()
    assert tone_spectrogram.max() > silence_spectrogram.max() + 1
