import numpy as np
import torch
from scipy.io import wavfile

from soundview.data import Esc50Record


# The classifier must produce one score for each of ESC-50's 50 classes.
def test_cnn_returns_fifty_class_scores():
    from soundview.cnn import SmallAudioCNN

    model = SmallAudioCNN()
    scores = model(torch.zeros(2, 1, 64, 32))

    assert scores.shape == (2, 50)


# A small WAV exercises the same loading and log-Mel path used for training.
def test_dataset_prepares_a_clip_once(tmp_path):
    from soundview.cnn import SpectrogramDataset

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    time = np.arange(4410) / 44100
    samples = (16000 * np.sin(2 * np.pi * 440 * time)).astype(np.int16)
    wavfile.write(audio_dir / "tone.wav", 44100, samples)
    record = Esc50Record("tone.wav", fold=1, target=7, category="siren")

    dataset = SpectrogramDataset(tmp_path, [record])
    spectrogram, target = dataset[0]

    assert len(dataset) == 1
    assert spectrogram.shape[0:2] == (1, 64)
    assert abs(float(spectrogram.mean())) < 1e-5
    assert target == 7


# Padding lets clips with slightly different lengths share one training batch.
def test_batch_padding_keeps_labels_and_fills_shorter_clip():
    from soundview.cnn import pad_spectrogram_batch

    short = torch.ones(1, 64, 3)
    long = torch.ones(1, 64, 5)
    features, targets = pad_spectrogram_batch([(short, 2), (long, 4)])

    assert features.shape == (2, 1, 64, 5)
    assert targets.tolist() == [2, 4]
    assert torch.count_nonzero(features[0, :, :, 3:]) == 0
