"""The spectrogram dataset and small CNN used for the baseline."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from soundview.data import Esc50Record, load_audio, log_mel_spectrogram


# Calculate each clip's features once on the Colab CPU. The resulting tensors
# fit in memory and can be reused across training epochs without rereading WAVs.
class SpectrogramDataset(Dataset):
    # Standardize each clip on its own, so validation uses no training statistics.
    def __init__(self, data_dir: Path, records: list[Esc50Record]):
        audio_dir = Path(data_dir) / "audio"
        self.examples = []
        for record in records:
            audio, sample_rate = load_audio(audio_dir / record.filename)
            spectrogram = log_mel_spectrogram(audio, sample_rate)
            spectrogram = (spectrogram - spectrogram.mean()) / max(
                float(spectrogram.std()), 1e-6
            )
            features = torch.from_numpy(
                np.ascontiguousarray(spectrogram, dtype=np.float32)
            ).unsqueeze(0)
            self.examples.append((features, record.target))

    # DataLoader uses the length to decide how many examples form an epoch.
    def __len__(self) -> int:
        return len(self.examples)

    # Return the ready-to-train spectrogram together with its numeric label.
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.examples[index]


# Pad only the time axis, since ESC-50 clips can differ slightly in duration.
def pad_spectrogram_batch(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    spectrograms, labels = zip(*batch)
    longest = max(spectrogram.shape[-1] for spectrogram in spectrograms)
    padded = [
        F.pad(spectrogram, (0, longest - spectrogram.shape[-1]))
        for spectrogram in spectrograms
    ]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


# Each block learns patterns and then halves both spectrogram dimensions.
def _conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


# A compact three-block model is enough to establish a trainable baseline.
# Adaptive pooling allows slightly different clip lengths in the same model.
class SmallAudioCNN(nn.Module):
    # Build the feature extractor and a final layer with one score per class.
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.network = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    # Return raw scores, cross-entropy applies the needed softmax internally.
    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        return self.network(spectrograms)
