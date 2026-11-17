"""Prepare a small, validation-only partial fine-tuning trial for PANNs."""

import math
import time
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from soundview.data import Esc50Record, load_audio, read_metadata, split_folds
from soundview.panns import load_panns_model, prepare_panns_audio


# Keep early AudioSet layers fixed while allowing the last convolution block,
# embedding layer, and ESC-50 classifier to adapt to the new sound classes.
class PannsPartialModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_checkpoint: dict):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for layer in (self.backbone.conv_block6, self.backbone.fc1):
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 50))
        self.head.load_state_dict(head_checkpoint["model_state_dict"])
        self.register_buffer("feature_mean", head_checkpoint["feature_mean"].float())
        self.register_buffer("feature_scale", head_checkpoint["feature_scale"].float())

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(waveforms)["embedding"]
        normalized = (embedding - self.feature_mean) / self.feature_scale
        return self.head(normalized)


# Build the model from the same frozen-head checkpoint selected using the
# validation fold, so the trial starts from an already trained ESC-50 head.
def make_partial_panns_model(
    backbone: nn.Module, head_checkpoint: dict
) -> PannsPartialModel:
    return PannsPartialModel(backbone, head_checkpoint)


# Load only requested ESC-50 waveforms on demand; the pilot never reads its
# reserved test fold and does not keep all resampled clips in GPU memory.
class PannsWaveformDataset(Dataset):
    def __init__(self, data_dir: Path, records: list[Esc50Record]):
        self.audio_dir = Path(data_dir) / "audio"
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        audio, _ = load_audio(self.audio_dir / record.filename)
        waveform = torch.from_numpy(prepare_panns_audio(audio))
        return waveform, record.target


# Pad small duration differences within a batch before CNN14 sees waveforms.
def pad_waveform_batch(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms, targets = zip(*batch)
    return pad_sequence(waveforms, batch_first=True), torch.tensor(targets)


# Measure a few real training updates on the T4 before choosing an epoch count
# for a full partial fine-tuning run. These updates are not saved or tested.
def benchmark_partial_finetuning(
    data_dir: Path,
    source_dir: Path,
    pretrained_checkpoint: Path,
    head_checkpoint: Path,
    test_fold: int = 5,
    batch_size: int = 2,
    measured_batches: int = 3,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU runtime before fine-tuning PANNs")
    if batch_size <= 0 or measured_batches <= 0:
        raise ValueError("Batch size and measured batches must be positive")

    records = read_metadata(data_dir)
    train_records, _, _ = split_folds(records, test_fold)
    loader = DataLoader(
        PannsWaveformDataset(data_dir, train_records),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_waveform_batch,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda")
    backbone = load_panns_model(source_dir, pretrained_checkpoint, device)
    saved_head = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    if saved_head["test_fold"] != test_fold:
        raise ValueError("The frozen head belongs to a different test fold")
    model = make_partial_panns_model(backbone, saved_head).to(device)
    model.train()
    model.backbone.eval()  # Preserve running statistics in pretrained layers.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.conv_block6.parameters(), "lr": 1e-5},
            {"params": model.backbone.fc1.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-4},
        ]
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_function = nn.CrossEntropyLoss()
    durations = []
    torch.cuda.reset_peak_memory_stats(device)
    batches = iter(loader)

    for batch_number in range(measured_batches + 1):
        start = time.perf_counter()
        waveforms, targets = next(batches)
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_function(model(waveforms), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if batch_number > 0:
            durations.append(time.perf_counter() - start)

    seconds_per_batch = mean(durations)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "test_fold_reserved": test_fold,
        "training_clips": len(train_records),
        "batch_size": batch_size,
        "measured_batches": measured_batches,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "seconds_per_batch": round(seconds_per_batch, 3),
        "estimated_training_minutes_per_epoch": round(
            seconds_per_batch * math.ceil(len(train_records) / batch_size) / 60, 1
        ),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 2
        ),
    }
