"""Benchmark partial PANNs fine-tuning and run one full ESC-50 fold."""

import json
import math
import time
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from soundview.data import Esc50Record, load_audio, read_metadata, split_folds
from soundview.evaluate_cnn import summarize_predictions
from soundview.panns import load_panns_model, prepare_panns_audio
from soundview.train_cnn import classification_metrics, set_seed


# Pair CNN14 with the ESC-50 head and the training-fold normalization saved
# during the frozen experiment. Choose which learned feature layers can change.
class PannsEsc50Model(nn.Module):
    def __init__(self, backbone: nn.Module, head_checkpoint: dict, train_all: bool):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if train_all:
            fixed_layers = {
                "spectrogram_extractor",
                "logmel_extractor",
                "fc_audioset",
            }
            trainable_layers = [
                layer
                for name, layer in self.backbone.named_children()
                if name not in fixed_layers
            ]
        else:
            trainable_layers = [self.backbone.conv_block6, self.backbone.fc1]
        for layer in trainable_layers:
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
) -> PannsEsc50Model:
    return PannsEsc50Model(backbone, head_checkpoint, train_all=False)


# Adapt all learned CNN14 feature layers while leaving its fixed spectrogram
# filters and unused AudioSet classifier untouched.
def make_full_panns_model(
    backbone: nn.Module, head_checkpoint: dict
) -> PannsEsc50Model:
    return PannsEsc50Model(backbone, head_checkpoint, train_all=True)


# Load requested ESC-50 waveforms on demand without keeping all resampled
# clips in GPU memory.
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


# Fine-tune all learned CNN14 feature layers on one official split. Validation
# chooses the checkpoint; the held-out test fold is scored once afterward.
def train_full_panns_fold(
    backbone: nn.Module,
    head_checkpoint: dict,
    data_dir: Path,
    output_dir: Path,
    test_fold: int = 5,
    epochs: int = 20,
    batch_size: int = 2,
    accumulation_steps: int = 4,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Colab GPU runtime before fine-tuning PANNs")
        device = torch.device("cuda")
    if head_checkpoint["test_fold"] != test_fold:
        raise ValueError("The frozen head belongs to a different test fold")
    if min(epochs, batch_size, accumulation_steps, patience) <= 0:
        raise ValueError(
            "Epochs, batch size, accumulation, and patience must be positive"
        )
    if min(backbone_lr, head_lr) <= 0:
        raise ValueError("Learning rates must be positive")

    set_seed(seed)
    records = read_metadata(data_dir)
    train_records, validation_records, test_records = split_folds(records, test_fold)
    class_names = [""] * 50
    for record in records:
        class_names[record.target] = record.category

    def make_loader(selected_records: list[Esc50Record], shuffle: bool) -> DataLoader:
        return DataLoader(
            PannsWaveformDataset(data_dir, selected_records),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_waveform_batch,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader(train_records, shuffle=True)
    validation_loader = make_loader(validation_records, shuffle=False)
    model = make_full_panns_model(backbone, head_checkpoint).to(device)
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = backbone_parameters + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )
    loss_function = nn.CrossEntropyLoss()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_full.pt"
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()  # Use pretrained batch-norm statistics with small batches.
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        total_batches = len(train_loader)
        for batch_index, (waveforms, targets) in enumerate(train_loader):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            completed_batches = batch_index + 1
            group_start = batch_index // accumulation_steps * accumulation_steps
            group_size = min(accumulation_steps, total_batches - group_start)
            # Float32 avoids the overflow observed with CNN14 and the saved
            # feature normalization in the first mixed-precision run.
            loss = loss_function(model(waveforms), targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Epoch {epoch}, batch {completed_batches}: non-finite training loss"
                )
            (loss / group_size).backward()
            train_loss += loss.item() * len(targets)
            should_step = (
                completed_batches % accumulation_steps == 0
                or completed_batches == total_batches
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation_loss = 0.0
        validation_targets = []
        validation_predictions = []
        with torch.no_grad():
            for waveforms, targets in validation_loader:
                waveforms = waveforms.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                scores = model(waveforms)
                loss = loss_function(scores, targets)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Epoch {epoch}: non-finite validation loss"
                    )
                validation_loss += loss.item() * len(targets)
                validation_targets.extend(targets.cpu().tolist())
                validation_predictions.extend(scores.argmax(dim=1).cpu().tolist())

        metrics = classification_metrics(validation_targets, validation_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(train_records),
                "validation_loss": validation_loss / len(validation_records),
                **metrics,
            }
        )
        print(
            f"Epoch {epoch}: validation macro-F1 {metrics['macro_f1']:.4f}, "
            f"accuracy {metrics['accuracy']:.4f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "test_fold": test_fold,
                "validation_fold": test_fold % 5 + 1,
                "epoch": epoch,
                "validation_macro_f1": best_f1,
            }
            temporary_path = checkpoint_path.with_name("best_full.pt.part")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No new best validation macro-F1 in {patience} epochs")
                break

    selected = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    model.eval()
    test_targets = []
    test_predictions = []
    test_confidences = []
    test_loader = make_loader(test_records, shuffle=False)
    with torch.no_grad():
        for waveforms, targets in test_loader:
            waveforms = waveforms.to(device, non_blocking=True)
            scores = model(waveforms)
            if not torch.isfinite(scores).all().item():
                raise FloatingPointError("Non-finite test scores")
            confidences, predictions = scores.softmax(dim=1).max(dim=1)
            test_targets.extend(targets.tolist())
            test_predictions.extend(predictions.cpu().tolist())
            test_confidences.extend(confidences.cpu().tolist())

    report = {
        "model": "panns_full_finetune",
        "checkpoint_epoch": best_epoch,
        "checkpoint_validation_macro_f1": best_f1,
        "test_fold": test_fold,
        "test_clips": len(test_records),
        **summarize_predictions(test_targets, test_predictions, class_names),
        "clips": [
            {
                "filename": record.filename,
                "target": target,
                "prediction": prediction,
                "confidence": confidence,
            }
            for record, target, prediction, confidence in zip(
                test_records,
                test_targets,
                test_predictions,
                test_confidences,
                strict=True,
            )
        ],
    }
    summary = {
        "model": "panns_full_finetune",
        "test_fold_reserved": test_fold,
        "validation_fold": test_fold % 5 + 1,
        "training_clips": len(train_records),
        "validation_clips": len(validation_records),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "frozen_head_validation_macro_f1": head_checkpoint["validation_macro_f1"],
        "elapsed_seconds": round(time.perf_counter() - start_time, 1),
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "precision": "float32",
            "gradient_clip_norm": 1.0,
            "patience": patience,
            "seed": seed,
        },
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "test_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Fold {test_fold}: test macro-F1 {report['macro_f1']:.4f}")
    return report
