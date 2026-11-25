"""Compare PANNs fine-tuning settings using validation folds only."""

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from soundview.data import read_metadata, split_folds
from soundview.finetune_panns import (
    PannsWaveformDataset,
    make_full_panns_model,
    pad_waveform_batch,
)
from soundview.train_cnn import classification_metrics, set_seed


# Read a finished trial only when it belongs to the same split and settings.
# An interrupted trial has no summary and can start again safely.
def load_completed_validation_trial(
    output_dir: Path, reserved_test_fold: int, settings: dict
) -> dict | None:
    summary_path = Path(output_dir) / "training_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["settings"] != settings:
        raise ValueError("The saved validation trial used different settings")
    if (
        summary["model"] != "panns_validation_trial"
        or summary["test_fold_reserved"] != reserved_test_fold
        or summary["validation_fold"] != reserved_test_fold % 5 + 1
        or summary["best_epoch"] not in [item["epoch"] for item in summary["history"]]
    ):
        raise ValueError("The saved validation trial is inconsistent")
    return summary


# Train on the three permitted folds and choose an epoch using validation.
# The reserved test fold is identified from metadata but its audio is not read.
def run_panns_validation_trial(
    backbone: nn.Module,
    head_checkpoint: dict,
    data_dir: Path,
    output_dir: Path,
    reserved_test_fold: int = 5,
    epochs: int = 20,
    batch_size: int = 2,
    accumulation_steps: int = 4,
    backbone_lr: float = 1e-6,
    head_lr: float = 1e-5,
    weight_decay: float = 0.01,
    head_dropout: float = 0.2,
    patience: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Colab GPU runtime before PANNs search")
        device = torch.device("cuda")
    if head_checkpoint["test_fold"] != reserved_test_fold:
        raise ValueError("The frozen head belongs to a different test fold")
    if min(epochs, batch_size, accumulation_steps, patience) <= 0:
        raise ValueError(
            "Epochs, batch size, accumulation, and patience must be positive"
        )
    if min(backbone_lr, head_lr) <= 0 or weight_decay < 0:
        raise ValueError(
            "Learning rates must be positive and weight decay non-negative"
        )
    if not 0 <= head_dropout < 1:
        raise ValueError("Head dropout must be between zero and one")

    set_seed(seed)
    records = read_metadata(data_dir)
    train_records, validation_records, _ = split_folds(records, reserved_test_fold)

    def make_loader(selected_records, shuffle):
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
    model.head[0].p = head_dropout
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
        ],
        weight_decay=weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()  # Keep pretrained batch-normalization statistics.
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        total_batches = len(train_loader)
        for batch_index, (waveforms, targets) in enumerate(train_loader):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            completed_batches = batch_index + 1
            group_start = batch_index // accumulation_steps * accumulation_steps
            group_size = min(accumulation_steps, total_batches - group_start)
            loss = loss_function(model(waveforms), targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Epoch {epoch}, batch {completed_batches}: non-finite training loss"
                )
            (loss / group_size).backward()
            train_loss += loss.item() * len(targets)
            if (
                completed_batches % accumulation_steps == 0
                or completed_batches == total_batches
            ):
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
        print(f"Trial epoch {epoch}: validation macro-F1 {metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    summary = {
        "model": "panns_validation_trial",
        "test_fold_reserved": reserved_test_fold,
        "validation_fold": reserved_test_fold % 5 + 1,
        "training_clips": len(train_records),
        "validation_clips": len(validation_records),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "elapsed_seconds": round(time.perf_counter() - start_time, 1),
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "head_dropout": head_dropout,
            "patience": patience,
            "seed": seed,
            "precision": "float32",
            "gradient_clip_norm": 1.0,
        },
        "history": history,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "training_summary.json.part"
    temporary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary_path.replace(output_dir / "training_summary.json")
    return summary
