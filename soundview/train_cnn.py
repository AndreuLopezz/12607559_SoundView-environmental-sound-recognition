"""Train the CNN baseline on ESC-50 without touching the held-out test fold."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from soundview.cnn import SmallAudioCNN, SpectrogramDataset, pad_spectrogram_batch
from soundview.data import read_metadata, split_folds


# Macro-F1 gives each ESC-50 class equal weight, including classes the model
# misses completely. Accuracy shows the overall proportion of correct clips.
def classification_metrics(
    targets: list[int], predictions: list[int], num_classes: int = 50
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
    }


# Make the initial weights, example order, and other random choices repeatable.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Update the network on every training batch and report average loss per clip.
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_clips = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(features)
        loss = loss_function(scores, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        total_clips += len(targets)
    return total_loss / total_clips


# Evaluate only on the validation fold. Predictions are kept as class IDs so
# macro-F1 and accuracy can be computed after all batches have finished.
@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        scores = model(features)
        total_loss += loss_function(scores, targets).item() * len(targets)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(scores.argmax(dim=1).cpu().tolist())

    return {
        "loss": total_loss / len(all_targets),
        **classification_metrics(all_targets, all_predictions),
    }


# Prepare the features and train only when called from the Colab notebook.
def train_cnn(
    data_dir: Path,
    output_dir: Path,
    test_fold: int = 5,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 5,
    seed: int = 42,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Google Colab before training")
    if test_fold not in range(1, 6):
        raise ValueError("Test fold must be between 1 and 5")
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("Epochs, batch size, and patience must be positive")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    set_seed(seed)
    device = torch.device("cuda")
    start_time = time.perf_counter()
    records = read_metadata(data_dir)
    train_records, validation_records, _ = split_folds(records, test_fold)
    validation_fold = test_fold % 5 + 1
    print(
        f"Preparing {len(train_records)} training and "
        f"{len(validation_records)} validation clips on the Colab CPU..."
    )
    train_data = SpectrogramDataset(data_dir, train_records)
    validation_data = SpectrogramDataset(data_dir, validation_records)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_spectrogram_batch,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        collate_fn=pad_spectrogram_batch,
        num_workers=0,
        pin_memory=True,
    )

    model = SmallAudioCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        validation = validate_epoch(model, validation_loader, loss_function, device)
        history.append({"epoch": epoch, "train_loss": training_loss, **validation})
        print(
            f"Epoch {epoch:02d}: train loss {training_loss:.4f}, "
            f"validation loss {validation['loss']:.4f}, "
            f"macro-F1 {validation['macro_f1']:.4f}, "
            f"accuracy {validation['accuracy']:.4f}"
        )

        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "test_fold": test_fold,
                    "validation_fold": validation_fold,
                    "sample_rate": 44100,
                    "n_mels": 64,
                    "n_fft": 2048,
                    "hop_length": 1024,
                },
                output_dir / "best_cnn.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Stopping after validation macro-F1 stopped improving.")
                break

    summary = {
        "model": "small_cnn",
        "test_fold_reserved": test_fold,
        "validation_fold": validation_fold,
        "training_clips": len(train_data),
        "validation_clips": len(validation_data),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "elapsed_seconds": round(time.perf_counter() - start_time, 1),
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
        },
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Best validation macro-F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoint and summary saved in {output_dir}")
