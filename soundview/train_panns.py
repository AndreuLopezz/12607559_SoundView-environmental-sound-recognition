"""Train a 50-class head on frozen PANNs embeddings from ESC-50."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from soundview.data import read_metadata
from soundview.evaluate_cnn import summarize_predictions
from soundview.train_cnn import classification_metrics, set_seed


# For each run, hold out one official fold for test and the following fold for
# validation, matching the protocol used for the CNN baseline.
def split_feature_folds(
    folds: np.ndarray, test_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if test_fold not in range(1, 6):
        raise ValueError("Test fold must be between 1 and 5")
    validation_fold = test_fold % 5 + 1
    train = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    validation = np.flatnonzero(folds == validation_fold)
    test = np.flatnonzero(folds == test_fold)
    return train, validation, test


# Fit the scale on training embeddings only; applying it to validation and test
# does not reveal their distributions during model selection.
def normalize_features(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_indices].mean(axis=0).astype(np.float32)
    scale = np.maximum(features[train_indices].std(axis=0), 1e-6).astype(np.float32)
    return ((features - mean) / scale).astype(np.float32), mean, scale


# Check that the cached vectors still match the official CSV before fitting a
# classifier; the cache can then be reused without rereading all WAV files.
def load_panns_features(
    data_dir: Path, features_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(features_path, allow_pickle=False) as saved:
        features = saved["embeddings"]
        filenames = saved["filenames"]
        folds = saved["folds"]
        targets = saved["targets"]
    records = read_metadata(data_dir)
    expected = {record.filename: record for record in records}
    if (
        len(records) != 2000
        or features.shape != (2000, 2048)
        or len(set(filenames.tolist())) != 2000
    ):
        raise ValueError("Expected one 2048-value embedding per ESC-50 clip")
    for filename, fold, target in zip(filenames, folds, targets, strict=True):
        record = expected.get(str(filename))
        if record is None or record.fold != fold or record.target != target:
            raise ValueError(f"Embedding metadata does not match ESC-50: {filename}")
    if not np.isfinite(features).all():
        raise ValueError("The PANNs feature cache contains non-finite values")

    class_names = [""] * 50
    for record in records:
        class_names[record.target] = record.category
    return features, filenames, folds, targets, class_names


# Train only the new linear classifier; the large pretrained CNN14 is never
# updated in this first transfer-learning experiment.
def _train_head_epoch(
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    head.train()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_clips = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(head(features), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        total_clips += len(targets)
    return total_loss / total_clips


# Return predicted classes and their top probabilities after every example in
# a fold has been scored, including classes that receive no predictions.
@torch.no_grad()
def _predict_head(
    head: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict:
    head.eval()
    all_predictions = []
    all_confidences = []
    total_loss = 0.0
    loss_function = nn.CrossEntropyLoss()
    for start in range(0, len(targets), batch_size):
        batch_features = features[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        scores = head(batch_features)
        total_loss += loss_function(scores, batch_targets).item() * len(batch_targets)
        confidence, prediction = scores.softmax(dim=1).max(dim=1)
        all_predictions.extend(prediction.cpu().tolist())
        all_confidences.extend(confidence.cpu().tolist())
    return {
        "loss": total_loss / len(targets),
        "predictions": all_predictions,
        "confidences": all_confidences,
    }


# Run one fixed train/validation/test split and save the selected head and
# its test report to Drive. The notebook calls this once for each fold.
def train_frozen_panns_fold(
    data_dir: Path,
    features_path: Path,
    output_dir: Path,
    test_fold: int,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 10,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Colab GPU runtime before training PANNs")
        device = torch.device("cuda")
    if epochs <= 0 or batch_size <= 0 or patience <= 0 or learning_rate <= 0:
        raise ValueError("Training settings must be positive")

    features, filenames, folds, targets, class_names = load_panns_features(
        data_dir, features_path
    )
    train_indices, validation_indices, test_indices = split_feature_folds(
        folds, test_fold
    )
    normalized, feature_mean, feature_scale = normalize_features(
        features, train_indices
    )
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    train_features = torch.from_numpy(normalized[train_indices])
    train_targets = torch.from_numpy(targets[train_indices].astype(np.int64))
    validation_features = torch.from_numpy(normalized[validation_indices])
    validation_targets = torch.from_numpy(targets[validation_indices].astype(np.int64))
    test_features = torch.from_numpy(normalized[test_indices])
    test_targets = torch.from_numpy(targets[test_indices].astype(np.int64))
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )

    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 50)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = output_dir / "best_head.pt"

    for epoch in range(1, epochs + 1):
        train_loss = _train_head_epoch(head, train_loader, optimizer, device)
        validation = _predict_head(
            head, validation_features, validation_targets, device, batch_size
        )
        scores = classification_metrics(
            validation_targets.tolist(), validation["predictions"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "loss": validation["loss"],
                **scores,
            }
        )
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Fold {test_fold}, epoch {epoch}: "
                f"validation macro-F1 {scores['macro_f1']:.4f}"
            )

        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": head.state_dict(),
                    "feature_mean": torch.from_numpy(feature_mean),
                    "feature_scale": torch.from_numpy(feature_scale),
                    "test_fold": test_fold,
                    "validation_fold": test_fold % 5 + 1,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Fold {test_fold}: no new best in {patience} epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(checkpoint["model_state_dict"])
    test_result = _predict_head(head, test_features, test_targets, device, batch_size)
    test_predictions = test_result["predictions"]
    report = {
        "model": "panns_frozen_head",
        "checkpoint_epoch": best_epoch,
        "checkpoint_validation_macro_f1": best_f1,
        "test_fold": test_fold,
        "test_clips": len(test_indices),
        **summarize_predictions(test_targets.tolist(), test_predictions, class_names),
        "clips": [
            {
                "filename": str(filenames[index]),
                "target": int(targets[index]),
                "prediction": prediction,
                "confidence": confidence,
            }
            for index, prediction, confidence in zip(
                test_indices,
                test_predictions,
                test_result["confidences"],
                strict=True,
            )
        ],
    }
    summary = {
        "model": "panns_frozen_head",
        "test_fold_reserved": test_fold,
        "validation_fold": test_fold % 5 + 1,
        "training_clips": len(train_indices),
        "validation_clips": len(validation_indices),
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
    with (output_dir / "test_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Fold {test_fold}: test macro-F1 {report['macro_f1']:.4f}")
    return report
