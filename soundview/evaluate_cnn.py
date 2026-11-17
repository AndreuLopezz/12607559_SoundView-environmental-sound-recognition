"""Evaluate a saved CNN checkpoint on its reserved ESC-50 test fold."""

import json
import time
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from soundview.cnn import SmallAudioCNN, SpectrogramDataset, pad_spectrogram_batch
from soundview.data import read_metadata, split_folds
from soundview.train_cnn import classification_metrics


# Keep the overall scores and the errors for individual classes in one report.
def summarize_predictions(
    targets: list[int], predictions: list[int], class_names: list[str]
) -> dict:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    per_class = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id, class_name in enumerate(class_names)
    ]
    return {
        **classification_metrics(targets, predictions, num_classes=len(labels)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
    }


# Load the selected checkpoint and measure its untouched test fold once.
@torch.no_grad()
def evaluate_cnn(
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    test_fold: int = 5,
    batch_size: int = 32,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Google Colab before evaluation")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["test_fold"] != test_fold:
        raise ValueError("Checkpoint and requested test fold do not match")

    records = read_metadata(data_dir)
    _, _, test_records = split_folds(records, test_fold)
    class_names = [""] * 50
    for record in records:
        class_names[record.target] = record.category

    test_data = SpectrogramDataset(data_dir, test_records)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=pad_spectrogram_batch,
        num_workers=0,
        pin_memory=True,
    )
    model = SmallAudioCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    targets = []
    predictions = []
    confidences = []
    inference_seconds = 0.0
    for features, batch_targets in test_loader:
        features = features.to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores = model(features)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - start

        probabilities = scores.softmax(dim=1)
        confidence, predicted_class = probabilities.max(dim=1)
        targets.extend(batch_targets.tolist())
        predictions.extend(predicted_class.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())

    report = {
        "model": "small_cnn",
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_macro_f1": checkpoint["validation_macro_f1"],
        "test_fold": test_fold,
        "test_clips": len(test_records),
        "inference_ms_per_clip": 1000 * inference_seconds / len(test_records),
        **summarize_predictions(targets, predictions, class_names),
        "clips": [
            {
                "filename": record.filename,
                "target": target,
                "prediction": prediction,
                "confidence": confidence,
            }
            for record, target, prediction, confidence in zip(
                test_records, targets, predictions, confidences, strict=True
            )
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report
