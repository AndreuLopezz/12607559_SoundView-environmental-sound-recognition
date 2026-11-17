"""Combine five held-out ESC-50 test reports from the same model."""

import json
from pathlib import Path
from statistics import mean

from soundview.evaluate_cnn import summarize_predictions
from soundview.train_cnn import classification_metrics


# Check all five reports before combining them, so no clip is counted twice.
def combine_fold_reports(report_paths: list[Path], output_path: Path) -> dict:
    reports = []
    for path in report_paths:
        with Path(path).open(encoding="utf-8") as file:
            reports.append(json.load(file))
    reports.sort(key=lambda report: report["test_fold"])
    if [report["test_fold"] for report in reports] != list(range(1, 6)):
        raise ValueError("Expected reports for all five folds")

    model_name = reports[0]["model"]
    if any(report["model"] != model_name for report in reports):
        raise ValueError("All five reports must come from the same model")
    class_names = [entry["class_name"] for entry in reports[0]["per_class"]]
    if len(class_names) != 50:
        raise ValueError("Expected the 50 ESC-50 classes")

    filenames = set()
    all_clips = []
    fold_scores = []
    for report in reports:
        fold = report["test_fold"]
        clips = report["clips"]
        if report["test_clips"] != 400 or len(clips) != 400:
            raise ValueError(f"Test fold {fold} must contain 400 clips")
        report_class_ids = [entry["class_id"] for entry in report["per_class"]]
        report_class_names = [entry["class_name"] for entry in report["per_class"]]
        if report_class_ids != list(range(50)) or report_class_names != class_names:
            raise ValueError(f"Class labels differ in test fold {fold}")

        for clip in clips:
            if clip["filename"] in filenames:
                raise ValueError(f"Found duplicate clip: {clip['filename']}")
            if clip["target"] not in range(50) or clip["prediction"] not in range(50):
                raise ValueError(f"Invalid class ID in test fold {fold}")
            filenames.add(clip["filename"])
            all_clips.append({"test_fold": fold, **clip})

        fold_metrics = classification_metrics(
            [clip["target"] for clip in clips],
            [clip["prediction"] for clip in clips],
        )
        fold_scores.append(
            {
                "test_fold": fold,
                "test_clips": len(clips),
                "checkpoint_epoch": report["checkpoint_epoch"],
                **fold_metrics,
            }
        )

    pooled = summarize_predictions(
        [clip["target"] for clip in all_clips],
        [clip["prediction"] for clip in all_clips],
        class_names,
    )
    if any(entry["support"] != 40 for entry in pooled["per_class"]):
        raise ValueError("Expected 40 test clips per ESC-50 class")

    summary = {
        "model": model_name,
        "test_clips": len(all_clips),
        "mean_fold_accuracy": mean(fold["accuracy"] for fold in fold_scores),
        "mean_fold_macro_f1": mean(fold["macro_f1"] for fold in fold_scores),
        "folds": fold_scores,
        "pooled": pooled,
        "clips": all_clips,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
