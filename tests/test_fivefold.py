import json

import pytest


# Use five complete synthetic folds to check pooled and per-fold results.
def test_fivefold_summary_counts_each_clip_once(tmp_path):
    from soundview.fivefold import combine_fold_reports

    report_paths = []
    for fold in range(1, 6):
        clips = [
            {
                "filename": f"fold{fold}_clip{index}.wav",
                "target": index % 50,
                "prediction": 1 if fold == 1 and index == 0 else index % 50,
            }
            for index in range(400)
        ]
        report = {
            "test_fold": fold,
            "test_clips": 400,
            "checkpoint_epoch": 24,
            "per_class": [
                {"class_id": class_id, "class_name": f"class_{class_id}"}
                for class_id in range(50)
            ],
            "clips": clips,
        }
        path = tmp_path / f"fold{fold}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)

    summary_path = tmp_path / "fivefold_summary.json"
    summary = combine_fold_reports(report_paths, summary_path)

    assert summary["test_clips"] == 2000
    assert summary["pooled"]["accuracy"] == pytest.approx(1999 / 2000)
    assert summary["mean_fold_accuracy"] == pytest.approx(1999 / 2000)
    fold_one_f1 = (48 + 14 / 15 + 16 / 17) / 50
    pooled_f1 = (48 + 78 / 79 + 80 / 81) / 50
    assert summary["mean_fold_macro_f1"] == pytest.approx((fold_one_f1 + 4) / 5)
    assert summary["pooled"]["macro_f1"] == pytest.approx(pooled_f1)
    assert summary["pooled"]["confusion_matrix"][0][1] == 1
    assert all(item["support"] == 40 for item in summary["pooled"]["per_class"])
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary

    with pytest.raises(ValueError, match="five folds"):
        combine_fold_reports(report_paths[:-1], tmp_path / "missing.json")

    duplicate = json.loads(report_paths[-1].read_text(encoding="utf-8"))
    duplicate["clips"][0]["filename"] = "fold1_clip0.wav"
    report_paths[-1].write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        combine_fold_reports(report_paths, tmp_path / "duplicate.json")
