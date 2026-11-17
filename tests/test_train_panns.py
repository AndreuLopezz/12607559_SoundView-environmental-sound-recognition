import numpy as np
import torch


# Keep test and validation embeddings out of the head's training statistics.
def test_frozen_head_uses_separate_folds_and_training_statistics():
    from soundview.train_panns import normalize_features, split_feature_folds

    folds = np.array([1, 2, 3, 4, 5])
    train, validation, test = split_feature_folds(folds, test_fold=5)

    assert train.tolist() == [1, 2, 3]
    assert validation.tolist() == [0]
    assert test.tolist() == [4]

    features = np.array([[100.0], [1.0], [2.0], [3.0], [-100.0]])
    normalized, mean, scale = normalize_features(features, train)

    assert mean.tolist() == [2.0]
    assert np.isclose(normalized[train].mean(), 0)
    assert normalized[validation, 0] > 0
    assert normalized[test, 0] < 0
    assert scale[0] > 0


# A tiny CPU run checks checkpoint selection and the test report without
# processing ESC-50 audio on the local machine.
def test_frozen_head_writes_a_complete_fold_report(tmp_path):
    from soundview.train_panns import train_frozen_panns_fold

    metadata = tmp_path / "meta"
    metadata.mkdir()
    indices = np.arange(2000)
    folds = indices // 400 + 1
    targets = indices % 50
    filenames = np.array([f"clip{index}.wav" for index in indices])
    rows = ["filename,fold,target,category"] + [
        f"{filename},{fold},{target},class_{target}"
        for filename, fold, target in zip(filenames, folds, targets, strict=True)
    ]
    (metadata / "esc50.csv").write_text("\n".join(rows), encoding="utf-8")
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        embeddings=np.zeros((2000, 2048), dtype=np.float32),
        filenames=filenames,
        folds=folds,
        targets=targets,
    )

    output_dir = tmp_path / "fold5"
    report = train_frozen_panns_fold(
        tmp_path,
        features_path,
        output_dir,
        test_fold=5,
        epochs=1,
        batch_size=400,
        device=torch.device("cpu"),
    )

    assert report["model"] == "panns_frozen_head"
    assert report["test_clips"] == 400
    assert len(report["clips"]) == 400
    assert all(row["support"] == 8 for row in report["per_class"])
    assert (output_dir / "best_head.pt").is_file()
    assert (output_dir / "training_summary.json").is_file()
    assert (output_dir / "test_evaluation.json").is_file()
