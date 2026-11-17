import pytest


# Include a class the model never predicts when computing macro-F1.
def test_metrics_count_missed_classes():
    from soundview.train_cnn import classification_metrics

    metrics = classification_metrics([0, 1, 2], [0, 1, 1], num_classes=3)

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx((1 + 2 / 3) / 3)
