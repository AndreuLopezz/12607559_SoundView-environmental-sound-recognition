import pytest


# Check that a missed class affects macro-F1 and appears in the class report.
def test_prediction_summary_keeps_class_errors_visible():
    from soundview.evaluate_cnn import summarize_predictions

    summary = summarize_predictions(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        class_names=["knock", "siren"],
    )

    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert summary["confusion_matrix"] == [[1, 1], [0, 2]]
    assert summary["per_class"][0]["recall"] == pytest.approx(0.5)
    assert summary["per_class"][1]["precision"] == pytest.approx(2 / 3)
