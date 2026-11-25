import numpy as np
import pytest
import torch
from scipy.io import wavfile
from torch import nn


# A search trial trains and validates normally while leaving its reserved
# test audio unopened, even when that audio is absent from the local dataset.
def test_validation_trial_never_reads_test_audio(tmp_path):
    from soundview.search_panns import (
        load_completed_validation_trial,
        run_panns_validation_trial,
    )

    class SmallPanns(nn.Module):
        def __init__(self):
            super().__init__()
            self.early = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 2048)
            self.fc_audioset = nn.Linear(2048, 527)

        def forward(self, waveforms):
            return {"embedding": self.fc1(self.early(waveforms[:, :4]))}

    (tmp_path / "meta").mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = ["filename,fold,target,category"]
    for fold in range(1, 6):
        for target in range(2):
            filename = f"fold{fold}_{target}.wav"
            rows.append(f"{filename},{fold},{target},class_{target}")
            if fold != 5:
                wavfile.write(audio_dir / filename, 44100, np.zeros(4410, np.int16))
    (tmp_path / "meta" / "esc50.csv").write_text("\n".join(rows), encoding="utf-8")

    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 50))
    saved_head = {
        "model_state_dict": head.state_dict(),
        "feature_mean": torch.zeros(2048),
        "feature_scale": torch.ones(2048),
        "test_fold": 5,
    }
    settings = {
        "epochs": 1,
        "batch_size": 2,
        "accumulation_steps": 2,
        "backbone_lr": 1e-6,
        "head_lr": 1e-5,
        "weight_decay": 0.05,
        "head_dropout": 0.4,
        "patience": 1,
        "seed": 42,
        "precision": "float32",
        "gradient_clip_norm": 1.0,
    }
    output_dir = tmp_path / "trial"
    assert load_completed_validation_trial(output_dir, 5, settings) is None
    summary = run_panns_validation_trial(
        SmallPanns(),
        saved_head,
        tmp_path,
        output_dir,
        reserved_test_fold=5,
        epochs=1,
        batch_size=2,
        accumulation_steps=2,
        backbone_lr=1e-6,
        head_lr=1e-5,
        weight_decay=0.05,
        head_dropout=0.4,
        patience=1,
        device=torch.device("cpu"),
    )

    assert summary["settings"] == settings
    assert summary["training_clips"] == 6
    assert summary["validation_clips"] == 2
    assert summary["best_epoch"] == 1
    assert (output_dir / "training_summary.json").is_file()
    assert not (output_dir / "test_evaluation.json").exists()
    assert load_completed_validation_trial(output_dir, 5, settings) == summary
    with pytest.raises(ValueError, match="different settings"):
        load_completed_validation_trial(output_dir, 5, {"epochs": 2})
