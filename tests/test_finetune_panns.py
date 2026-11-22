import json

import numpy as np
import pytest
import torch
from scipy.io import wavfile
from torch import nn


# Fine-tuning should change only the late pretrained layers and the ESC-50 head.
def test_partial_model_freezes_early_layers_and_uses_saved_head():
    from soundview.finetune_panns import (
        make_full_panns_model,
        make_partial_panns_model,
    )

    class SmallPanns(nn.Module):
        def __init__(self):
            super().__init__()
            self.early = nn.Linear(4, 4)
            self.conv_block6 = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 2048)
            self.fc_audioset = nn.Linear(2048, 527)

        def forward(self, waveforms):
            features = self.fc1(self.conv_block6(self.early(waveforms)))
            return {"embedding": features}

    saved_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 50))
    checkpoint = {
        "model_state_dict": saved_head.state_dict(),
        "feature_mean": torch.zeros(2048),
        "feature_scale": torch.ones(2048),
    }
    model = make_partial_panns_model(SmallPanns(), checkpoint)
    scores = model(torch.ones(2, 4))
    scores.sum().backward()

    assert scores.shape == (2, 50)
    assert model.backbone.early.weight.grad is None
    assert model.backbone.fc_audioset.weight.grad is None
    assert model.backbone.conv_block6.weight.grad is not None
    assert model.backbone.fc1.weight.grad is not None
    assert model.head[1].weight.grad is not None

    full_model = make_full_panns_model(SmallPanns(), checkpoint)
    full_model(torch.ones(2, 4)).sum().backward()
    assert full_model.backbone.early.weight.grad is not None
    assert full_model.backbone.fc_audioset.weight.grad is None


# Full fine-tuning uses all pretrained feature layers and saves a test report
# only after the best epoch has been chosen using validation.
def test_full_finetuning_saves_a_selected_test_report(tmp_path):
    from soundview.finetune_panns import train_full_panns_fold

    class SmallPanns(nn.Module):
        def __init__(self):
            super().__init__()
            self.early = nn.Linear(4, 4)
            self.conv_block6 = nn.Linear(4, 4)
            self.fc1 = nn.Linear(4, 2048)
            self.fc_audioset = nn.Linear(2048, 527)

        def forward(self, waveforms):
            values = self.fc1(self.conv_block6(self.early(waveforms[:, :4])))
            return {"embedding": values}

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = ["filename,fold,target,category"]
    samples = np.zeros(4410, dtype=np.int16)
    for fold in range(1, 6):
        for target in range(2):
            filename = f"fold{fold}_{target}.wav"
            rows.append(f"{filename},{fold},{target},class_{target}")
            wavfile.write(audio_dir / filename, 44100, samples)
    (meta_dir / "esc50.csv").write_text("\n".join(rows), encoding="utf-8")

    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 50))
    checkpoint = {
        "model_state_dict": head.state_dict(),
        "feature_mean": torch.zeros(2048),
        "feature_scale": torch.ones(2048),
        "test_fold": 5,
        "validation_macro_f1": 0.1,
    }
    result = train_full_panns_fold(
        SmallPanns(),
        checkpoint,
        tmp_path,
        tmp_path / "full_fold5",
        test_fold=5,
        epochs=1,
        batch_size=2,
        accumulation_steps=2,
        device=torch.device("cpu"),
    )

    assert result["model"] == "panns_full_finetune"
    assert result["test_fold"] == 5
    assert result["test_clips"] == 2
    assert (tmp_path / "full_fold5" / "best_full.pt").is_file()
    assert (tmp_path / "full_fold5" / "training_summary.json").is_file()
    assert (tmp_path / "full_fold5" / "test_evaluation.json").is_file()


# Resume only a completed fold whose checkpoint, split, and settings match.
def test_load_completed_full_fold_checks_saved_run(tmp_path):
    from soundview.finetune_panns import load_completed_full_fold

    settings = {"epochs": 20, "precision": "float32"}
    assert load_completed_full_fold(tmp_path, 5, settings) is None

    (tmp_path / "best_full.pt").touch()
    summary = {
        "model": "panns_full_finetune",
        "test_fold_reserved": 5,
        "validation_fold": 1,
        "best_epoch": 3,
        "best_validation_macro_f1": 0.88,
        "settings": settings,
    }
    report = {
        "model": "panns_full_finetune",
        "test_fold": 5,
        "test_clips": 400,
        "checkpoint_epoch": 3,
        "checkpoint_validation_macro_f1": 0.88,
    }
    (tmp_path / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (tmp_path / "test_evaluation.json").write_text(json.dumps(report), encoding="utf-8")

    assert load_completed_full_fold(tmp_path, 5, settings) == report
    with pytest.raises(ValueError, match="different settings"):
        load_completed_full_fold(tmp_path, 5, {"epochs": 10})
