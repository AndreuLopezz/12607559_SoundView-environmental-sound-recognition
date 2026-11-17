import torch
from torch import nn


# Fine-tuning should change only the late pretrained layers and the ESC-50 head.
def test_partial_model_freezes_early_layers_and_uses_saved_head():
    from soundview.finetune_panns import make_partial_panns_model

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
