"""Prepare ESC-50 audio and load the official pretrained PANNs CNN14."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


# ESC-50 uses 44.1 kHz; the chosen AudioSet CNN14 checkpoint expects 32 kHz.
def prepare_panns_audio(audio: np.ndarray) -> np.ndarray:
    return resample_poly(audio, 320, 441).astype(np.float32)


# Import CNN14 from the official source checked out in Colab and restore its
# AudioSet weights before extracting any ESC-50 representations.
def load_panns_model(
    source_dir: Path, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    pytorch_dir = Path(source_dir) / "pytorch"
    model_file = pytorch_dir / "models.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"PANNs model source is missing: {model_file}")

    source_path = str(pytorch_dir)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    spec = importlib.util.spec_from_file_location("panns_official_models", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import PANNs models from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
