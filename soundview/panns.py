"""Prepare ESC-50 audio and load the official pretrained PANNs CNN14."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.nn.utils.rnn import pad_sequence

from soundview.data import load_audio, read_metadata


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


# Run the frozen pretrained network once per clip and keep its features beside
# the official filename, fold, and target needed for later training splits.
@torch.no_grad()
def extract_panns_embeddings(
    model: torch.nn.Module,
    data_dir: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    records = read_metadata(data_dir)
    model.eval()
    feature_batches = []
    start_time = time.perf_counter()

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        waveforms = []
        for record in batch_records:
            audio, _ = load_audio(Path(data_dir) / "audio" / record.filename)
            waveforms.append(torch.from_numpy(prepare_panns_audio(audio)))
        padded = pad_sequence(waveforms, batch_first=True).to(device)
        embeddings = model(padded)["embedding"]
        if embeddings.shape != (len(batch_records), 2048):
            raise ValueError(f"Unexpected PANNs embedding shape: {embeddings.shape}")
        feature_batches.append(embeddings.cpu().numpy().astype(np.float32))

        processed = start + len(batch_records)
        if processed % 200 == 0 or processed == len(records):
            print(f"Prepared {processed}/{len(records)} audio clips")

    features = np.concatenate(feature_batches)
    if not np.isfinite(features).all():
        raise ValueError("PANNs produced non-finite embeddings")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.part")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            embeddings=features,
            filenames=np.array([record.filename for record in records]),
            folds=np.array([record.fold for record in records]),
            targets=np.array([record.target for record in records]),
        )
    temporary_path.replace(output_path)
    return {
        "clips": len(records),
        "embedding_dim": features.shape[1],
        "elapsed_seconds": round(time.perf_counter() - start_time, 1),
    }
