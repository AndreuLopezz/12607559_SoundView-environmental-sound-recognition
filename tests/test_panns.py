import numpy as np
import torch
from scipy.io import wavfile


# PANNs expects 32 kHz waveforms, while ESC-50 stores audio at 44.1 kHz.
def test_prepare_panns_audio_preserves_tone_when_resampling():
    from soundview.panns import prepare_panns_audio

    time = np.arange(44100, dtype=np.float32) / 44100
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    prepared = prepare_panns_audio(audio)

    assert prepared.shape == (32000,)
    assert prepared.dtype == np.float32
    peak_hz = np.fft.rfftfreq(len(prepared), 1 / 32000)[
        np.abs(np.fft.rfft(prepared)).argmax()
    ]
    assert peak_hz == 440


# Extract features in metadata order so every vector keeps its label and fold.
def test_extract_panns_embeddings_keeps_esc50_metadata(tmp_path):
    from soundview.panns import extract_panns_embeddings

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "esc50.csv").write_text(
        "filename,fold,target,category\nfirst.wav,1,7,siren\nsecond.wav,2,3,rain\n",
        encoding="utf-8",
    )
    samples = (10000 * np.sin(2 * np.pi * 440 * np.arange(4410) / 44100)).astype(
        np.int16
    )
    wavfile.write(audio_dir / "first.wav", 44100, samples)
    wavfile.write(audio_dir / "second.wav", 44100, samples)

    class SmallFeatureModel(torch.nn.Module):
        def forward(self, waveforms):
            assert waveforms.shape == (2, 3200)
            return {"embedding": waveforms.mean(dim=1, keepdim=True).repeat(1, 2048)}

    output_path = tmp_path / "features.npz"
    extract_panns_embeddings(
        SmallFeatureModel(), tmp_path, output_path, torch.device("cpu"), batch_size=2
    )

    with np.load(output_path, allow_pickle=False) as saved:
        assert saved["embeddings"].shape == (2, 2048)
        assert saved["embeddings"].dtype == np.float32
        assert saved["filenames"].tolist() == ["first.wav", "second.wav"]
        assert saved["folds"].tolist() == [1, 2]
        assert saved["targets"].tolist() == [7, 3]
