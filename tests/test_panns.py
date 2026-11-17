import numpy as np


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
