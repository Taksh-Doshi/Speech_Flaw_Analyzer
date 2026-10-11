"""Audio conversion tests for baseline preparation."""

import numpy as np
import soundfile as sf

from scripts.prepare_baseline import convert_audio


def test_convert_audio_writes_mono_16khz_pcm_wav(tmp_path):
    """A generated source tone is trimmed and converted to the required WAV format."""
    input_path = tmp_path / "input audio.wav"
    output_path = tmp_path / "output audio.wav"
    sample_rate = 22050
    samples = np.arange(sample_rate) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 220 * samples)
    sf.write(input_path, audio, sample_rate)

    duration = convert_audio(input_path, output_path, "00:00:00", "00:00:00.5")

    info = sf.info(output_path)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert abs(info.duration - duration) < 1 / info.samplerate
