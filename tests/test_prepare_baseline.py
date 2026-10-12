"""Audio conversion tests for baseline preparation."""

import numpy as np
import soundfile as sf

from scripts.prepare_baseline import convert_audio, upsert_dataset_card_row


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


def test_upsert_dataset_card_row_replaces_existing_speech_id(tmp_path):
    """Updating a baseline replaces matching rows without duplicating the ID."""
    dataset_card = tmp_path / "DATASET_CARD.md"
    dataset_card.write_text(
        "# Dataset card\n\n"
        "| speech_id | source URL | notes |\n"
        "|---|---|---|\n"
        "| speech01 | old source | old details |\n"
        "| speech02 | other source | keep this |\n"
        "| speech01 | duplicate source | duplicate details |\n",
        encoding="utf-8",
    )
    replacement = "| speech01 | new source | updated details |\n"

    upsert_dataset_card_row(dataset_card, "speech01", replacement)

    contents = dataset_card.read_text(encoding="utf-8")
    assert contents.count("| speech01 |") == 1
    assert replacement in contents
    assert "| speech02 | other source | keep this |" in contents
