"""Tests for synthetic audio flaw injection."""

import librosa
import numpy as np
import pytest

from speechflaw.dataset_builder.injectors import apply_flaws
from speechflaw.dataset_builder.validate import (
    measure_f0_std_semitones,
    measure_silent_duration,
    validate_flaws,
)
from speechflaw.seed import set_global_seed


@pytest.fixture
def speech_like_audio():
    """Build a deterministic voiced, harmonically rich signal."""
    set_global_seed(42)
    sample_rate = 16000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    fundamental = 145 + 12 * np.sin(2 * np.pi * 0.7 * time)
    phase = 2 * np.pi * np.cumsum(fundamental) / sample_rate
    audio = envelope * (
        np.sin(phase)
        + 0.35 * np.sin(2 * phase + 0.2)
        + 0.15 * np.sin(3 * phase + 0.4)
    )
    return audio.astype(np.float32), sample_rate


@pytest.mark.parametrize(
    ("flaw_type", "severity", "rate", "label_type"),
    [
        ("speed_up", 0.6, 1.3, "too_fast"),
        ("slow_down", 0.5, 0.8, "too_slow"),
    ],
)
def test_apply_flaws_uses_expected_stretched_length(
    speech_like_audio, flaw_type, severity, rate, label_type
):
    audio, sample_rate = speech_like_audio
    start_sample = int(0.5 * sample_rate)
    end_sample = int(1.5 * sample_rate)

    result, labels = apply_flaws(
        audio,
        sample_rate,
        [
            {
                "type": flaw_type,
                "start": 0.5,
                "end": 1.5,
                "severity": severity,
            }
        ],
    )

    expected_stretched_length = round((end_sample - start_sample) / rate)
    assert result.shape[-1] == (
        audio.shape[-1] - (end_sample - start_sample) + expected_stretched_length
    )
    assert labels[0]["type"] == label_type
    assert labels[0]["params"]["rate"] == rate


def test_apply_flaws_preserves_untouched_audio_bit_identically(speech_like_audio):
    audio, sample_rate = speech_like_audio
    start_sample = int(0.5 * sample_rate)
    end_sample = int(1.5 * sample_rate)
    stretched_length = librosa.effects.time_stretch(
        audio[start_sample:end_sample], rate=1.25
    ).shape[-1]

    result, _ = apply_flaws(
        audio,
        sample_rate,
        [{"type": "speed_up", "start": 0.5, "end": 1.5, "severity": 0.5}],
    )

    assert np.array_equal(result[:start_sample], audio[:start_sample])
    assert np.array_equal(
        result[start_sample + stretched_length :],
        audio[end_sample:],
    )


def test_apply_flaws_shifts_sequential_labels_on_output_timeline(
    speech_like_audio,
):
    audio, sample_rate = speech_like_audio
    first_start, first_end = int(0.2 * sample_rate), int(0.6 * sample_rate)
    second_start, second_end = int(1.0 * sample_rate), int(1.4 * sample_rate)
    first_length = librosa.effects.time_stretch(
        audio[first_start:first_end], rate=1.2
    ).shape[-1]
    second_length = librosa.effects.time_stretch(
        audio[second_start:second_end], rate=0.8
    ).shape[-1]

    _, labels = apply_flaws(
        audio,
        sample_rate,
        [
            {"type": "speed_up", "start": 0.2, "end": 0.6, "severity": 0.4},
            {"type": "slow_down", "start": 1.0, "end": 1.4, "severity": 0.5},
        ],
    )

    first_start_output = first_start
    first_end_output = first_start_output + first_length
    second_start_output = second_start + first_length - (first_end - first_start)
    second_end_output = second_start_output + second_length
    assert labels[0]["start"] == pytest.approx(first_start_output / sample_rate)
    assert labels[0]["end"] == pytest.approx(first_end_output / sample_rate)
    assert labels[1]["start"] == pytest.approx(second_start_output / sample_rate)
    assert labels[1]["end"] == pytest.approx(second_end_output / sample_rate)


def test_apply_flaws_rejects_overlapping_specs(speech_like_audio):
    audio, sample_rate = speech_like_audio

    with pytest.raises(ValueError, match="must not overlap"):
        apply_flaws(
            audio,
            sample_rate,
            [
                {"type": "speed_up", "start": 0.4, "end": 1.0, "severity": 0.5},
                {"type": "slow_down", "start": 0.9, "end": 1.2, "severity": 0.5},
            ],
        )


def test_monotone_compresses_pitch_and_preserves_length(speech_like_audio):
    audio, sample_rate = speech_like_audio
    spec = {"type": "monotone", "start": 0.2, "end": 1.8, "severity": 0.9}

    result, labels = apply_flaws(audio, sample_rate, [spec])

    original_span = audio[int(0.2 * sample_rate) : int(1.8 * sample_rate)]
    flawed_span = result[
        int(labels[0]["start"] * sample_rate) : int(labels[0]["end"] * sample_rate)
    ]
    original_std = measure_f0_std_semitones(original_span, sample_rate)
    flawed_std = measure_f0_std_semitones(flawed_span, sample_rate)
    assert result.shape == audio.shape
    assert labels[0]["type"] == "monotone"
    assert flawed_std <= original_std * 0.7
    validate_flaws(audio, result, sample_rate, labels, [])


def test_add_pause_inserts_and_validates_silence(speech_like_audio):
    audio, sample_rate = speech_like_audio
    insertion_sample = int(0.8 * sample_rate)
    severity = 0.5
    expected_pause_samples = round((0.4 + 1.6 * severity) * sample_rate)

    result, labels = apply_flaws(
        audio,
        sample_rate,
        [
            {
                "type": "add_pause",
                "start": 0.8,
                "end": 0.8,
                "severity": severity,
            }
        ],
    )

    assert result.shape[-1] == audio.shape[-1] + expected_pause_samples
    assert labels[0]["type"] == "long_pause"
    assert np.array_equal(
        result[insertion_sample : insertion_sample + expected_pause_samples],
        np.zeros(expected_pause_samples, dtype=result.dtype),
    )
    validate_flaws(audio, result, sample_rate, labels, [])


def test_add_pause_shifts_later_flaw_on_output_timeline(speech_like_audio):
    audio, sample_rate = speech_like_audio
    pause_length = round((0.4 + 1.6 * 0.5) * sample_rate)

    _, labels = apply_flaws(
        audio,
        sample_rate,
        [
            {"type": "add_pause", "start": 0.5, "end": 0.5, "severity": 0.5},
            {"type": "volume_drop", "start": 1.0, "end": 1.5, "severity": 0.5},
        ],
    )

    assert labels[0]["start"] == pytest.approx(0.5)
    assert labels[0]["end"] == pytest.approx(0.5 + pause_length / sample_rate)
    assert labels[1]["start"] == pytest.approx(1.0 + pause_length / sample_rate)
    assert labels[1]["params"]["original_start"] == pytest.approx(1.0)


def test_remove_pause_shrinks_long_silence(speech_like_audio):
    audio, sample_rate = speech_like_audio
    audio = audio.copy()
    silence_start = int(0.7 * sample_rate)
    silence_end = int(1.1 * sample_rate)
    audio[silence_start:silence_end] = 0

    result, labels = apply_flaws(
        audio,
        sample_rate,
        [
            {
                "type": "remove_pause",
                "start": 0.2,
                "end": 1.8,
                "severity": 0.7,
            }
        ],
    )

    original_silence = measure_silent_duration(audio, sample_rate)
    flawed_span = result[
        int(labels[0]["start"] * sample_rate) : int(labels[0]["end"] * sample_rate)
    ]
    flawed_silence = measure_silent_duration(
        flawed_span, sample_rate, float(np.max(np.abs(audio)))
    )
    assert labels[0]["type"] == "missing_pause"
    assert result.shape[-1] == audio.shape[-1] - (silence_end - silence_start) + round(
        0.05 * sample_rate
    )
    assert flawed_silence <= original_silence * 0.5
    validate_flaws(audio, result, sample_rate, labels, [])


def test_volume_drop_attenuates_span_and_validates(speech_like_audio):
    audio, sample_rate = speech_like_audio
    start_sample, end_sample = int(0.3 * sample_rate), int(1.7 * sample_rate)

    result, labels = apply_flaws(
        audio,
        sample_rate,
        [
            {
                "type": "volume_drop",
                "start": 0.3,
                "end": 1.7,
                "severity": 0.5,
            }
        ],
    )

    assert result.shape == audio.shape
    assert labels[0]["type"] == "low_volume"
    assert np.array_equal(result[:start_sample], audio[:start_sample])
    assert np.array_equal(result[end_sample:], audio[end_sample:])
    validate_flaws(audio, result, sample_rate, labels, [])


@pytest.mark.parametrize(
    ("flaw_type", "severity"),
    [("speed_up", 0.6), ("slow_down", 0.5)],
)
def test_speed_validation_uses_original_alignment(
    speech_like_audio, flaw_type, severity
):
    audio, sample_rate = speech_like_audio
    alignment = [
        {"word": "one", "start": 0.5, "end": 0.7},
        {"word": "two", "start": 0.8, "end": 1.0},
        {"word": "three", "start": 1.1, "end": 1.3},
        {"word": "four", "start": 1.4, "end": 1.6},
    ]
    result, labels = apply_flaws(
        audio,
        sample_rate,
        [{"type": flaw_type, "start": 0.5, "end": 1.5, "severity": severity}],
    )

    metrics = validate_flaws(audio, result, sample_rate, labels, alignment)

    assert metrics[0]["original_words_per_second"] == pytest.approx(4)
    if flaw_type == "speed_up":
        assert metrics[0]["flawed_words_per_second"] >= 4 * 1.08
    else:
        assert metrics[0]["flawed_words_per_second"] <= 4 * 0.92
