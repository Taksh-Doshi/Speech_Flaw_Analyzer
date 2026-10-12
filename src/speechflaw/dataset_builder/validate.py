"""Measurements and acceptance checks for injected audio flaws."""

import json
from pathlib import Path

import librosa
import numpy as np


_SILENCE_DB = -40.0
_SILENCE_FRAME_SECONDS = 0.01


def _mono(audio):
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("Audio measurements require mono audio.")
    return audio


def _aligned_words(alignment):
    if isinstance(alignment, (str, Path)):
        with Path(alignment).open(encoding="utf-8") as file:
            alignment = json.load(file)
    if isinstance(alignment, dict):
        if "words" not in alignment:
            raise ValueError("Alignment object must contain a 'words' list.")
        alignment = alignment["words"]
    if not isinstance(alignment, list):
        raise ValueError("Alignment must be a list of word timings.")
    return alignment


def _count_aligned_words(alignment, start, end):
    count = 0
    for word in alignment:
        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each aligned word needs numeric start and end.") from error
        if word_start < end and word_end > start:
            count += 1
    return count


def _span(audio, start, end, sr):
    start_sample = round(float(start) * sr)
    end_sample = round(float(end) * sr)
    if start_sample < 0 or end_sample > audio.shape[-1] or end_sample < start_sample:
        raise ValueError(
            f"Invalid audio span {start:.6f}-{end:.6f}s for {audio.shape[-1] / sr:.6f}s audio."
        )
    return audio[start_sample:end_sample]


def measure_rms_db(audio):
    """Return whole-span RMS level in dBFS."""
    audio = _mono(audio)
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms == 0:
        return float("-inf")
    return 20 * np.log10(rms)


def measure_silent_duration(audio, sr, clip_peak=None):
    """Sum 10 ms frames below -40 dB relative to the clip peak."""
    audio = _mono(audio)
    if audio.size == 0:
        return 0.0
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if clip_peak is None:
        clip_peak = float(np.max(np.abs(audio)))
    if clip_peak <= 0:
        return audio.size / sr

    frame_size = max(1, round(_SILENCE_FRAME_SECONDS * sr))
    threshold = clip_peak * 10 ** (_SILENCE_DB / 20)
    silent_samples = 0
    for start in range(0, audio.size, frame_size):
        frame = audio[start : start + frame_size]
        rms = np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
        if rms < threshold:
            silent_samples += frame.size
    return silent_samples / sr


def measure_f0_std_semitones(audio, sr):
    """Measure voiced F0 variation in semitones using librosa.pyin."""
    audio = _mono(audio)
    if audio.size == 0:
        raise ValueError("Cannot measure F0 on an empty span.")
    f0, voiced, _ = librosa.pyin(
        audio.astype(np.float32, copy=False),
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=2048,
        hop_length=160,
    )
    voiced_f0 = f0[voiced & np.isfinite(f0) & (f0 > 0)]
    if voiced_f0.size == 0:
        raise ValueError("Cannot measure F0: no voiced frames were detected.")
    return float(np.std(12 * np.log2(voiced_f0)))


def measure_words_per_second(alignment, start, end):
    """Count aligned words overlapping an original-time span per second."""
    duration = float(end) - float(start)
    if duration <= 0:
        raise ValueError("Word rate requires a positive-duration source span.")
    return _count_aligned_words(_aligned_words(alignment), start, end) / duration


def _source_bounds(label):
    params = label.get("params", {})
    try:
        return float(params["original_start"]), float(params["original_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Label {label.get('type')!r} is missing original span metadata."
        ) from error


def validate_flaws(original_audio, flawed_audio, sr, labels, alignment):
    """Measure every label and raise AssertionError when its flaw is not real.

    ``alignment`` is the rough word-alignment list (or its JSON file path).
    Labels must be returned by ``apply_flaws`` so their params retain the
    matching original-timeline boundaries.
    """
    original_audio = _mono(original_audio)
    flawed_audio = _mono(flawed_audio)
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    words = _aligned_words(alignment)
    original_peak = float(np.max(np.abs(original_audio))) if original_audio.size else 0.0
    measurements = []

    for label in labels:
        label_type = label["type"]
        source_start, source_end = _source_bounds(label)
        flawed_span = _span(flawed_audio, label["start"], label["end"], sr)
        original_span = _span(original_audio, source_start, source_end, sr)
        metrics = {"type": label_type}

        if label_type in {"too_fast", "too_slow"}:
            word_count = _count_aligned_words(words, source_start, source_end)
            if word_count <= 0:
                raise ValueError(
                    f"No aligned words overlap the {label_type} source span."
                )
            original_duration = original_span.size / sr
            flawed_duration = flawed_span.size / sr
            original_rate = word_count / original_duration
            flawed_rate = word_count / flawed_duration
            metrics.update(
                original_words_per_second=original_rate,
                flawed_words_per_second=flawed_rate,
                original_duration=original_duration,
                flawed_duration=flawed_duration,
            )
            if label_type == "too_fast":
                assert flawed_duration <= original_duration * 0.92, (
                    f"too_fast span shrank by less than 8%: "
                    f"{original_duration:.3f}s to {flawed_duration:.3f}s "
                    f"({original_rate:.3f} to {flawed_rate:.3f} words/s)."
                )
            else:
                assert flawed_duration >= original_duration * 1.08, (
                    f"too_slow span grew by less than 8%: "
                    f"{original_duration:.3f}s to {flawed_duration:.3f}s "
                    f"({original_rate:.3f} to {flawed_rate:.3f} words/s)."
                )
        elif label_type == "monotone":
            original_std = measure_f0_std_semitones(original_span, sr)
            flawed_std = measure_f0_std_semitones(flawed_span, sr)
            metrics.update(
                original_f0_std_semitones=original_std,
                flawed_f0_std_semitones=flawed_std,
            )
            if label.get("severity", 0) >= 0.6:
                assert flawed_std <= original_std * 0.7, (
                    f"monotone F0 variation fell by less than 30%: "
                    f"{original_std:.3f} to {flawed_std:.3f} semitones."
                )
        elif label_type == "long_pause":
            rms_db = measure_rms_db(flawed_span)
            duration = flawed_span.size / sr
            metrics.update(rms_db=rms_db, duration=duration)
            assert duration >= 0.4, (
                f"long_pause is only {duration:.3f}s; expected at least 0.4s."
            )
            assert rms_db < _SILENCE_DB, (
                f"long_pause RMS is {rms_db:.2f} dBFS; expected below -40 dBFS."
            )
        elif label_type == "missing_pause":
            original_silence = measure_silent_duration(
                original_span, sr, original_peak
            )
            flawed_silence = measure_silent_duration(
                flawed_span, sr, original_peak
            )
            metrics.update(
                original_silent_seconds=original_silence,
                flawed_silent_seconds=flawed_silence,
            )
            assert original_silence > 0, (
                "missing_pause source span contains no detectable silent time."
            )
            assert flawed_silence <= original_silence * 0.5, (
                f"missing_pause reduced silence from {original_silence:.3f}s "
                f"to {flawed_silence:.3f}s, less than 50%."
            )
        elif label_type == "low_volume":
            original_rms_db = measure_rms_db(original_span)
            flawed_rms_db = measure_rms_db(flawed_span)
            metrics.update(
                original_rms_db=original_rms_db,
                flawed_rms_db=flawed_rms_db,
            )
            assert flawed_rms_db <= original_rms_db - 3, (
                f"low_volume level fell by less than 3 dB: "
                f"{original_rms_db:.2f} to {flawed_rms_db:.2f} dBFS."
            )
        else:
            raise ValueError(f"Unsupported flaw label type: {label_type!r}.")

        measurements.append(metrics)

    return measurements
