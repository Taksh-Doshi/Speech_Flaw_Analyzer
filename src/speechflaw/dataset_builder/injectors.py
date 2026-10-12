"""Audio flaw injectors for constructing synthetic speech samples."""

import librosa
import numpy as np
import pyworld


_SUPPORTED_TYPES = {
    "speed_up",
    "slow_down",
    "monotone",
    "add_pause",
    "remove_pause",
    "volume_drop",
}
_SILENCE_DB = -40.0
_SILENCE_FRAME_SECONDS = 0.01


def _normalize_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Each flaw spec must be a dict.")
    try:
        flaw_type = spec["type"]
        start = float(spec["start"])
        end = float(spec["end"])
        severity = float(spec["severity"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Each flaw spec must include type, start, end, and severity."
        ) from error

    if not isinstance(flaw_type, str):
        raise ValueError("Flaw type must be a string.")
    if flaw_type not in _SUPPORTED_TYPES:
        raise ValueError(f"Unsupported flaw type: {flaw_type!r}.")
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("Flaw start and end must be finite seconds.")
    if not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Flaw severity must be a finite value between 0 and 1.")
    if start < 0 or end < start:
        raise ValueError("Flaw intervals must have start >= 0 and end >= start.")
    if flaw_type == "add_pause" and end != start:
        raise ValueError("An add_pause spec must have start == end.")
    if flaw_type != "add_pause" and end <= start:
        raise ValueError(f"A {flaw_type} spec must have end > start.")

    return {
        "type": flaw_type,
        "start": start,
        "end": end,
        "severity": severity,
    }


def _crossfade(original, modified, sr):
    length = original.shape[-1]
    fade_length = min(round(0.05 * sr), (length + 1) // 2)
    if fade_length == 0:
        return modified
    alpha = np.ones(length, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, fade_length, endpoint=True)
    alpha[:fade_length] = ramp
    alpha[-fade_length:] = np.minimum(alpha[-fade_length:], ramp[::-1])
    return original * (1.0 - alpha) + modified * alpha


def _fit_length(audio, length):
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        padding = [(0, 0)] * audio.ndim
        padding[-1] = (0, length - audio.shape[-1])
        return np.pad(audio, padding)
    return audio


def _remove_silent_gaps(span, sr, clip_peak):
    if clip_peak <= 0:
        return span.copy()
    frame_size = max(1, round(_SILENCE_FRAME_SECONDS * sr))
    threshold = clip_peak * 10 ** (_SILENCE_DB / 20)
    frame_rms = []
    for frame_start in range(0, span.shape[-1], frame_size):
        frame = span[..., frame_start : frame_start + frame_size]
        frame_rms.append(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    quiet_frames = np.asarray(frame_rms) < threshold

    keep = []
    cursor = 0
    frame_index = 0
    while frame_index < len(quiet_frames):
        if not quiet_frames[frame_index]:
            frame_index += 1
            continue
        run_start = frame_index
        while frame_index < len(quiet_frames) and quiet_frames[frame_index]:
            frame_index += 1
        run_start_sample = run_start * frame_size
        run_end_sample = min(frame_index * frame_size, span.shape[-1])
        run_length = run_end_sample - run_start_sample
        if run_length <= round(0.15 * sr):
            continue

        target_length = round(0.05 * sr)
        retained_left = target_length // 2
        retained_right = target_length - retained_left
        keep.append(span[..., cursor : run_start_sample + retained_left])
        cursor = run_end_sample - retained_right

    keep.append(span[..., cursor:])
    return np.concatenate(keep, axis=-1)


def _make_monotone(span, sr, severity):
    if span.ndim != 1:
        raise ValueError("The monotone injector currently requires mono audio.")
    signal = np.asarray(span, dtype=np.float64)
    frame_period = 5.0
    f0, time_positions = pyworld.harvest(signal, sr, frame_period=frame_period)
    refined_f0 = pyworld.stonemask(signal, f0, time_positions, sr)
    voiced = refined_f0 > 0
    if not np.any(voiced):
        raise ValueError("Cannot inject monotone: no voiced F0 frames were detected.")

    log_f0 = np.log(refined_f0[voiced])
    mean_log_f0 = np.mean(log_f0)
    refined_f0[voiced] = np.exp(
        mean_log_f0 + (1.0 - severity) * (log_f0 - mean_log_f0)
    )
    spectral_envelope = pyworld.cheaptrick(
        signal, refined_f0, time_positions, sr
    )
    aperiodicity = pyworld.d4c(signal, refined_f0, time_positions, sr)
    synthesized = pyworld.synthesize(
        refined_f0,
        spectral_envelope,
        aperiodicity,
        sr,
        frame_period=frame_period,
    )
    return _fit_length(synthesized, span.shape[-1])


def apply_flaws(audio, sr, specs):
    """Apply non-overlapping flaws defined on the input timeline.

    Returns the concatenated audio and labels whose boundaries use the output
    timeline. Audio samples are arranged along the final axis.
    """
    audio = np.asarray(audio)
    if audio.ndim == 0:
        raise ValueError("Audio must contain a sample axis.")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    normalized_specs = sorted(
        (_normalize_spec(spec) for spec in specs),
        key=lambda spec: (spec["start"], spec["type"] != "add_pause"),
    )
    duration = audio.shape[-1] / sr
    nonzero_intervals = [
        spec
        for spec in normalized_specs
        if spec["type"] != "add_pause"
    ]
    for spec in normalized_specs:
        if spec["end"] > duration:
            raise ValueError(
                f"Flaw interval {spec['start']:.6f}-{spec['end']:.6f}s "
                f"exceeds audio duration {duration:.6f}s."
            )
        if spec["type"] == "add_pause" and any(
            interval["start"] < spec["start"] < interval["end"]
            for interval in nonzero_intervals
        ):
            raise ValueError(
                "An add_pause insertion point must not fall inside another "
                "flaw interval."
            )

    previous_end = 0.0
    for spec in nonzero_intervals:
        if spec["start"] < previous_end:
            raise ValueError(
                "Flaw intervals must not overlap: "
                f"an interval starts at {spec['start']:.6f}s before the "
                f"previous interval ends at {previous_end:.6f}s."
            )
        previous_end = spec["end"]

    output_parts = []
    labels = []
    source_cursor = 0
    output_cursor = 0
    clip_peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    for spec in normalized_specs:
        start_sample = round(spec["start"] * sr)
        end_sample = round(spec["end"] * sr)
        gap = audio[..., source_cursor:start_sample]
        if gap.shape[-1]:
            output_parts.append(gap)
            output_cursor += gap.shape[-1]

        severity = spec["severity"]
        original_start = start_sample / sr
        original_end = end_sample / sr
        output_start = output_cursor / sr

        if spec["type"] == "add_pause":
            inserted_length = round((0.4 + 1.6 * severity) * sr)
            output_parts.append(
                np.zeros((*audio.shape[:-1], inserted_length), dtype=audio.dtype)
            )
            label_type = "long_pause"
            params = {
                "original_start": original_start,
                "original_end": original_start,
            }
            output_cursor += inserted_length
            source_cursor = start_sample
        else:
            span = audio[..., start_sample:end_sample]
            if span.shape[-1] == 0:
                raise ValueError(
                    f"Flaw interval {spec['start']:.6f}-{spec['end']:.6f}s "
                    "contains no audio samples."
                )
            if spec["type"] == "speed_up":
                rate = 1 + 0.5 * severity
                modified_span = librosa.effects.time_stretch(span, rate=rate)
                label_type = "too_fast"
                params = {"rate": rate}
            elif spec["type"] == "slow_down":
                rate = 1 - 0.4 * severity
                modified_span = librosa.effects.time_stretch(span, rate=rate)
                label_type = "too_slow"
                params = {"rate": rate}
            elif spec["type"] == "monotone":
                modified_span = _crossfade(
                    span, _make_monotone(span, sr, severity), sr
                )
                label_type = "monotone"
                params = {}
            elif spec["type"] == "remove_pause":
                modified_span = _remove_silent_gaps(span, sr, clip_peak)
                label_type = "missing_pause"
                params = {}
            else:
                gain = 10 ** (-(4 + 14 * severity) / 20)
                gained_span = span * gain
                modified_span = _crossfade(span, gained_span, sr)
                label_type = "low_volume"
                params = {"gain_db": -(4 + 14 * severity)}

            output_parts.append(modified_span)
            output_cursor += modified_span.shape[-1]
            source_cursor = end_sample

        params["original_start"] = original_start
        params["original_end"] = original_end
        labels.append(
            {
                "type": label_type,
                "start": output_start,
                "end": output_cursor / sr,
                "severity": severity,
                "params": params,
            }
        )

    trailing_audio = audio[..., source_cursor:]
    if trailing_audio.shape[-1]:
        output_parts.append(trailing_audio)
    if not output_parts:
        return audio.copy(), labels
    return np.concatenate(output_parts, axis=-1), labels
