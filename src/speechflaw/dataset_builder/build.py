"""Build deterministic synthetic speech flaw levels from prepared baselines."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from ..seed import set_global_seed
from .injectors import apply_flaws
from .validate import measure_silent_duration, validate_flaws


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLAW_TYPES = (
    "too_fast",
    "too_slow",
    "monotone",
    "long_pause",
    "missing_pause",
    "low_volume",
)
INJECTOR_TYPES = {
    "too_fast": "speed_up",
    "too_slow": "slow_down",
    "monotone": "monotone",
    "long_pause": "add_pause",
    "missing_pause": "remove_pause",
    "low_volume": "volume_drop",
}
FLAWS_PER_LEVEL = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}
LEVEL_SEVERITY = {1: (0.3,), 2: (0.4, 0.5), 3: (0.6,), 4: (0.8,), 5: (1.0,)}


def _load_alignment(path):
    with Path(path).open(encoding="utf-8") as file:
        words = json.load(file)
    if not isinstance(words, list) or not words:
        raise ValueError(f"Alignment must contain a non-empty word list: {path}")

    normalized = []
    previous_start = -1.0
    for word in words:
        try:
            text = str(word["word"])
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid word entry in alignment {path}: {word!r}"
            ) from error
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid word interval {start}-{end} in {path}.")
        if start < previous_start:
            raise ValueError(f"Words are not ordered by start time in {path}.")
        previous_start = start
        normalized.append({"word": text, "start": start, "end": end})
    return normalized


def _candidate_spans(words):
    sentences = []
    sentence = []
    for index, word in enumerate(words):
        sentence.append(index)
        if word["word"].rstrip().endswith((".", "?", "!")):
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)

    candidates = []
    seen = set()
    for sentence_indices in sentences:
        first, last = sentence_indices[0], sentence_indices[-1]
        key = (first, last + 1)
        seen.add(key)
        candidates.append(
            {
                "start": words[first]["start"],
                "end": words[last]["end"],
                "word_count": len(sentence_indices),
                "first_word": first,
                "after_last_word": last + 1,
            }
        )
    for count in range(4, 13):
        for first in range(len(words) - count + 1):
            last = first + count - 1
            key = (first, last + 1)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "start": words[first]["start"],
                    "end": words[last]["end"],
                    "word_count": count,
                    "first_word": first,
                    "after_last_word": last + 1,
                }
            )

    pause_points = [
        {
            "start": word["end"],
            "end": word["end"],
            "word_count": 0,
            "first_word": index + 1,
            "after_last_word": index + 1,
        }
        for index, word in enumerate(words[:-1])
    ]
    return candidates, pause_points


def _overlaps(candidate, selected):
    start, end = candidate["start"], candidate["end"]
    if start == end:
        return any(
            existing["start"] <= start < existing["end"]
            or (existing["start"] == existing["end"] == start)
            for existing in selected
        )
    return any(
        (
            existing["start"] == existing["end"]
            and start <= existing["start"] < end
        )
        or (
            existing["start"] < existing["end"]
            and start < existing["end"]
            and existing["start"] < end
        )
        for existing in selected
    )


def _available_candidates(
    flaw_type, candidates, pause_points, selected, level, audio, sr
):
    choices = pause_points if flaw_type == "long_pause" else candidates
    if flaw_type == "monotone" and level == 5:
        choices = [candidate for candidate in choices if candidate["word_count"] >= 10]
    if flaw_type == "missing_pause":
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        choices = [
            candidate
            for candidate in choices
            if measure_silent_duration(
                audio[
                    round(candidate["start"] * sr) : round(candidate["end"] * sr)
                ],
                sr,
                peak,
            )
            > 0.15
        ]
    return [
        candidate
        for candidate in choices
        if not _overlaps(candidate, selected)
    ]


def _plan_level(level, candidate_pool, flaw_types, audio, sr):
    candidates, pause_points = candidate_pool
    flaw_count = FLAWS_PER_LEVEL[level]
    planned_types = flaw_types[:flaw_count]
    del flaw_types[:flaw_count]
    severities = LEVEL_SEVERITY[level]
    plans = [
        {
            "label_type": flaw_type,
            "severity": severities[0]
            if len(severities) == 1
            else random.choice(severities),
        }
        for flaw_type in planned_types
    ]

    def choose(index, selected, result):
        if index == len(plans):
            return result
        plan = plans[index]
        choices = _available_candidates(
            plan["label_type"],
            candidates,
            pause_points,
            selected,
            level,
            audio,
            sr,
        )
        random.shuffle(choices)
        for candidate in choices:
            selected.append(candidate)
            result.append(
                {
                    "type": INJECTOR_TYPES[plan["label_type"]],
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "severity": plan["severity"],
                    "_label_type": plan["label_type"],
                    "_word_count": candidate["word_count"],
                }
            )
            chosen = choose(index + 1, selected, result)
            if chosen is not None:
                return chosen
            result.pop()
            selected.pop()
        return None

    result = choose(0, [], [])
    if result is None:
        raise ValueError(
            f"Could not find {flaw_count} non-overlapping flaw spans for L{level}."
        )
    return result


def _injector_specs(plans):
    return [
        {
            key: spec[key]
            for key in ("type", "start", "end", "severity")
        }
        for spec in plans
    ]


def _build_level(audio, sr, alignment, level, plans):
    plans.sort(key=lambda plan: (plan["start"], plan["end"]))
    specs = _injector_specs(plans)
    flawed_audio, labels = apply_flaws(audio, sr, specs)
    if len(labels) != len(plans):
        raise RuntimeError(
            f"L{level}: expected {len(plans)} labels from apply_flaws, got {len(labels)}."
        )
    for plan, label in zip(plans, labels):
        label["params"]["word_count"] = plan["_word_count"]

    validation_rows = []
    failures = []
    for index, label in enumerate(labels):
        try:
            validate_flaws(audio, flawed_audio, sr, [label], alignment)
        except (AssertionError, ValueError) as error:
            validation_rows.append(
                (label["type"], label["start"], label["end"], label["severity"], False)
            )
            failures.append((index, str(error)))
        else:
            validation_rows.append(
                (label["type"], label["start"], label["end"], label["severity"], True)
            )
    return flawed_audio, labels, validation_rows, failures


def _replace_failed_span(level, plans, failed_index, candidate_pool, audio, sr):
    failed = plans[failed_index]
    unchanged = [
        {
            "start": plan["start"],
            "end": plan["end"],
        }
        for index, plan in enumerate(plans)
        if index != failed_index
    ]
    candidates, pause_points = candidate_pool
    choices = _available_candidates(
        failed["_label_type"],
        candidates,
        pause_points,
        unchanged,
        level,
        audio,
        sr,
    )
    choices = [
        candidate
        for candidate in choices
        if (candidate["start"], candidate["end"])
        != (failed["start"], failed["end"])
    ]
    random.shuffle(choices)
    if not choices:
        return False
    replacement = choices[0]
    failed["start"] = replacement["start"]
    failed["end"] = replacement["end"]
    failed["_word_count"] = replacement["word_count"]
    return True


def _build_validated_level(audio, sr, alignment, level, plans, candidate_pool):
    retries = set()
    while True:
        flawed_audio, labels, rows, failures = _build_level(
            audio, sr, alignment, level, plans
        )
        if not failures:
            return flawed_audio, labels, rows, []

        failed_index, reason = failures[0]
        if failed_index in retries or not _replace_failed_span(
            level, plans, failed_index, candidate_pool, audio, sr
        ):
            return flawed_audio, labels, rows, failures
        retries.add(failed_index)
        print(
            f"L{level} {labels[failed_index]['type']}: validation failed "
            f"({reason}); retrying once on a different span.",
            file=sys.stderr,
        )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _update_dataset_card(root, speech_ids):
    card_path = root / "data" / "DATASET_CARD.md"
    contents = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
    section_header = "## Generated dataset levels"
    base_contents = contents.split(section_header, 1)[0].rstrip()
    rows = [
        section_header,
        "",
        "| Level | Description | Files |",
        "|---|---|---:|",
        f"| L0 | Original baseline speech | {len(speech_ids)} |",
        f"| L1 | One flaw, severity 0.3 | {len(speech_ids)} |",
        f"| L2 | Two flaws, severity 0.4-0.5 | {len(speech_ids)} |",
        f"| L3 | Three flaws, severity 0.6 | {len(speech_ids)} |",
        f"| L4 | Four flaws, severity 0.8 | {len(speech_ids)} |",
        f"| L5 | Six flaws, severity 1.0 | {len(speech_ids)} |",
    ]
    section = "\n".join(rows)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(
        (base_contents + "\n\n" if base_contents else "") + section + "\n",
        encoding="utf-8",
    )


def build_dataset(root=PROJECT_ROOT):
    """Build levels L0-L5 for each baseline and return validation rows."""
    root = Path(root)
    raw_dir = root / "data" / "raw" / "good"
    alignment_dir = root / "data" / "alignments"
    processed_dir = root / "data" / "processed"
    labels_dir = root / "data" / "labels"
    source_paths = sorted(
        path for path in raw_dir.glob("*.wav") if path.is_file()
    )
    if not source_paths:
        raise FileNotFoundError(f"No baseline WAV files found in {raw_dir}.")

    set_global_seed(42)
    type_cycle = list(FLAW_TYPES)
    random.shuffle(type_cycle)
    type_offset = 0
    speech_ids = [path.stem for path in source_paths]
    all_rows = []
    all_failures = []

    for source_path in source_paths:
        speech_id = source_path.stem
        alignment_path = alignment_dir / f"{speech_id}_rough.json"
        if not alignment_path.is_file():
            raise FileNotFoundError(f"Missing rough alignment: {alignment_path}")
        audio, sr = sf.read(source_path, dtype="float32", always_2d=False)
        if sr != 16000 or audio.ndim != 1:
            raise ValueError(
                f"{source_path} must be 16 kHz mono audio; got {sr} Hz, "
                f"{audio.ndim} channel dimensions."
            )
        words = _load_alignment(alignment_path)
        candidate_pool = _candidate_spans(words)

        original_label = {
            "speech_id": speech_id,
            "level": "L0",
            "file": source_path.relative_to(root).as_posix(),
            "source_file": source_path.relative_to(root).as_posix(),
            "flaws": [],
        }
        _write_json(labels_dir / f"{speech_id}_L0.json", original_label)

        for level in range(1, 6):
            level_count = FLAWS_PER_LEVEL[level]
            level_types = [
                type_cycle[(type_offset + offset) % len(type_cycle)]
                for offset in range(level_count)
            ]
            type_offset += level_count
            plans = _plan_level(
                level, candidate_pool, level_types, audio, sr
            )
            flawed_audio, flaws, rows, failures = _build_validated_level(
                audio, sr, words, level, plans, candidate_pool
            )
            output_name = f"{speech_id}_L{level}.wav"
            output_path = processed_dir / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, flawed_audio, sr, subtype="PCM_16")
            label_payload = {
                "speech_id": speech_id,
                "level": f"L{level}",
                "file": output_path.relative_to(root).as_posix(),
                "source_file": source_path.relative_to(root).as_posix(),
                "flaws": flaws,
            }
            _write_json(labels_dir / f"{speech_id}_L{level}.json", label_payload)
            all_rows.extend((f"L{level}", *row) for row in rows)
            all_failures.extend(
                (f"L{level}", flaws[index]["type"], reason)
                for index, reason in failures
            )

    _update_dataset_card(root, speech_ids)
    print("| Level | Flaw type | Start | End | Severity | Check passed |")
    print("|---|---|---:|---:|---:|:---:|")
    for level, flaw_type, start, end, severity, passed in all_rows:
        print(
            f"| {level} | {flaw_type} | {start:.3f} | {end:.3f} | "
            f"{severity:.1f} | {'yes' if passed else 'no'} |"
        )
    return all_rows, all_failures


def main():
    """Run the deterministic dataset build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        _, failures = build_dataset()
    except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as error:
        raise SystemExit(f"Dataset build failed: {error}") from error
    if failures:
        raise SystemExit(
            f"Dataset build failed: {len(failures)} flaw validation check(s) failed."
        )


if __name__ == "__main__":
    main()
