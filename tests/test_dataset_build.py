"""Tests for deterministic dataset generation."""

import json

import numpy as np
import soundfile as sf

from speechflaw.dataset_builder.build import build_dataset
from speechflaw.seed import set_global_seed


def _make_baseline_dataset(root):
    set_global_seed(42)
    sample_rate = 16000
    word_count = 120
    duration = 32.0
    time = np.arange(round(duration * sample_rate), dtype=np.float64) / sample_rate
    frequency = 145 + 12 * np.sin(2 * np.pi * 0.3 * time)
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    audio = 0.5 * np.sin(phase)

    words = []
    for index in range(word_count):
        sentence_index, position = divmod(index, 12)
        start = sentence_index * 3.2 + position * 0.25
        end = start + 0.2
        words.append(
            {
                "word": f" word{index}." if position == 11 else f" word{index}",
                "start": start,
                "end": end,
            }
        )
    for sentence_index in range(9):
        silence_start = round((sentence_index * 3.2 + 2.95) * sample_rate)
        silence_end = round(((sentence_index + 1) * 3.2) * sample_rate)
        audio[silence_start:silence_end] = 0

    raw_dir = root / "data" / "raw" / "good"
    alignment_dir = root / "data" / "alignments"
    raw_dir.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    sf.write(raw_dir / "sample.wav", audio.astype(np.float32), sample_rate)
    (alignment_dir / "sample_rough.json").write_text(
        json.dumps(words, indent=2) + "\n", encoding="utf-8"
    )
    (root / "data" / "DATASET_CARD.md").write_text(
        "# Dataset card\n", encoding="utf-8"
    )


def test_build_dataset_rerun_is_byte_identical(tmp_path):
    _make_baseline_dataset(tmp_path)
    build_dataset(tmp_path)

    generated_paths = sorted(
        list((tmp_path / "data" / "processed").glob("*.wav"))
        + list((tmp_path / "data" / "labels").glob("*.json"))
        + [tmp_path / "data" / "DATASET_CARD.md"]
    )
    first_run = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in generated_paths
    }

    build_dataset(tmp_path)

    second_run = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in generated_paths
    }
    assert second_run == first_run
    assert len(list((tmp_path / "data" / "processed").glob("*.wav"))) == 5
    assert len(list((tmp_path / "data" / "labels").glob("*.json"))) == 6
    level_five = json.loads(
        (tmp_path / "data" / "labels" / "sample_L5.json").read_text(
            encoding="utf-8"
        )
    )
    monotone = next(flaw for flaw in level_five["flaws"] if flaw["type"] == "monotone")
    assert monotone["params"]["word_count"] >= 10
