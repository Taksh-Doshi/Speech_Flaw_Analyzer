"""Prepare and transcribe a trimmed baseline speech recording."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):"
    r"(?P<seconds>\d{1,2}(?:\.\d+)?)$|^(?P<plain>\d+(?:\.\d+)?)$"
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_timestamp(value):
    """Convert seconds, MM:SS, or HH:MM:SS to seconds."""
    match = TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Invalid timestamp {value!r}; use seconds, MM:SS, or HH:MM:SS."
        )
    if match.group("plain") is not None:
        return float(match.group("plain"))

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if minutes >= 60 and match.group("hours") is not None:
        raise ValueError(f"Invalid timestamp {value!r}: minutes must be below 60.")
    if seconds >= 60 and (match.group("hours") or minutes):
        raise ValueError(f"Invalid timestamp {value!r}: seconds must be below 60.")
    return hours * 3600 + minutes * 60 + seconds


def convert_audio(input_path, output_path, start, end):
    """Trim audio and write mono, 16 kHz, 16-bit PCM WAV."""
    start_seconds = parse_timestamp(start)
    end_seconds = parse_timestamp(end)
    duration = end_seconds - start_seconds
    if start_seconds < 0 or duration <= 0:
        raise ValueError("The start must be non-negative and the end must follow it.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install FFmpeg and try again."
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            str(start_seconds),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )
    return duration


@contextmanager
def resolve_input(input_value):
    """Yield a local input path, downloading remote audio into a temporary folder."""
    parsed = urlparse(input_value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        local_path = Path(input_value).expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"Input audio file does not exist: {local_path}")
        yield local_path
        return

    with tempfile.TemporaryDirectory(prefix="speechflaw-download-") as temp_dir:
        output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--no-progress",
                "-f",
                "bestaudio/best",
                "-o",
                output_template,
                input_value,
            ],
            check=True,
        )
        downloads = [
            path
            for path in Path(temp_dir).iterdir()
            if path.is_file() and not path.name.endswith((".part", ".json"))
        ]
        if len(downloads) != 1:
            raise RuntimeError(
                f"Expected yt-dlp to download one audio file, found {len(downloads)}."
            )
        yield downloads[0]


def markdown_cell(value):
    """Format a value for a single Markdown table cell."""
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def prepare_baseline(args):
    """Convert, transcribe, and record one baseline speech sample."""
    if ID_PATTERN.fullmatch(args.id) is None:
        raise ValueError("ID may contain only letters, digits, underscores, and hyphens.")
    start_seconds = parse_timestamp(args.start)
    end_seconds = parse_timestamp(args.end)
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("The start must be non-negative and the end must follow it.")

    raw_dir = PROJECT_ROOT / "data" / "raw" / "good"
    transcript_dir = PROJECT_ROOT / "data" / "transcripts"
    alignment_dir = PROJECT_ROOT / "data" / "alignments"
    wav_path = raw_dir / f"{args.id}.wav"
    transcript_path = transcript_dir / f"{args.id}.txt"
    alignment_path = alignment_dir / f"{args.id}_rough.json"
    dataset_card_path = PROJECT_ROOT / "data" / "DATASET_CARD.md"

    for directory in (raw_dir, transcript_dir, alignment_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with resolve_input(args.input) as input_path:
        convert_audio(input_path, wav_path, args.start, args.end)

    import soundfile as sf
    import whisper

    duration = float(sf.info(str(wav_path)).duration)
    model = whisper.load_model("small")
    result = model.transcribe(
        str(wav_path),
        language="en",
        word_timestamps=True,
        fp16=False,
        temperature=0,
    )
    transcript = result["text"].strip()
    words = [
        {
            "word": word["word"],
            "start": word["start"],
            "end": word["end"],
        }
        for segment in result["segments"]
        for word in segment.get("words", [])
    ]

    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    alignment_path.write_text(
        json.dumps(words, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    row = "| " + " | ".join(
        markdown_cell(value)
        for value in (
            args.id,
            args.source,
            args.licence,
            f"{duration:.3f}",
            args.speaker,
            "rough alignment from Whisper",
        )
    ) + " |\n"
    with dataset_card_path.open("a", encoding="utf-8", newline="") as dataset_card:
        dataset_card.write(row)

    print(f"Duration: {duration:.3f} seconds")
    print("Transcript:")
    print(transcript)
    print(f"Correct the draft transcript by hand: {transcript_path}")


def build_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Safe speech sample ID.")
    parser.add_argument("--input", required=True, help="Local audio path or HTTP(S) URL.")
    parser.add_argument("--start", required=True, help="Clip start (HH:MM:SS, MM:SS, or seconds).")
    parser.add_argument("--end", required=True, help="Clip end (HH:MM:SS, MM:SS, or seconds).")
    parser.add_argument("--source", required=True, help="Source URL for the dataset card.")
    parser.add_argument("--licence", required=True, help="Recording licence.")
    parser.add_argument("--speaker", required=True, help="Speaker name or identifier.")
    return parser


def main():
    """Run baseline preparation from the command line."""
    args = build_parser().parse_args()
    try:
        prepare_baseline(args)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
