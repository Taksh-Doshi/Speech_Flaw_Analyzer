# Day 1 log (2026-10-11)

## Goal
Set up the project and prepare the first baseline ("ideal") recording for Track C: Contrastive Speech Analytics & Temporal Flaw Grounding.

## What was done
- Planned the project: system architecture, folder structure, five-phase roadmap and gates.
- Decided to build from scratch. Not used: an earlier Whisper/Gemini judge notebook (scores whole recordings against fixed thresholds, with no baseline or timestamps) and Vapi (a platform for live voice agents, with no acoustic feature extraction and not reproducible offline).
- Set up the environment: Python virtual environment in `.venv`, pinned `requirements.txt` (librosa, praat-parselmouth, pyworld, numpy, scipy, pandas, soundfile, pydub, openai-whisper, fastapi, yt-dlp and others), ffmpeg 9.0.1. All installs succeeded, including pyworld.
- Created the project skeleton: `src/speechflaw/` modules, `configs/` (features, thresholds, rubric), `data/` folders, `tests/`, `scripts/tasks.py`. 18 environment tests passed.
- Wrote `scripts/prepare_baseline.py`: trims and converts audio to 16 kHz mono WAV with ffmpeg, transcribes with Whisper `small` (word timestamps, temperature 0), saves the draft transcript and rough word timings, and appends a row to `data/DATASET_CARD.md`. Covered by `tests/test_prepare_baseline.py`.
- Prepared the first baseline, `speech01`: the LibriVox reading of the Gettysburg Address by John Greenman (public domain), https://archive.org/details/gettysburg_johng_librivox. Trimmed from 22.3 s to remove the spoken LibriVox intro and cut the "End of..." outro. Duration 125.26 s. Transcript corrected by hand.

## Decisions
- Baselines must be public domain or openly licensed, because the flawed copies are modified versions. TED talks are avoided (their usual licence forbids derivatives).
- A calm reading is used only to test the pipeline. Real orators will be added later for the final dataset.
- Alignment from Whisper is rough, and is used only to choose flaw positions. Precise forced alignment comes in Days 3-4.

## Problems found
- The first trim included the LibriVox intro; the second included the outro. Both were fixed by locating word timestamps and adjusting `--start` and `--end`.
- `prepare_baseline.py` appends a new dataset-card row on every run, which created duplicates that were removed by hand. TODO: make it replace the row with the same `speech_id`.
- The project lives under OneDrive, which may slow syncing of `.venv` and audio files.

## Next (Day 2)
1. First injector, `speed_up`, with exact label timestamps and a validation check.
2. More injectors: monotone, pauses, volume drop.
3. Generate the L1-L5 gradient for `speech01` with label JSON.
4. Add a second baseline from a real orator.
