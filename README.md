# Speech Flaw Analyzer

Speech Flaw Analyzer compares a speaker's delivery with an ideal baseline of
the same text, identifies flaw regions with timestamps, and explains them.
This Day 1 setup contains the project skeleton and environment checks; analysis
logic has not been implemented.

## Project structure

```text
configs/                 Feature, threshold, and scoring rubric settings
data/                    Dataset card and local data directories
src/speechflaw/          Speech analysis package skeleton
  features/              Acoustic feature module placeholders
  dataset_builder/       Dataset tooling placeholders
api/                     API placeholders
docs/                    Documentation
eval/                    Evaluation placeholder
notebooks/               Exploratory notebooks
scripts/                 Project task runner
tests/                   Environment smoke test
```

## Setup (Windows PowerShell)

Python 3.11 and Git are required. Create and activate a virtual environment,
then install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal.
FFmpeg must also be installed and available on `PATH` for audio conversion and
Whisper. `pyworld` is optional; its installation may require a working C++
build toolchain on Windows.

Run `python scripts/tasks.py --help` to see the available task commands.
