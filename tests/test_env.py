"""Environment and fundamental frequency smoke tests."""

import importlib
import importlib.util

import numpy as np
import pytest

PACKAGES = {
    "librosa": "librosa",
    "parselmouth": "parselmouth",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "soundfile": "soundfile",
    "pydub": "pydub",
    "pyyaml": "yaml",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "pytest": "pytest",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "pydantic": "pydantic",
    "openai-whisper": "whisper",
}


@pytest.mark.parametrize("distribution,module", PACKAGES.items())
def test_required_package_imports(distribution, module):
    """Every requested dependency can be imported."""
    importlib.import_module(module)


def test_optional_pyworld_imports_when_installed():
    """Import pyworld when its optional package is present."""
    if importlib.util.find_spec("pyworld") is not None:
        importlib.import_module("pyworld")


def test_librosa_pyin_detects_220_hz_tone():
    """librosa.pyin estimates a one-second 220 Hz sine within 5 Hz."""
    import librosa

    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    signal = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    f0, _, _ = librosa.pyin(
        signal,
        fmin=65,
        fmax=400,
        sr=sample_rate,
        frame_length=2048,
        hop_length=160,
    )
    median_f0 = np.nanmedian(f0)
    assert np.isfinite(median_f0)
    assert abs(median_f0 - 220) <= 5
