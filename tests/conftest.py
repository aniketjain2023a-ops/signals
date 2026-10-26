"""Shared test fixtures: synthetic speech-like audio and in-memory WAV files."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16_000


def speech_like(duration=1.0, sr=SR, seed=0):
    """Harmonic 'vowel' with syllabic amplitude modulation and pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 15))
    env = np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None) ** 2
    return 0.5 * x * env / np.max(np.abs(x * env)) + 1e-4 * rng.standard_normal(t.size)


def wav_bytes(signal, sr=SR, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


@pytest.fixture
def speech():
    return speech_like()


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """Genuinely trains a very small model (synthetic data) so ML tests never need a committed binary."""
    import train_model as tm
    out = tmp_path_factory.mktemp("model") / "model.joblib"
    tm.train(synthetic=True, n_synthetic=6, max_frames=4000, n_estimators=5, max_depth=6,
             output=out, summary_path=out.with_name("training_summary.json"), log=lambda *_: None)
    return out
