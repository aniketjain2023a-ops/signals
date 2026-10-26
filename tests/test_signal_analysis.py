import numpy as np
import pytest

import config
import signal_analysis as sa
from conftest import SR


def test_fft_peak_frequency_and_amplitude():
    t = np.arange(SR) / SR
    x = 0.6 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)
    f, m = sa.compute_fft(x, SR)
    assert f.size == SR // 2 + 1
    assert np.isclose(f[-1], SR / 2)
    assert sa.dominant_frequency(f, m) == pytest.approx(1000, abs=1)
    assert m[np.argmin(np.abs(f - 1000))] == pytest.approx(0.6, rel=0.02)
    assert m[np.argmin(np.abs(f - 3000))] == pytest.approx(0.2, rel=0.02)


def test_dominant_frequency_ignores_dc():
    x = np.ones(1000) + 0.1 * np.sin(2 * np.pi * 200 * np.arange(1000) / SR)
    f, m = sa.compute_fft(x, SR, window=None)
    assert sa.dominant_frequency(f, m) == pytest.approx(200, abs=SR / 1000)


def test_band_energy_sums_to_100(speech):
    f, m = sa.compute_fft(speech, SR)
    rows = sa.band_energy_distribution(f, m)
    assert sum(r["Energy (%)"] for r in rows) == pytest.approx(100, abs=1e-6)
    assert all(not r["Band"].startswith("Above 8") for r in rows)  # no bins above Nyquist at 16 kHz


def test_psd_parseval_like(speech):
    f, p = sa.compute_psd(speech, SR)
    total = np.trapezoid(p, f)
    assert total == pytest.approx(np.mean(speech ** 2), rel=0.1)


@pytest.mark.parametrize("n", [config.N_FFT, 1000, 16_000, 16_001])
def test_stft_istft_perfect_reconstruction(n):
    x = np.random.default_rng(1).standard_normal(n)
    res = sa.stft(x, SR)
    assert res.stft.shape[0] == config.N_FFT // 2 + 1
    y = sa.istft(res.stft, SR, n)
    assert y.size == n
    assert np.max(np.abs(y - x)) < 1e-10


def test_stft_short_signal_still_works():
    x = np.random.default_rng(2).standard_normal(64)
    y = sa.istft(sa.stft(x, SR).stft, SR, 64)
    assert np.allclose(x, y, atol=1e-10)


def test_stft_empty_raises():
    with pytest.raises(ValueError):
        sa.stft(np.zeros(0), SR)


def test_spectrogram_db_is_finite_for_silence():
    t, f, s = sa.spectrogram_db(np.zeros(4000), SR)
    assert np.all(np.isfinite(s))
    assert s.shape == (f.size, t.size)


def test_time_domain_stats(speech):
    st = sa.time_domain_stats(speech, SR)
    assert st["Peak amplitude"] == pytest.approx(np.max(np.abs(speech)))
    assert st["RMS amplitude"] == pytest.approx(np.sqrt(np.mean(speech ** 2)))
    assert st["Duration (s)"] == pytest.approx(1.0)
    silent = sa.time_domain_stats(np.zeros(100), SR)
    assert np.isnan(silent["Crest factor (dB)"])
