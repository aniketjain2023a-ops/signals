import numpy as np
import pytest

import metrics
import noise_simulation as ns
from conftest import SR


@pytest.mark.parametrize("snr", [20.0, 10.0, 0.0, -5.0])
@pytest.mark.parametrize("noise_type", ["white", "pink", "brown"])
def test_noise_at_requested_snr(speech, snr, noise_type):
    noisy, noise = ns.add_noise_at_snr(speech, snr, noise_type, seed=7)
    assert np.allclose(noisy, speech + noise)
    measured = 10 * np.log10(np.mean(speech ** 2) / np.mean(noise ** 2))
    assert measured == pytest.approx(snr, abs=1e-9)
    assert metrics.snr_db(speech, noisy) == pytest.approx(snr, abs=1e-6)


def test_noise_is_reproducible(speech):
    a, _ = ns.add_noise_at_snr(speech, 10, seed=1)
    b, _ = ns.add_noise_at_snr(speech, 10, seed=1)
    c, _ = ns.add_noise_at_snr(speech, 10, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_noise_on_silence_rejected():
    with pytest.raises(ValueError):
        ns.add_noise_at_snr(np.zeros(1000), 10)


def test_snr_definitions():
    s = np.ones(100)
    y = s + 0.1  # error power 0.01, signal power 1 -> 20 dB
    assert metrics.snr_db(s, y) == pytest.approx(20.0, abs=1e-6)
    assert metrics.snr_db(s, s) > 100  # perfect -> very large but finite
    assert np.isfinite(metrics.snr_db(s, s))
    assert np.isnan(metrics.snr_db(np.zeros(10), np.ones(10)))


def test_rmse_and_correlation():
    s = np.sin(np.linspace(0, 10, 1000))
    assert metrics.rmse(s, s) == 0
    assert metrics.rmse(s, s + 0.5) == pytest.approx(0.5)
    assert metrics.correlation(s, 2 * s) == pytest.approx(1.0)
    assert metrics.correlation(s, -s) == pytest.approx(-1.0)
    assert np.isnan(metrics.correlation(s, np.zeros_like(s)))


def test_alignment_trims_to_shorter():
    s = np.ones(100)
    assert metrics.rmse(s, np.ones(90)) == 0


def test_snr_improvement_row(speech):
    noisy, _ = ns.add_noise_at_snr(speech, 5, seed=0)
    enhanced = speech + 0.5 * (noisy - speech)   # halve the noise amplitude -> +6.02 dB
    row = metrics.evaluate_output("x", enhanced, SR, 0.1, speech, noisy)
    assert row["Input SNR (dB)"] == pytest.approx(5.0, abs=1e-6)
    assert row["SNR improvement (dB)"] == pytest.approx(20 * np.log10(2), abs=1e-6)


def test_no_reference_gives_unavailable_metrics(speech):
    row = metrics.evaluate_output("x", speech, SR, 0.1)
    for k in ("RMSE vs reference", "Input SNR (dB)", "Output SNR (dB)", "SNR improvement (dB)"):
        assert row[k] is None
    assert row["Output RMS"] > 0


def test_segmental_snr_skips_silence():
    s = np.concatenate([np.zeros(SR // 2), np.sin(np.arange(SR // 2))])
    y = s + 0.01 * np.random.default_rng(0).standard_normal(s.size)
    seg = metrics.segmental_snr_db(s, y, SR)
    assert np.isfinite(seg) and seg > 20
