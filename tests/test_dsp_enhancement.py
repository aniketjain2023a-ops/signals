import numpy as np
import pytest

import dsp_enhancement as dsp
import metrics
import noise_simulation as ns
from conftest import SR


def tone(f, n=SR):
    return np.sin(2 * np.pi * f * np.arange(n) / SR)


def test_lowpass_passes_low_and_attenuates_high():
    low, high = tone(300), tone(6000)
    out_low = dsp.butterworth_lowpass(low, SR, 3000, 6).output
    out_high = dsp.butterworth_lowpass(high, SR, 3000, 6).output
    mid = slice(2000, -2000)
    assert np.std(out_low[mid]) / np.std(low[mid]) == pytest.approx(1.0, abs=0.01)
    assert np.std(out_high[mid]) / np.std(high[mid]) < 1e-3


def test_lowpass_is_zero_phase():
    x = tone(200)
    r = dsp.butterworth_lowpass(x, SR, 3000, 4)
    assert r.zero_phase
    lag = np.argmax(np.correlate(r.output[1000:3000], x[1000:3000], "full")) - 1999
    assert lag == 0


def test_cutoff_minus_3db_single_pass():
    sos = dsp.design_butterworth_lowpass(2000, 4, SR)
    f, db = dsp.filter_frequency_response(sos, SR, 8192)
    assert db[np.argmin(np.abs(f - 2000))] == pytest.approx(-3.01, abs=0.05)


@pytest.mark.parametrize("cutoff,order", [(8000, 4), (9000, 4), (0, 4), (-5, 4), (1000, 0), (1000, 20)])
def test_invalid_filter_params(cutoff, order):
    with pytest.raises(dsp.FilterDesignError):
        dsp.design_butterworth_lowpass(cutoff, order, SR)


def test_short_signal_falls_back_to_causal():
    x = np.random.default_rng(0).standard_normal(20)
    r = dsp.butterworth_lowpass(x, SR, 3000, 8)
    assert not r.zero_phase and r.note
    assert r.output.size == 20 and np.all(np.isfinite(r.output))


def test_filter_empty_raises():
    with pytest.raises(ValueError):
        dsp.butterworth_lowpass(np.zeros(0), SR)


def test_silent_input_gives_silent_output():
    out = dsp.butterworth_lowpass(np.zeros(4000), SR).output
    assert np.all(out == 0)
    ss = dsp.spectral_subtraction(np.zeros(4000), SR).output
    assert np.all(np.isfinite(ss)) and np.max(np.abs(ss)) < 1e-12


def test_spectral_subtraction_improves_snr(speech):
    noisy, _ = ns.add_noise_at_snr(speech, 5, "white", seed=3)
    r = dsp.spectral_subtraction(noisy, SR)
    assert r.output.size == noisy.size
    assert np.all(np.isfinite(r.output))
    assert metrics.snr_db(speech, r.output) > metrics.snr_db(speech, noisy) + 2


def test_lowpass_improves_snr_for_white_noise(speech):
    noisy, _ = ns.add_noise_at_snr(speech, 5, "white", seed=3)
    out = dsp.butterworth_lowpass(noisy, SR, 3000, 6).output
    assert metrics.snr_db(speech, out) > metrics.snr_db(speech, noisy)


def test_spectral_subtraction_param_validation(speech):
    with pytest.raises(ValueError):
        dsp.spectral_subtraction(speech, SR, alpha=-1)
    with pytest.raises(ValueError):
        dsp.spectral_subtraction(speech, SR, beta=1.5)
