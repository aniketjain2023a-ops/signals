"""
Approach A – traditional DSP noise reduction (no machine learning).

1. Butterworth low-pass filter
   A Butterworth filter has a maximally flat pass-band. It removes energy
   above the cut-off frequency. Because speech intelligibility lives mostly
   below ~4 kHz, broadband noise above the cut-off is attenuated. It cannot
   remove noise that overlaps the speech band, and it also removes genuine
   high-frequency speech content (e.g. "s", "f", "sh" sounds).

   Filters are implemented as second-order sections (SOS), which are
   numerically more stable than a single high-order polynomial, and applied
   with forward-backward (zero-phase) filtering so the output is not delayed.

2. Spectral subtraction (Boll, 1979)
   The noise magnitude spectrum is estimated from the quietest frames of the
   recording and subtracted from every frame's magnitude. The noisy phase is
   kept. An over-subtraction factor (alpha) and a spectral floor (beta) trade
   residual noise against "musical noise" artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz

import config
import signal_analysis as sa


class FilterDesignError(ValueError):
    """Raised for physically impossible filter settings."""


@dataclass
class FilterResult:
    output: np.ndarray
    sos: np.ndarray
    cutoff_hz: float
    order: int
    zero_phase: bool
    note: str = ""


def validate_filter_params(cutoff_hz: float, order: int, sr: int) -> None:
    nyquist = sr / 2.0
    if not (config.MIN_FILTER_ORDER <= int(order) <= config.MAX_FILTER_ORDER):
        raise FilterDesignError(
            f"Filter order must be between {config.MIN_FILTER_ORDER} and {config.MAX_FILTER_ORDER}."
        )
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise FilterDesignError("Cut-off frequency must be a positive number of hertz.")
    if cutoff_hz >= nyquist:
        raise FilterDesignError(
            f"Cut-off {cutoff_hz:.0f} Hz must be below the Nyquist frequency ({nyquist:.0f} Hz) "
            f"for a sample rate of {sr} Hz."
        )


def design_butterworth_lowpass(cutoff_hz: float, order: int, sr: int) -> np.ndarray:
    validate_filter_params(cutoff_hz, order, sr)
    return butter(int(order), cutoff_hz, btype="lowpass", fs=sr, output="sos")


def butterworth_lowpass(signal: np.ndarray, sr: int, cutoff_hz: float = config.DEFAULT_CUTOFF_HZ,
                        order: int = config.DEFAULT_FILTER_ORDER, zero_phase: bool = True) -> FilterResult:
    """Apply a Butterworth low-pass filter.

    Zero-phase filtering (``sosfiltfilt``) needs the signal to be longer than
    its edge padding. For very short recordings we fall back to causal
    ``sosfilt`` and say so in ``note``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("Cannot filter an empty signal.")
    sos = design_butterworth_lowpass(cutoff_hz, order, sr)
    note = ""
    used_zero_phase = False
    if zero_phase:
        # Default padlen used by sosfiltfilt.
        padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if signal.size > padlen:
            out = sosfiltfilt(sos, signal)
            used_zero_phase = True
        else:
            out = sosfilt(sos, signal)
            note = (f"Signal too short for zero-phase filtering ({signal.size} ≤ {padlen} samples); "
                    "used single-pass causal filtering instead (adds a small phase delay).")
    else:
        out = sosfilt(sos, signal)
    return FilterResult(out, sos, float(cutoff_hz), int(order), used_zero_phase, note)


def filter_frequency_response(sos: np.ndarray, sr: int, n_points: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs_hz, magnitude_db)`` of a single pass of the filter.

    Note: zero-phase forward-backward filtering squares the magnitude response,
    i.e. doubles the attenuation in dB. ``plots.py`` shows both curves.
    """
    w, h = sosfreqz(sos, worN=n_points, fs=sr)
    return w, 20 * np.log10(np.abs(h) + config.EPS)


@dataclass
class SpectralSubtractionResult:
    output: np.ndarray
    noise_estimate: np.ndarray     # per-bin noise magnitude
    n_noise_frames: int
    alpha: float
    beta: float


def estimate_noise_spectrum(magnitude: np.ndarray, percentile: float = config.NOISE_ESTIMATE_PERCENTILE) -> tuple[np.ndarray, int]:
    """Average the magnitude of the lowest-energy frames (assumed speech pauses)."""
    energy = np.sum(magnitude ** 2, axis=0)
    n_frames = magnitude.shape[1]
    k = max(1, int(np.ceil(n_frames * percentile / 100.0)))
    idx = np.argsort(energy)[:k]
    return magnitude[:, idx].mean(axis=1), k


def spectral_subtraction(signal: np.ndarray, sr: int, alpha: float = config.SPECTRAL_SUB_ALPHA,
                         beta: float = config.SPECTRAL_SUB_BETA,
                         noise_percentile: float = config.NOISE_ESTIMATE_PERCENTILE) -> SpectralSubtractionResult:
    """Magnitude spectral subtraction with over-subtraction and spectral floor.

    |S_hat| = max(|Y| - alpha*|N_hat|, beta*|Y|),  phase(S_hat) = phase(Y)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if alpha < 0 or not (0 <= beta < 1):
        raise ValueError("alpha must be ≥ 0 and beta must be in [0, 1).")
    res = sa.stft(signal, sr)
    mag, phase = res.magnitude, res.phase
    noise_mag, k = estimate_noise_spectrum(mag, noise_percentile)
    clean_mag = np.maximum(mag - alpha * noise_mag[:, None], beta * mag)
    out = sa.istft(clean_mag * np.exp(1j * phase), sr, signal.size)
    return SpectralSubtractionResult(out, noise_mag, k, float(alpha), float(beta))
