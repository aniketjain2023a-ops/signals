"""
Time-domain and frequency-domain analysis.

FFT   : the Discrete Fourier Transform of the whole recording shows which
        frequencies are present on average.
PSD   : Welch's method averages periodograms of overlapping segments, giving a
        smoother, statistically better estimate of power per hertz.
STFT  : the Short-Time Fourier Transform applies the FFT to short, overlapping,
        windowed frames, showing how the spectrum changes over time
        (the spectrogram). It is invertible, which is what lets spectral
        subtraction and the ML model modify magnitudes and rebuild a waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import ShortTimeFFT, get_window, welch

import config


# --------------------------------------------------------------------------
# Time-domain statistics
# --------------------------------------------------------------------------
def time_domain_stats(signal: np.ndarray, sr: int) -> dict:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return {"Duration (s)": 0.0, "Samples": 0, "Min amplitude": 0.0, "Max amplitude": 0.0,
                "Peak amplitude": 0.0, "RMS amplitude": 0.0, "RMS (dBFS)": float("-inf"),
                "Crest factor (dB)": float("nan"), "Zero-crossing rate (/s)": 0.0}
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    zc = np.count_nonzero(np.diff(np.signbit(signal))) if signal.size > 1 else 0
    return {
        "Duration (s)": signal.size / sr,
        "Samples": int(signal.size),
        "Min amplitude": float(signal.min()),
        "Max amplitude": float(signal.max()),
        "Peak amplitude": peak,
        "RMS amplitude": rms,
        "RMS (dBFS)": 20 * np.log10(rms + config.EPS),
        "Crest factor (dB)": 20 * np.log10(peak / rms) if rms > config.EPS else float("nan"),
        "Zero-crossing rate (/s)": zc * sr / signal.size,
    }


# --------------------------------------------------------------------------
# FFT
# --------------------------------------------------------------------------
def compute_fft(signal: np.ndarray, sr: int, window: str | None = "hann") -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of the whole signal.

    A window reduces spectral leakage. The magnitude is scaled so that a
    sinusoid of amplitude A shows a peak of approximately A.
    Returns ``(frequencies_hz, magnitude)``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    w = get_window(window, n, fftbins=True) if window else np.ones(n)
    spectrum = np.fft.rfft(signal * w)
    mag = np.abs(spectrum) / (np.sum(w) + config.EPS)
    mag[1:] *= 2.0                       # fold negative frequencies into the one-sided spectrum
    if n % 2 == 0:
        mag[-1] /= 2.0                   # the Nyquist bin has no negative twin
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return freqs, mag


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray, fmin: float = 50.0) -> float:
    """Frequency with the largest magnitude above ``fmin`` (skips DC / rumble)."""
    if freqs.size == 0:
        return float("nan")
    mask = freqs >= fmin
    if not np.any(mask) or np.max(mag[mask]) <= 0:
        return float("nan")
    return float(freqs[mask][np.argmax(mag[mask])])


SPEECH_BANDS_HZ = [
    ("Sub-bass (<80 Hz)", 0, 80),
    ("Fundamental / pitch (80–300 Hz)", 80, 300),
    ("Low formants (300–1000 Hz)", 300, 1000),
    ("Mid formants (1–2 kHz)", 1000, 2000),
    ("Upper formants (2–4 kHz)", 2000, 4000),
    ("Fricatives / sibilance (4–8 kHz)", 4000, 8000),
    ("Above 8 kHz", 8000, np.inf),
]


def band_energy_distribution(freqs: np.ndarray, mag: np.ndarray) -> list[dict]:
    """Percentage of spectral energy in speech-relevant frequency bands."""
    power = mag ** 2
    total = float(power.sum())
    rows = []
    f_top = freqs[-1] if freqs.size else 0.0
    for name, lo, hi in SPEECH_BANDS_HZ:
        if lo >= f_top:
            continue                      # band lies at/above Nyquist: no content possible
        sel = (freqs >= lo) & (freqs < hi)
        if hi >= f_top:
            sel |= freqs == f_top         # the band reaching Nyquist includes the Nyquist bin
        pct = 100.0 * float(power[sel].sum()) / total if total > 0 else 0.0
        rows.append({"Band": name, "Energy (%)": pct})
    return rows


def spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    total = float(mag.sum())
    return float(np.sum(freqs * mag) / total) if total > 0 else float("nan")


def compute_psd(signal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density in V²/Hz (here: full-scale²/Hz)."""
    signal = np.asarray(signal, dtype=np.float64)
    nperseg = int(min(1024, max(8, signal.size)))
    freqs, psd = welch(signal, fs=sr, window="hann", nperseg=nperseg)
    return freqs, psd


# --------------------------------------------------------------------------
# STFT / inverse STFT
# --------------------------------------------------------------------------
def make_stft(sr: int, n_fft: int = config.N_FFT, hop: int = config.HOP_LENGTH,
              window: str = config.WINDOW) -> ShortTimeFFT:
    """Create the STFT operator used everywhere in the project.

    ``scale_to='magnitude'`` makes |X| comparable to sinusoid amplitude. The
    inverse uses SciPy's canonical dual window, which gives exact
    reconstruction for an unmodified spectrum.
    """
    win = get_window(window, n_fft, fftbins=True)
    return ShortTimeFFT(win, hop=hop, fs=sr, scale_to="magnitude")


@dataclass
class STFTResult:
    stft: np.ndarray          # complex, shape (n_bins, n_frames)
    freqs: np.ndarray         # Hz
    times: np.ndarray         # s (frame centres)
    n_samples: int
    sr: int

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.stft)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.stft)


def stft(signal: np.ndarray, sr: int, n_fft: int = config.N_FFT, hop: int = config.HOP_LENGTH) -> STFTResult:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("Cannot compute the STFT of an empty signal.")
    op = make_stft(sr, n_fft, hop)
    n_samples = signal.size
    # ShortTimeFFT needs at least half a window of input; zero-pad very short
    # signals (istft() trims back to n_samples, so reconstruction stays exact).
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    X = op.stft(signal)
    return STFTResult(stft=X, freqs=op.f, times=op.t(signal.size), n_samples=n_samples, sr=sr)


def istft(X: np.ndarray, sr: int, n_samples: int, n_fft: int = config.N_FFT,
          hop: int = config.HOP_LENGTH) -> np.ndarray:
    """Inverse STFT, trimmed to exactly ``n_samples`` samples."""
    op = make_stft(sr, n_fft, hop)
    y = op.istft(X, k1=max(n_samples, n_fft))     # matches the padding applied in stft()
    return np.real(y[:n_samples]).astype(np.float64)


def spectrogram_db(signal: np.ndarray, sr: int, n_fft: int = config.N_FFT,
                   hop: int = config.HOP_LENGTH, floor_db: float = -100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude spectrogram in dBFS (0 dB = full-scale sinusoid). Returns ``(times, freqs, S_db)``."""
    res = stft(signal, sr, n_fft, hop)
    s_db = 20 * np.log10(res.magnitude + config.EPS)
    return res.times, res.freqs, np.maximum(s_db, floor_db)
