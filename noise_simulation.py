"""
Reproducible noise simulation at a requested signal-to-noise ratio.

The noise amplitude is derived from the clean signal's power:

    SNR_dB      = 10 * log10(P_signal / P_noise)
    P_noise     = P_signal / 10^(SNR_dB / 10)
    noise       = sqrt(P_noise) * n / rms(n)    (n = unit noise sequence)

so the resulting mixture has exactly the requested SNR (up to floating-point
precision). A seeded ``numpy.random.Generator`` makes every experiment
repeatable.

Simulated stationary noise (white, pink, brown) is a controlled test signal.
It does NOT represent every real environment: babble, music, keyboard clicks,
reverberation and other non-stationary noises behave differently.
"""

from __future__ import annotations

import numpy as np

import config

NOISE_TYPES = {
    "white": "White Gaussian noise (flat spectrum)",
    "pink": "Pink noise (1/f power, more low-frequency energy)",
    "brown": "Brown noise (1/f² power, rumble-like)",
}


def signal_power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x ** 2)) if x.size else 0.0


def generate_noise(n_samples: int, noise_type: str = "white", seed: int | None = config.DEFAULT_SEED) -> np.ndarray:
    """Unit-RMS noise of the requested colour, generated from a seeded RNG."""
    if n_samples <= 0:
        return np.zeros(0)
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type '{noise_type}'. Choose from {list(NOISE_TYPES)}.")
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    if noise_type == "white":
        noise = white
    else:
        # Colour the noise by shaping its spectrum: amplitude ~ f^(-exponent/2).
        exponent = 1.0 if noise_type == "pink" else 2.0
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n_samples)
        f[0] = f[1] if f.size > 1 else 1.0          # avoid infinite gain at DC
        spec *= f ** (-exponent / 2.0)
        spec[0] = 0.0                               # zero-mean noise
        noise = np.fft.irfft(spec, n=n_samples)
    rms = np.sqrt(np.mean(noise ** 2))
    return noise / rms if rms > config.EPS else noise


def add_noise_at_snr(
    clean: np.ndarray,
    snr_db: float,
    noise_type: str = "white",
    seed: int | None = config.DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(noisy, noise)`` where ``noisy = clean + noise`` at the requested SNR.

    Raises ``ValueError`` for a silent clean signal, because the SNR of
    noise added to silence is undefined.
    """
    clean = np.asarray(clean, dtype=np.float64)
    p_signal = signal_power(clean)
    if p_signal < config.SILENCE_RMS_THRESHOLD ** 2:
        raise ValueError("The signal is silent, so a target SNR cannot be defined.")
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be a finite number of dB.")
    unit = generate_noise(clean.size, noise_type, seed)
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = unit * np.sqrt(p_noise)
    return clean + noise, noise
