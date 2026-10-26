"""
Objective quality metrics.

Two kinds of metric are kept strictly apart:

* Reference-free statistics (RMS, peak, duration, ...) can be measured on any
  signal.
* Reference-based metrics (SNR, RMSE, correlation) compare a signal with a
  KNOWN clean reference ``s``. The error of an estimate ``y`` is ``e = y - s``:

      SNR(y)          = 10*log10( sum(s^2) / sum((y - s)^2) )
      input SNR       = SNR(noisy)          (error = the added noise)
      output SNR      = SNR(enhanced)       (error = residual noise + distortion)
      SNR improvement = output SNR - input SNR

  These are only meaningful when ``s`` is genuinely known (e.g. noise was
  simulated, or a separate clean reference recording was supplied). For a
  real noisy recording without a reference they are reported as unavailable.
"""

from __future__ import annotations

import numpy as np

import config


def _align(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim both signals to the shorter length (documented reconstruction tolerance)."""
    n = min(reference.size, estimate.size)
    return np.asarray(reference[:n], dtype=np.float64), np.asarray(estimate[:n], dtype=np.float64)


def snr_db(reference: np.ndarray, estimate: np.ndarray, eps: float = config.EPS) -> float:
    """SNR of ``estimate`` relative to the clean ``reference`` (error = estimate − reference)."""
    s, y = _align(reference, estimate)
    if s.size == 0:
        return float("nan")
    p_signal = float(np.sum(s ** 2))
    p_error = float(np.sum((y - s) ** 2))
    if p_signal < eps:
        return float("nan")      # SNR of a silent reference is undefined
    return float(10.0 * np.log10(p_signal / (p_error + eps)))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    s, y = _align(reference, estimate)
    return float(np.sqrt(np.mean((y - s) ** 2))) if s.size else float("nan")


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Pearson correlation; NaN when either signal has (near) zero variance."""
    s, y = _align(reference, estimate)
    if s.size < 2 or np.std(s) < config.EPS or np.std(y) < config.EPS:
        return float("nan")
    return float(np.corrcoef(s, y)[0, 1])


def segmental_snr_db(reference: np.ndarray, estimate: np.ndarray, sr: int, frame_ms: float = 20.0,
                     lo: float = -10.0, hi: float = 35.0) -> float:
    """Mean frame-wise SNR, clamped to [lo, hi] dB per frame (standard practice).

    Frames where the reference is silent are skipped because their SNR is undefined.
    """
    s, y = _align(reference, estimate)
    n = max(1, int(sr * frame_ms / 1000))
    vals = []
    for i in range(0, s.size - n + 1, n):
        seg_s, seg_e = s[i:i + n], y[i:i + n] - s[i:i + n]
        ps = np.sum(seg_s ** 2)
        if ps < n * config.SILENCE_RMS_THRESHOLD ** 2:
            continue
        vals.append(np.clip(10 * np.log10(ps / (np.sum(seg_e ** 2) + config.EPS)), lo, hi))
    return float(np.mean(vals)) if vals else float("nan")


def signal_stats(signal: np.ndarray, sr: int) -> dict:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return {"Duration (s)": 0.0, "Output RMS": 0.0, "Peak amplitude": 0.0}
    return {
        "Duration (s)": signal.size / sr,
        "Output RMS": float(np.sqrt(np.mean(signal ** 2))),
        "Peak amplitude": float(np.max(np.abs(signal))),
    }


def evaluate_output(name: str, output: np.ndarray, sr: int, processing_time_s: float | None,
                    reference: np.ndarray | None = None, noisy: np.ndarray | None = None) -> dict:
    """One row of the comparison table.

    Reference-based columns are ``None`` (shown as "N/A") when no clean reference exists.
    """
    row = {"Method": name, **signal_stats(output, sr),
           "Processing time (s)": processing_time_s}
    if reference is not None:
        in_snr = snr_db(reference, noisy) if noisy is not None else None
        out_snr = snr_db(reference, output)
        row.update({
            "RMSE vs reference": rmse(reference, output),
            "Correlation vs reference": correlation(reference, output),
            "Input SNR (dB)": in_snr,
            "Output SNR (dB)": out_snr,
            "SNR improvement (dB)": (out_snr - in_snr) if in_snr is not None else None,
            "Segmental SNR (dB)": segmental_snr_db(reference, output, sr),
        })
    else:
        row.update({k: None for k in ("RMSE vs reference", "Correlation vs reference", "Input SNR (dB)",
                                      "Output SNR (dB)", "SNR improvement (dB)", "Segmental SNR (dB)")})
    return row
