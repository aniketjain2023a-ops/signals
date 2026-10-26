"""
End-to-end processing pipeline (independent of the Streamlit UI so it can be tested).

Two experiment modes:

* ``simulate`` – controlled experiment. The uploaded (preprocessed) recording
  is the clean reference ``s``; seeded noise is added at the requested SNR to
  give ``y = s + n``. All enhancement methods process ``y`` and are scored
  against ``s``.
* ``real`` – the uploaded recording is processed as-is (it already contains
  real noise). Reference-based metrics are only computed if the user supplies
  a separate, sample-aligned clean reference; otherwise they are unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import dsp_enhancement as dsp
import metrics
import noise_simulation as ns

METHOD_LOWPASS = "Butterworth low-pass"
METHOD_SPECSUB = "Spectral subtraction"
METHOD_ML = "ML enhancement"
ALL_METHODS = (METHOD_LOWPASS, METHOD_SPECSUB, METHOD_ML)


@dataclass
class PipelineSettings:
    mode: str = "simulate"                 # "simulate" | "real"
    snr_db: float = 10.0
    noise_type: str = "white"
    seed: int = config.DEFAULT_SEED
    methods: tuple[str, ...] = (METHOD_LOWPASS,)
    cutoff_hz: float = config.DEFAULT_CUTOFF_HZ
    order: int = config.DEFAULT_FILTER_ORDER
    alpha: float = config.SPECTRAL_SUB_ALPHA
    beta: float = config.SPECTRAL_SUB_BETA

    def as_dict(self) -> dict:
        d = {
            "Experiment mode": "Simulated noise (controlled)" if self.mode == "simulate" else "Real recording (as uploaded)",
            "Methods": ", ".join(self.methods),
            "Butterworth cut-off (Hz)": self.cutoff_hz,
            "Butterworth order": self.order,
            "Spectral subtraction alpha": self.alpha,
            "Spectral subtraction beta": self.beta,
            "STFT frame / hop (samples)": f"{config.N_FFT} / {config.HOP_LENGTH}",
        }
        if self.mode == "simulate":
            d.update({"Target input SNR (dB)": self.snr_db, "Noise type": ns.NOISE_TYPES[self.noise_type],
                      "Random seed": self.seed})
        return d


@dataclass
class PipelineResult:
    settings: PipelineSettings
    sr: int
    input_name: str                        # "Noisy" (simulated) or "Noisy recording" (real)
    input_signal: np.ndarray
    reference: np.ndarray | None
    noise: np.ndarray | None
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    table: pd.DataFrame | None = None

    @property
    def has_reference(self) -> bool:
        return self.reference is not None


def run(signal: np.ndarray, sr: int, settings: PipelineSettings, enhancer=None,
        reference: np.ndarray | None = None) -> PipelineResult:
    signal = np.asarray(signal, dtype=np.float64)
    noise = None
    if settings.mode == "simulate":
        reference = signal
        noisy, noise = ns.add_noise_at_snr(signal, settings.snr_db, settings.noise_type, settings.seed)
        input_name = "Noisy"
    else:
        noisy = signal
        input_name = "Noisy recording"
        if reference is not None:
            n = min(reference.size, noisy.size)
            reference, noisy = np.asarray(reference[:n], dtype=np.float64), noisy[:n]

    res = PipelineResult(settings, sr, input_name, noisy, reference, noise)

    for method in settings.methods:
        t0 = time.perf_counter()
        try:
            if method == METHOD_LOWPASS:
                fr = dsp.butterworth_lowpass(noisy, sr, settings.cutoff_hz, settings.order)
                res.outputs[method] = fr.output
                res.details[method] = fr
            elif method == METHOD_SPECSUB:
                ssr = dsp.spectral_subtraction(noisy, sr, settings.alpha, settings.beta)
                res.outputs[method] = ssr.output
                res.details[method] = ssr
            elif method == METHOD_ML:
                if enhancer is None:
                    raise RuntimeError("No trained ML model is loaded.")
                er = enhancer.enhance(noisy, sr)
                res.outputs[method] = er.output
                res.details[method] = er
            else:
                raise ValueError(f"Unknown method '{method}'.")
            res.timings[method] = time.perf_counter() - t0
        except Exception as exc:  # keep other methods running; report the failure in the UI
            res.errors[method] = f"{type(exc).__name__}: {exc}"

    rows = [metrics.evaluate_output(input_name, noisy, sr, None, reference, noisy)]
    for method, out in res.outputs.items():
        rows.append(metrics.evaluate_output(method, out, sr, res.timings[method], reference, noisy))
    if METHOD_ML in res.details:
        res.details[METHOD_ML].stage_status["Performance evaluation"] = "done"
    res.table = pd.DataFrame(rows)
    return res
