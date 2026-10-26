"""
Approach B – supervised machine-learning speech enhancement (baseline).

What the model learns
---------------------
For each STFT frame of NOISY speech the model predicts a per-band gain
(a "ratio mask") that, multiplied with the noisy magnitude, estimates the
CLEAN spectral magnitude:

    target_mask[band, t] = clip( |S|_band / |Y|_band , 0, 1 )
    |S_hat|[bin, t]      = interp_band_to_bin(mask_hat)[bin, t] * |Y|[bin, t]

where S is the clean STFT and Y the noisy STFT. Predicting a bounded mask is
a standard, more stable way of estimating clean magnitudes than regressing raw
magnitudes directly.

Features (identical in training and inference – both call ``build_features``)
------------------------------------------------------------------------------
1. |Y| is compressed into ``ML_N_BANDS`` mel-spaced bands (triangular filters).
2. log10 band energy of the current frame and ``ML_CONTEXT_FRAMES`` frames on
   each side. Edges are handled by repeating the first/last frame.
3. A per-band noise-floor estimate (low percentile of band energy over the
   whole utterance) so the model knows how loud the background is.
Feature size = n_bands * (2*context + 1) + n_bands.

Reconstruction
--------------
The waveform is rebuilt with the inverse STFT using the estimated clean
magnitude and the NOISY phase. Phase is not estimated, so at low SNR some
noise and "roughness" remains; this is a known limitation of magnitude-only
enhancement.

Model
-----
``sklearn.ensemble.RandomForestRegressor`` (multi-output, one output per band).
This is a genuine supervised baseline that runs on a CPU. It is far smaller
than modern deep-learning enhancers (e.g. DCCRN, Demucs, MetricGAN+), which
learn from hundreds of hours of data and model phase or raw waveforms.

The ``SpeechEnhancer`` interface (``enhance(signal, sr) -> EnhancementResult``)
is what the dashboard calls, so a pretrained neural model can be dropped in
later by implementing the same method.

Security note: joblib files are pickles and can execute code when loaded.
Only load model files that you created yourself; the app only loads from the
local ``models/`` folder and never from uploads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

import config
import signal_analysis as sa


# --------------------------------------------------------------------------
# Feature configuration and extraction (shared by train and inference)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    sr: int = config.PROCESSING_SAMPLE_RATE
    n_fft: int = config.N_FFT
    hop: int = config.HOP_LENGTH
    n_bands: int = config.ML_N_BANDS
    context: int = config.ML_CONTEXT_FRAMES
    noise_floor_percentile: float = config.ML_NOISE_FLOOR_PERCENTILE
    version: int = config.ML_FEATURE_VERSION

    @property
    def n_features(self) -> int:
        return self.n_bands * (2 * self.context + 1) + self.n_bands

    def to_dict(self) -> dict:
        return {"sr": self.sr, "n_fft": self.n_fft, "hop": self.hop, "n_bands": self.n_bands,
                "context": self.context, "noise_floor_percentile": self.noise_floor_percentile,
                "version": self.version}


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """Triangular mel filterbank ``(n_bands, n_bins)`` and the band centre frequencies (Hz)."""
    n_bins = cfg.n_fft // 2 + 1
    freqs = np.linspace(0, cfg.sr / 2, n_bins)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(cfg.sr / 2), cfg.n_bands + 2))
    fb = np.zeros((cfg.n_bands, n_bins))
    for b in range(cfg.n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        rising = (freqs - lo) / max(c - lo, 1e-9)
        falling = (hi - freqs) / max(hi - c, 1e-9)
        fb[b] = np.maximum(0.0, np.minimum(rising, falling))
        if fb[b].sum() == 0:                       # very narrow low bands: use nearest bin
            fb[b, np.argmin(np.abs(freqs - c))] = 1.0
    return fb, edges[1:-1]


def band_magnitudes(mag: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """RMS-like band magnitude: sqrt(sum_w |X|^2) per band -> ``(n_bands, n_frames)``."""
    return np.sqrt(fb @ (mag ** 2) + config.EPS)


def build_features(noisy_mag: np.ndarray, cfg: FeatureConfig, fb: np.ndarray | None = None) -> np.ndarray:
    """Build the per-frame feature matrix ``(n_frames, cfg.n_features)``.

    This single function is used by BOTH ``train_model.py`` and inference, so
    the context-window construction and boundary handling cannot diverge.
    """
    if fb is None:
        fb, _ = mel_filterbank(cfg)
    if noisy_mag.shape[0] != fb.shape[1]:
        raise ValueError(f"Magnitude has {noisy_mag.shape[0]} bins, expected {fb.shape[1]}.")
    log_bands = np.log10(band_magnitudes(noisy_mag, fb))          # (B, T)
    n_frames = log_bands.shape[1]
    c = cfg.context
    padded = np.pad(log_bands, ((0, 0), (c, c)), mode="edge")    # repeat edge frames
    context_blocks = [padded[:, k:k + n_frames] for k in range(2 * c + 1)]
    ctx = np.concatenate(context_blocks, axis=0).T                 # (T, B*(2c+1))
    floor = np.percentile(log_bands, cfg.noise_floor_percentile, axis=1)  # (B,)
    floor = np.broadcast_to(floor, (n_frames, cfg.n_bands))
    feats = np.hstack([ctx, floor]).astype(np.float32)
    assert feats.shape[1] == cfg.n_features
    return feats


def build_targets(clean_mag: np.ndarray, noisy_mag: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Band ratio-mask targets ``(n_frames, n_bands)`` in [0, 1]."""
    mask = band_magnitudes(clean_mag, fb) / band_magnitudes(noisy_mag, fb)
    return np.clip(mask, 0.0, 1.0).T.astype(np.float32)


def band_mask_to_bins(band_mask: np.ndarray, band_centres: np.ndarray, bin_freqs: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``(n_frames, n_bands)`` gains onto STFT bins -> ``(n_bins, n_frames)``."""
    out = np.empty((bin_freqs.size, band_mask.shape[0]))
    for t in range(band_mask.shape[0]):
        out[:, t] = np.interp(bin_freqs, band_centres, band_mask[t])
    return out


def apply_mask(noisy: np.ndarray, sr: int, mask_predictor, cfg: FeatureConfig,
               mask_floor: float = config.ML_MASK_FLOOR) -> tuple[np.ndarray, dict]:
    """Shared inference path: STFT -> features -> predicted mask -> |S_hat| -> iSTFT (noisy phase)."""
    fb, centres = mel_filterbank(cfg)
    res = sa.stft(noisy, sr, cfg.n_fft, cfg.hop)
    mag, phase = res.magnitude, res.phase
    X = build_features(mag, cfg, fb)
    band_mask = np.clip(np.asarray(mask_predictor(X), dtype=np.float64), mask_floor, 1.0)
    if band_mask.ndim == 1:
        band_mask = band_mask[:, None]
    bin_mask = band_mask_to_bins(band_mask, centres, res.freqs)
    est_mag = bin_mask * mag
    y = sa.istft(est_mag * np.exp(1j * phase), sr, noisy.size, cfg.n_fft, cfg.hop)
    info = {"input_feature_shape": tuple(X.shape), "output_feature_shape": tuple(band_mask.shape),
            "stft_shape": tuple(mag.shape), "estimated_magnitude": est_mag, "band_mask": band_mask}
    return y, info


# --------------------------------------------------------------------------
# Model bundle + inference wrapper
# --------------------------------------------------------------------------
class ModelUnavailableError(RuntimeError):
    """Raised when no valid trained model can be loaded."""


@dataclass
class EnhancementResult:
    output: np.ndarray
    processing_time_s: float
    input_feature_shape: tuple
    output_feature_shape: tuple
    stage_status: dict
    estimated_magnitude: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


class SpeechEnhancer:
    """Loads a trained bundle saved by ``train_model.py`` and runs inference."""

    REQUIRED_KEYS = {"model", "feature_config", "metadata"}

    def __init__(self, bundle: dict):
        missing = self.REQUIRED_KEYS - set(bundle)
        if missing:
            raise ModelUnavailableError(f"Model file is missing keys: {sorted(missing)}.")
        expected = FeatureConfig().to_dict()
        if bundle["feature_config"] != expected:
            raise ModelUnavailableError(
                "The saved model was trained with different STFT/feature settings "
                f"({bundle['feature_config']}) than the current configuration ({expected}). "
                "Retrain the model with train_model.py."
            )
        self.model = bundle["model"]
        self.cfg = FeatureConfig(**{k: v for k, v in bundle["feature_config"].items()})
        self.metadata = bundle["metadata"]
        n_in = getattr(self.model, "n_features_in_", None)
        if n_in is not None and n_in != self.cfg.n_features:
            raise ModelUnavailableError(
                f"Model expects {n_in} features but the feature builder produces {self.cfg.n_features}."
            )

    @classmethod
    def load(cls, path: str | Path = config.MODEL_PATH) -> "SpeechEnhancer":
        path = Path(path)
        if not path.exists():
            raise ModelUnavailableError(
                "No trained model found. Train one with `python train_model.py --data-dir data/clean` "
                "(real speech) or `python train_model.py --synthetic` (pipeline test only)."
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # corrupted / incompatible pickle
            raise ModelUnavailableError(f"The model file could not be loaded ({type(exc).__name__}).") from exc
        if not isinstance(bundle, dict):
            raise ModelUnavailableError("The model file has an unexpected format.")
        return cls(bundle)

    @property
    def name(self) -> str:
        return self.metadata.get("model_name", "Random Forest spectral-mask estimator")

    @property
    def model_type(self) -> str:
        return type(self.model).__name__

    @property
    def trained_on_synthetic(self) -> bool:
        return self.metadata.get("data_source") == "synthetic"

    def enhance(self, noisy: np.ndarray, sr: int) -> EnhancementResult:
        stages = {s: "pending" for s in ML_STAGES}
        warnings: list[str] = []
        t0 = time.perf_counter()
        noisy = np.asarray(noisy, dtype=np.float64)
        stages["Audio input"] = "done"
        if sr != self.cfg.sr:
            raise ValueError(f"ML model expects {self.cfg.sr} Hz audio, got {sr} Hz.")
        if noisy.size < self.cfg.n_fft:
            raise ValueError(f"Recording too short for ML enhancement (needs ≥ {self.cfg.n_fft} samples).")
        stages["Audio preprocessing"] = "done"

        def predictor(X):
            stages["STFT feature extraction"] = "done"
            stages["Noisy-feature preparation"] = "done"
            if X.shape[1] != self.cfg.n_features:
                raise ValueError("Feature dimension mismatch between training and inference.")
            out = self.model.predict(X)
            stages["Trained-model inference"] = "done"
            return out

        y, info = apply_mask(noisy, sr, predictor, self.cfg)
        stages["Estimated clean spectral magnitude"] = "done"
        stages["Waveform reconstruction"] = "done"

        if not np.all(np.isfinite(y)):
            warnings.append("Non-finite samples in ML output were replaced with zeros.")
            y = np.nan_to_num(y)
        if y.size != noisy.size:
            raise RuntimeError("Reconstructed length does not match the input length.")
        stages["Output validation"] = "done"
        elapsed = time.perf_counter() - t0
        return EnhancementResult(y, elapsed, info["input_feature_shape"], info["output_feature_shape"],
                                 stages, info["estimated_magnitude"], warnings)


ML_STAGES = [
    "Audio input",
    "Audio preprocessing",
    "STFT feature extraction",
    "Noisy-feature preparation",
    "Trained-model inference",
    "Estimated clean spectral magnitude",
    "Waveform reconstruction",
    "Output validation",
    "Performance evaluation",
]


def model_status(path: str | Path = config.MODEL_PATH) -> tuple[SpeechEnhancer | None, str]:
    """Return ``(enhancer or None, human-readable status)`` without ever raising."""
    try:
        enh = SpeechEnhancer.load(path)
    except ModelUnavailableError as exc:
        return None, str(exc)
    src = "SYNTHETIC test signals (pipeline check only)" if enh.trained_on_synthetic else "clean speech dataset"
    return enh, f"Trained model loaded ({enh.model_type}, trained on {src})."


def training_data_available(data_dir: str | Path = config.CLEAN_SPEECH_DIR) -> int:
    """Number of candidate clean-speech files in the training folder."""
    d = Path(data_dir)
    if not d.is_dir():
        return 0
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in {".wav", ".flac"})
