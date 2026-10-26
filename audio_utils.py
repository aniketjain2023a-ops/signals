"""
Audio loading, validation and preprocessing.

Design rules
------------
* The uploaded bytes are never modified or written back; all processing works
  on in-memory NumPy copies.
* Formats are decoded only through audio libraries (libsndfile, or ffmpeg for
  M4A). Uploaded files are never executed.
* Any temporary file needed for decoding is created inside a private
  ``tempfile.TemporaryDirectory`` and removed immediately afterwards.
* Every change to the signal (mono mixdown, resampling, DC removal,
  normalisation) is recorded in ``AudioData.processing_log`` so the dashboard
  can show the user exactly what happened.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config


class AudioLoadError(ValueError):
    """Raised when an upload cannot be decoded into valid audio."""


@dataclass
class AudioData:
    """Decoded audio plus metadata.

    ``original`` keeps the decoded samples with all channels at the original
    sample rate (shape ``(n_samples, n_channels)``). ``signal`` is the mono,
    possibly resampled working copy used for analysis.
    """

    filename: str
    original: np.ndarray
    original_sr: int
    signal: np.ndarray
    sr: int
    file_format: str
    subtype: str = "unknown"
    normalization_gain: float = 1.0
    processing_log: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return int(self.original.shape[1])

    @property
    def original_samples(self) -> int:
        return int(self.original.shape[0])

    @property
    def duration_s(self) -> float:
        return self.original_samples / float(self.original_sr)

    def metadata(self) -> dict:
        return {
            "Filename": self.filename,
            "Format": self.file_format,
            "Encoding": self.subtype,
            "Original sample rate (Hz)": self.original_sr,
            "Channels": self.n_channels,
            "Original sample count": self.original_samples,
            "Duration (s)": round(self.duration_s, 3),
            "Processing sample rate (Hz)": self.sr,
            "Processing sample count": int(self.signal.size),
        }


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int, str, str]:
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            sr = int(f.samplerate)
            fmt, subtype = f.format, f.subtype
            max_frames = int(config.MAX_DURATION_S * sr) + 1
            audio = f.read(frames=min(f.frames, max_frames) if f.frames > 0 else -1,
                           dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise AudioLoadError(f"The file could not be decoded as audio ({exc}).") from exc
    return audio, sr, fmt, subtype


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_with_ffmpeg(data: bytes, suffix: str) -> tuple[np.ndarray, int, str, str]:
    """Decode a compressed container (e.g. M4A/AAC) by converting it to WAV with ffmpeg.

    ffmpeg is called with an argument list (no shell) and only ever reads the
    temporary copy of the upload, so filenames supplied by the user are never
    interpreted as commands.
    """
    if not ffmpeg_available():
        raise AudioLoadError(
            f"Decoding '{suffix}' files requires ffmpeg, which is not installed. "
            "Install ffmpeg or convert the file to WAV/FLAC."
        )
    with tempfile.TemporaryDirectory(prefix="speech_upload_") as tmp:
        src = Path(tmp) / f"input{suffix}"
        dst = Path(tmp) / "decoded.wav"
        src.write_bytes(data)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
               "-t", str(config.MAX_DURATION_S), "-vn", "-acodec", "pcm_f32le", str(dst)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AudioLoadError("ffmpeg timed out while decoding the file.") from exc
        if proc.returncode != 0 or not dst.exists():
            raise AudioLoadError("ffmpeg could not decode the file; it may be corrupt or not audio.")
        audio, sr, _, _ = _decode_with_soundfile(dst.read_bytes())
    return audio, sr, suffix.lstrip(".").upper(), "AAC/ALAC (via ffmpeg)"


def decode_audio_bytes(data: bytes, filename: str) -> tuple[np.ndarray, int, str, str]:
    """Decode raw upload bytes into ``(samples[n, ch], sr, format, subtype)``."""
    if data is None or len(data) == 0:
        raise AudioLoadError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioLoadError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported file type '{suffix or '(none)'}'. Supported: "
            + ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        )
    if suffix in config.FFMPEG_EXTENSIONS:
        return _decode_with_ffmpeg(data, suffix)
    return _decode_with_soundfile(data)


# --------------------------------------------------------------------------
# Signal conditioning helpers
# --------------------------------------------------------------------------
def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average all channels into one. Accepts 1-D (already mono) or 2-D ``(n, ch)``."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        return audio.copy()
    if audio.ndim != 2:
        raise ValueError("Audio must be 1-D or 2-D (samples, channels).")
    return audio.mean(axis=1)


def resample(signal: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling with a polyphase anti-aliasing filter."""
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("Sample rates must be positive.")
    if sr_in == sr_out:
        return np.asarray(signal, dtype=np.float64).copy()
    ratio = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    return resample_poly(np.asarray(signal, dtype=np.float64), ratio.numerator, ratio.denominator)


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean so the waveform is centred on zero."""
    signal = np.asarray(signal, dtype=np.float64)
    return signal - signal.mean() if signal.size else signal.copy()


def peak_normalize(signal: np.ndarray, target: float = config.PEAK_NORMALIZE_TARGET) -> np.ndarray:
    """Scale so that max|x| == target. Silent signals are returned unchanged (no divide-by-zero)."""
    signal = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < config.EPS:
        return signal.copy()
    return signal * (target / peak)


def clipping_fraction(signal: np.ndarray, threshold: float = config.CLIPPING_THRESHOLD) -> float:
    """Fraction of samples at or above the clipping threshold."""
    signal = np.asarray(signal)
    if signal.size == 0:
        return 0.0
    return float(np.mean(np.abs(signal) >= threshold))


def is_silent(signal: np.ndarray) -> bool:
    signal = np.asarray(signal, dtype=np.float64)
    return signal.size == 0 or float(np.sqrt(np.mean(signal ** 2))) < config.SILENCE_RMS_THRESHOLD


def prepare_for_export(signal: np.ndarray, max_peak: float = 0.99) -> tuple[np.ndarray, float]:
    """Make a signal safe for 16-bit WAV export.

    Non-finite values are replaced by zero and, if the peak exceeds
    ``max_peak``, the whole signal is scaled down (never hard-clipped).
    Returns ``(signal, gain_applied)``.
    """
    out = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    gain = 1.0
    if peak > max_peak:
        gain = max_peak / peak
        out = out * gain
    return out, gain


def to_wav_bytes(signal: np.ndarray, sr: int, subtype: str = "PCM_16") -> bytes:
    """Encode a mono signal as an in-memory WAV file at the given sample rate."""
    safe, _ = prepare_for_export(signal)
    buf = io.BytesIO()
    sf.write(buf, safe.astype(np.float32), int(sr), format="WAV", subtype=subtype)
    return buf.getvalue()


# --------------------------------------------------------------------------
# High-level loader used by the dashboard and the training script
# --------------------------------------------------------------------------
def load_audio(
    data: bytes,
    filename: str,
    target_sr: int | None = config.PROCESSING_SAMPLE_RATE,
    remove_dc_offset: bool = True,
    normalize: bool = True,
) -> AudioData:
    """Decode, validate and condition an audio upload.

    Steps (each logged): decode -> validate -> mono mixdown -> optional
    resampling -> optional DC removal -> optional peak normalisation.
    """
    safe_name = Path(filename).name  # never expose client-side directory paths
    audio, sr, fmt, subtype = decode_audio_bytes(data, safe_name)
    log: list[str] = []

    if audio.size == 0 or audio.shape[0] == 0:
        raise AudioLoadError("The file decoded successfully but contains no audio samples.")
    if not np.all(np.isfinite(audio)):
        raise AudioLoadError("The audio contains invalid (NaN or infinite) sample values.")
    if audio.shape[0] < config.MIN_SAMPLES:
        raise AudioLoadError(
            f"The recording is too short ({audio.shape[0]} samples); at least "
            f"{config.MIN_SAMPLES} samples are required."
        )
    if audio.shape[0] >= int(config.MAX_DURATION_S * sr):
        log.append(f"Recording truncated to the first {config.MAX_DURATION_S:.0f} s for analysis.")

    clip = clipping_fraction(audio)
    if clip > 0.001:
        log.append(f"Warning: {clip * 100:.2f}% of samples are at full scale – the recording appears clipped.")

    mono = to_mono(audio)
    if audio.shape[1] > 1:
        log.append(f"Mixed {audio.shape[1]} channels down to mono (channel average) for analysis.")

    work_sr = sr
    if target_sr is not None and sr != target_sr:
        mono = resample(mono, sr, target_sr)
        work_sr = int(target_sr)
        log.append(f"Resampled from {sr} Hz to {target_sr} Hz (polyphase anti-aliasing filter).")

    if remove_dc_offset:
        dc = float(mono.mean())
        mono = remove_dc(mono)
        if abs(dc) > 1e-4:
            log.append(f"Removed DC offset of {dc:+.5f}.")

    gain = 1.0
    if is_silent(mono):
        log.append("Warning: the recording is silent or nearly silent; analysis results will be trivial.")
    elif normalize:
        gain = config.PEAK_NORMALIZE_TARGET / float(np.max(np.abs(mono)))
        mono = mono * gain
        log.append(f"Peak-normalised to {config.PEAK_NORMALIZE_TARGET:.2f} full scale (gain ×{gain:.3f}).")

    return AudioData(
        filename=safe_name,
        original=audio,
        original_sr=sr,
        signal=mono,
        sr=work_sr,
        file_format=fmt,
        subtype=subtype,
        normalization_gain=gain,
        processing_log=log,
    )


def load_audio_file(path: str | Path, **kwargs) -> AudioData:
    """Convenience wrapper that loads audio from disk (used by train_model.py)."""
    path = Path(path)
    return load_audio(path.read_bytes(), path.name, **kwargs)
