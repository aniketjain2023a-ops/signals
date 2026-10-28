"""
Session state and data access for the multipage dashboard.

Pages never process audio themselves – they call these helpers, which wrap
the tested processing modules (audio_utils, signal_analysis, pipeline, ...).

State model
-----------
* The uploaded file is kept as raw bytes in ``st.session_state`` (file
  uploaders lose their value when the user switches pages, so we copy it).
* Settings widgets use ``persist_state="session"``, so their values survive
  page switches. ``setting()`` returns the current value or the default.
* A processing run is stored with a *signature* of the inputs that produced
  it. If the inputs change, the result is marked stale instead of silently
  showing outdated numbers.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import streamlit as st

import audio_utils as au
import config
import ml_enhancement as ml
import pipeline as pl
import signal_analysis as sa

PAGES: dict = {}          # filled by app.py with the st.Page objects (for st.page_link)

METHOD_OPTIONS = {
    "Compare all methods": None,
    "Butterworth low-pass": (pl.METHOD_LOWPASS,),
    "Spectral subtraction": (pl.METHOD_SPECSUB,),
    "ML enhancement": (pl.METHOD_ML,),
}
MODES = ["Simulated noise", "Real recording"]

DEFAULTS = {
    "resample": True,
    "mode": "Simulated noise",
    "method": "Compare all methods",
    "snr_label": "10 dB (moderate noise)",
    "noise_type": "white",
    "seed": config.DEFAULT_SEED,
    "cutoff": int(config.DEFAULT_CUTOFF_HZ),
    "order": config.DEFAULT_FILTER_ORDER,
    "alpha": config.SPECTRAL_SUB_ALPHA,
    "beta": config.SPECTRAL_SUB_BETA,
    "freq_axis": "Log",
}


def init() -> None:
    ss = st.session_state
    ss.setdefault("audio_bytes", None)
    ss.setdefault("audio_name", None)
    ss.setdefault("audio_digest", None)
    ss.setdefault("ref_bytes", None)
    ss.setdefault("ref_name", None)
    ss.setdefault("uploader_gen", 0)
    ss.setdefault("ref_uploader_gen", 0)
    ss.setdefault("result", None)
    ss.setdefault("result_sig", None)
    ss.setdefault("run_id", 0)
    ss.setdefault("chart_cache", {})


def setting(key: str):
    return st.session_state.get(key, DEFAULTS[key])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _enhancer_for(mtime: float | None):
    return ml.model_status(config.MODEL_PATH)


def model_mtime() -> float | None:
    return config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else None


def enhancer():
    """(SpeechEnhancer | None, status message). Reloads automatically when the model file changes."""
    return _enhancer_for(model_mtime())


def clear_model_cache() -> None:
    _enhancer_for.clear()


def ml_possible() -> bool:
    return enhancer()[0] is not None and setting("resample")


# --------------------------------------------------------------------------
# Upload handling
# --------------------------------------------------------------------------
def clear_results() -> None:
    ss = st.session_state
    ss.result, ss.result_sig = None, None
    ss.chart_cache = {}
    for k in ("report_html", "snr_sweep"):
        ss.pop(k, None)


def on_upload(widget_key: str) -> None:
    """Callback: copy the uploaded file into session state and reset the widget."""
    f = st.session_state.get(widget_key)
    if f is None:
        return
    data = f.getvalue()
    st.session_state.audio_bytes = data
    st.session_state.audio_name = Path(f.name).name
    st.session_state.audio_digest = hashlib.sha1(data).hexdigest()
    st.session_state.uploader_gen += 1        # new key -> empty uploader on the next run
    clear_results()


def remove_upload() -> None:
    for k in ("audio_bytes", "audio_name", "audio_digest", "ref_bytes", "ref_name"):
        st.session_state[k] = None
    clear_results()


def on_ref_upload(widget_key: str) -> None:
    f = st.session_state.get(widget_key)
    if f is None:
        return
    st.session_state.ref_bytes = f.getvalue()
    st.session_state.ref_name = Path(f.name).name
    st.session_state.ref_uploader_gen += 1


def remove_reference() -> None:
    st.session_state.ref_bytes = None
    st.session_state.ref_name = None


@st.cache_data(show_spinner=False, max_entries=6)
def _load(digest: str, _data: bytes, name: str, target_sr: int | None, normalize: bool) -> au.AudioData:
    # ``digest`` is the cache key; the leading underscore stops Streamlit hashing the raw bytes again.
    return au.load_audio(_data, name, target_sr=target_sr, normalize=normalize)


def target_sr() -> int | None:
    return config.PROCESSING_SAMPLE_RATE if setting("resample") else None


def current_audio() -> tuple[au.AudioData | None, str | None]:
    """(audio, error message). Both None when nothing has been uploaded."""
    ss = st.session_state
    if not ss.audio_bytes:
        return None, None
    try:
        return _load(ss.audio_digest, ss.audio_bytes, ss.audio_name, target_sr(), True), None
    except au.AudioLoadError as exc:
        return None, str(exc)


def reference_signal(audio: au.AudioData) -> tuple[np.ndarray | None, str | None]:
    """Optional clean reference for real-recording mode, on the same amplitude scale as ``audio``."""
    ss = st.session_state
    if setting("mode") != "Real recording" or not ss.ref_bytes:
        return None, None
    try:
        digest = hashlib.sha1(ss.ref_bytes).hexdigest()
        ref = _load(digest, ss.ref_bytes, ss.ref_name, target_sr(), False)
    except au.AudioLoadError as exc:
        return None, f"Reference could not be loaded: {exc}"
    if ref.sr != audio.sr:
        return None, "Reference sample rate differs from the recording – enable 16 kHz resampling."
    warn = None
    if abs(ref.signal.size - audio.signal.size) > 0.01 * audio.signal.size:
        warn = ("Reference and recording lengths differ by more than 1 %; both are trimmed to the shorter one. "
                "Metrics are valid only if the files are sample-aligned.")
    # Apply exactly the recording's normalisation gain so both share one amplitude scale (required for SNR).
    return ref.signal * audio.normalization_gain, warn


# --------------------------------------------------------------------------
# Analysis of the uploaded signal (cached per file + sample-rate setting)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=6)
def _analysis(key: tuple, _signal: np.ndarray, sr: int) -> dict:
    freqs, mag = sa.compute_fft(_signal, sr)
    t, f, sdb = sa.spectrogram_db(_signal, sr)
    pf, pp = sa.compute_psd(_signal, sr)
    return {
        "fft": (freqs, mag),
        "dominant": sa.dominant_frequency(freqs, mag),
        "centroid": sa.spectral_centroid(freqs, mag),
        "bands": sa.band_energy_distribution(freqs, mag),
        "psd": (pf, pp),
        "spectrogram": (t, f, sdb),
        "stats": sa.time_domain_stats(_signal, sr),
    }


def analysis(audio: au.AudioData) -> dict:
    return _analysis((st.session_state.audio_digest, audio.sr), audio.signal, audio.sr)


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------
def methods_for(method_label: str, cutoff_ok: bool) -> tuple[str, ...]:
    methods = METHOD_OPTIONS.get(method_label)
    if methods is None:   # compare all available
        methods = (pl.METHOD_LOWPASS, pl.METHOD_SPECSUB) + ((pl.METHOD_ML,) if ml_possible() else ())
    if method_label == "ML enhancement" and not ml_possible():
        methods = ()
    if not cutoff_ok:
        methods = tuple(m for m in methods if m != pl.METHOD_LOWPASS)
    return methods


def current_settings(sr: int) -> pl.PipelineSettings:
    cutoff = float(setting("cutoff"))
    return pl.PipelineSettings(
        mode="simulate" if setting("mode") == "Simulated noise" else "real",
        snr_db=config.SNR_PRESETS_DB[setting("snr_label")],
        noise_type=setting("noise_type"),
        seed=int(setting("seed")),
        methods=methods_for(setting("method"), cutoff < sr / 2),
        cutoff_hz=cutoff, order=int(setting("order")),
        alpha=float(setting("alpha")), beta=float(setting("beta")),
    )


def signature(audio: au.AudioData) -> tuple:
    ref = st.session_state.ref_bytes if setting("mode") == "Real recording" else None
    return (st.session_state.audio_digest, audio.sr, repr(current_settings(audio.sr)),
            hashlib.sha1(ref).hexdigest() if ref else None, model_mtime())


def result() -> pl.PipelineResult | None:
    return st.session_state.result


def is_stale(audio: au.AudioData) -> bool:
    return st.session_state.result is not None and st.session_state.result_sig != signature(audio)


def run_processing(audio: au.AudioData) -> tuple[bool, str]:
    """Run the pipeline with live status updates. Returns (success, message)."""
    settings = current_settings(audio.sr)
    if au.is_silent(audio.signal):
        return False, "The recording is silent, so noise simulation and enhancement are not meaningful."
    if not settings.methods:
        return False, "No processing method can run with these settings (check the cut-off and the ML model)."
    reference, _ = reference_signal(audio)
    enh = enhancer()[0] if ml_possible() else None
    t0 = time.perf_counter()
    with st.status("Processing audio…", expanded=True) as box:
        def progress(step: str) -> None:
            box.write(f"{step}…")
        try:
            res = pl.run(audio.signal, audio.sr, settings, enh, reference, progress=progress)
        except ValueError as exc:
            box.update(label="Processing failed", state="error")
            return False, str(exc)
        box.update(label=f"Processing complete · {time.perf_counter() - t0:.2f} s", state="complete", expanded=False)
    clear_results()
    st.session_state.result = res
    st.session_state.result_sig = signature(audio)
    st.session_state.run_id += 1
    n = len(res.outputs)
    msg = f"{n} method{'s' if n != 1 else ''} processed in {time.perf_counter() - t0:.2f} s"
    if res.errors:
        msg += f" · {len(res.errors)} failed"
    return True, msg


def cached_chart_data(kind: str, name: str, signal: np.ndarray, sr: int):
    """Spectrogram/FFT of a pipeline output, computed once per processing run."""
    key = (st.session_state.run_id, kind, name)
    cache = st.session_state.chart_cache
    if key not in cache:
        cache[key] = sa.spectrogram_db(signal, sr) if kind == "spec" else sa.compute_fft(signal, sr)
    return cache[key]


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return ""


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


@st.cache_data(show_spinner=False, max_entries=24)
def wav_bytes(key: tuple, _signal: np.ndarray, sr: int) -> bytes:
    """16-bit WAV, cached per (run, signal name) so reruns don't re-encode audio."""
    return au.to_wav_bytes(_signal, sr)


def wav_for(name: str, signal: np.ndarray, sr: int) -> bytes:
    return wav_bytes((st.session_state.audio_digest, st.session_state.run_id, name, sr), signal, sr)
