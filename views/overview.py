"""Overview page: hero, upload, workflow, key metrics and the main waveform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

import app_state as state
import config
import plots
import ui
from views.common import default_kw

FORMATS = sorted(e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS)
MIME = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg", "ogg": "audio/ogg", "m4a": "audio/mp4"}


def _resample_toggle() -> None:
    st.toggle(f"Resample to {config.PROCESSING_SAMPLE_RATE // 1000} kHz for processing", key="resample",
              persist_state="session", **default_kw("resample"),
              help="Standard rate for speech processing, and required by the ML model (trained at 16 kHz). "
                   "Every change to the signal is listed in the preprocessing log.")


def _uploader(key_prefix: str, label_visibility: str = "collapsed") -> None:
    key = f"{key_prefix}_{st.session_state.uploader_gen}"
    st.file_uploader("Audio file", type=FORMATS, key=key, on_change=state.on_upload, args=(key,),
                     label_visibility=label_visibility, help="WAV and FLAC always work. M4A needs ffmpeg.")


def upload_card(audio, error) -> None:
    with st.container(key="card-upload"):
        if not st.session_state.audio_bytes:
            ui.upload_header(config.MAX_UPLOAD_MB, FORMATS)
            _uploader("uploader")
            _resample_toggle()
            return

        name = st.session_state.audio_name
        size = ui.esc(state.human_size(len(st.session_state.audio_bytes)))
        if error:
            ui.file_card(name, [size])
            ui.notice("bad", "We couldn't read this file", ui.esc(error))
        else:
            ch = {1: "Mono", 2: "Stereo"}.get(audio.n_channels, f"{audio.n_channels} channels")
            rate = (f"{audio.original_sr / 1000:g} → {audio.sr / 1000:g} kHz" if audio.sr != audio.original_sr
                    else f"{audio.sr / 1000:g} kHz")
            ui.file_card(name, [f"{audio.duration_s:.2f} s", rate, ch, state.human_size(len(st.session_state.audio_bytes)),
                                audio.file_format])
            ext = Path(name).suffix.lower().lstrip(".")
            st.audio(st.session_state.audio_bytes, format=MIME.get(ext, "audio/wav"))
        c1, c2 = st.columns(2)
        with c1:
            with st.popover("Replace file", icon=":material/swap_horiz:", width="stretch"):
                st.caption("Choose a new recording. Current results will be cleared.")
                _uploader("replace", "collapsed")
        with c2:
            st.button("Remove", icon=":material/delete:", on_click=state.remove_upload, width="stretch", key="remove_audio")
        _resample_toggle()
        if audio is not None and audio.processing_log:
            with st.expander("Preprocessing log", icon=":material/history:"):
                for line in audio.processing_log:
                    st.markdown(f"- {line}")
                st.caption("The uploaded file itself is never modified; processing uses an in-memory copy.")


def workflow_card(audio) -> None:
    res = state.result()
    processed = audio is not None and res is not None and not state.is_stale(audio)
    with st.container(key="card-workflow"):
        ui.section("Workflow", "From raw recording to measurable results.", "flask", tone="cyan")
        ui.steps([
            ("Upload a recording", "WAV, FLAC, MP3, OGG or M4A", "done" if audio else "active"),
            ("Analyse the signal", "Waveform, FFT and spectrogram", "done" if audio else ""),
            ("Reduce the noise", "DSP filtering and ML enhancement",
             "done" if processed else ("active" if audio else "")),
            ("Export results", "WAV, CSV and HTML report", "active" if processed else ""),
        ])
        st.page_link(state.PAGES["analysis"], label="Open audio analysis", icon=":material/graphic_eq:",
                     disabled=audio is None)
        st.page_link(state.PAGES["lab"], label="Open noise reduction lab", icon=":material/tune:",
                     disabled=audio is None)


def metric_row(audio) -> None:
    if audio is None:
        ph = dict(value="—", placeholder=True, desc="Upload audio to measure")
        ui.metric_cards([
            dict(icon="clock", label="Duration", **ph),
            dict(icon="rate", label="Sampling frequency", tone="cyan", **ph),
            dict(icon="target", label="Dominant frequency", **ph),
            dict(icon="cpu", label="Processing status", tone="gray", value="Idle", placeholder=True,
                 desc="Waiting for a recording"),
        ])
        return
    an = state.analysis(audio)
    res = state.result()
    if res is None:
        status, sdesc, tone = "Ready", "Open Noise Reduction and run a method", "gray"
    elif state.is_stale(audio):
        status, sdesc, tone = "Outdated", "Settings changed since the last run", "amber"
    else:
        n = len(res.outputs)
        status, sdesc, tone = "Processed", f"{n} method{'s' if n != 1 else ''} · run #{st.session_state.run_id}", "green"
    rate_desc = (f"Resampled from {audio.original_sr:,} Hz" if audio.sr != audio.original_sr
                 else "Native sample rate")
    f_dom = an["dominant"]
    ui.metric_cards([
        dict(icon="clock", label="Duration", value=f"{audio.duration_s:.2f}", unit="s",
             desc=f"{audio.signal.size:,} samples analysed"),
        dict(icon="rate", label="Sampling frequency", tone="cyan", value=f"{audio.sr:,}", unit="Hz",
             desc=f"{rate_desc} · Nyquist {audio.sr // 2:,} Hz"),
        dict(icon="target", label="Dominant frequency", value=f"{f_dom:.1f}" if np.isfinite(f_dom) else "N/A",
             unit="Hz" if np.isfinite(f_dom) else None, desc="Largest FFT peak above 50 Hz"),
        dict(icon="cpu", label="Processing status", tone=tone, value=status, desc=sdesc),
    ])


def waveform_card(audio) -> None:
    with st.container(key="card-wave"):
        if audio is None:
            ui.section("Waveform", "The time-domain signal of your recording will appear here.", "wave")
            ui.empty_state("wave", "Waiting for audio", "Drop a recording above to see its waveform, spectrum and "
                                                        "measurements.")
            return
        ch = "mono mix" if audio.n_channels > 1 else "mono"
        ui.section("Waveform", f"Amplitude over time ({ch}, peak-normalised). Long recordings are drawn as a min/max "
                               "envelope for display; analysis always uses every sample.", "wave",
                   right_html=ui.pill(f"{audio.sr / 1000:g} kHz", "cyan") + " " + ui.pill(f"{audio.duration_s:.1f} s"))
        st.plotly_chart(plots.waveform(audio.signal, audio.sr, "Original", height=320), width="stretch",
                        config=plots.CONFIG, key="ov_wave")


def render() -> None:
    audio, error = state.current_audio()
    enh, _ = state.enhancer()
    model_pill = (ui.pill("ML model · synthetic", "warn", dot=True) if enh and enh.trained_on_synthetic
                  else ui.pill("ML model loaded", "violet", dot=True) if enh else ui.pill("ML model not trained", dot=True))
    ui.hero(ui.pill("Workspace ready", "ok", dot=True, live=True) + " " + ui.pill("DSP engine online", "cyan", dot=True)
            + " " + model_pill)
    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        upload_card(audio, error)
    with right:
        workflow_card(audio)
    st.space("small")
    metric_row(audio)
    waveform_card(audio)
