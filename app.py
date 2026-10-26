"""
Streamlit dashboard – AI-Based Speech Signal Analysis and Noise Reduction.

Run with:   streamlit run app.py

The app works without an ML model: DSP processing is always available and
the ML panel explains how to train a model when none is present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import audio_utils as au
import config
import dsp_enhancement as dsp
import metrics
import ml_enhancement as ml
import noise_simulation as ns
import pipeline as pl
import plots
import report_generator as rg
import signal_analysis as sa

st.set_page_config(page_title=config.PROJECT_SHORT_NAME, page_icon="🎙️", layout="wide")

CSS = """
<style>
:root { --surface:#141c2b; --border:#263147; --muted:#a7b0c0; --text:#e6e9ef; --accent:#3987e5; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
h1 { font-size: 1.55rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
h2 { font-size: 1.2rem !important; } h3 { font-size: 1.02rem !important; }
.subtitle { color: var(--muted); font-size: 0.92rem; margin-top: -0.6rem; margin-bottom: 1rem; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.9rem; height: 100%; }
.card .label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em; }
.card .value { color: var(--text); font-size: 0.95rem; font-weight: 600; margin-top: 2px; word-break: break-word; }
.badge { display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .78rem; font-weight: 600; }
.ok { background:#12382b; color:#6fdcaf; } .warn { background:#3b2c0b; color:#f1c25a; } .off { background:#2a3142; color:#a7b0c0; }
.stage { display:flex; align-items:center; gap:.5rem; padding:.28rem 0; border-bottom:1px solid var(--border); font-size:.9rem; }
div[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def load_cached(data: bytes, name: str, target_sr: int | None, normalize: bool = True) -> au.AudioData:
    return au.load_audio(data, name, target_sr=target_sr, normalize=normalize)


@st.cache_resource(show_spinner=False)
def get_enhancer(model_mtime: float | None):
    """Reloaded automatically when the model file changes (mtime is the cache key)."""
    return ml.model_status()


def model_mtime() -> float | None:
    return config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else None


def load_training_summary() -> dict | None:
    try:
        return json.loads(config.TRAINING_SUMMARY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def card(label: str, value) -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{text}</span>'


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format numbers for display; unavailable values become 'N/A'."""
    out = df.copy().astype(object)
    for c in out.columns:
        if c == "Method":
            continue
        def fmt_cell(v, c=c):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "N/A"
            if "improvement" in c:
                return f"{v:+.2f}"
            if "(dB)" in c or "(s)" in c:
                return f"{v:.2f}" if c != "Processing time (s)" else f"{v:.3f}"
            return f"{v:.4f}" if isinstance(v, float) else v
        out[c] = [fmt_cell(v) for v in df[c]]
    return out


def wav(signal: np.ndarray, sr: int) -> bytes:
    return au.to_wav_bytes(signal, sr)


def chart(fig, key: str | None = None):
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
enhancer, model_msg = get_enhancer(model_mtime())
n_training_files = ml.training_data_available()

with st.sidebar:
    st.markdown(f"### 🎙️ {config.PROJECT_SHORT_NAME}")
    st.caption("FFT · Digital filtering · Machine learning")

    upload = st.file_uploader("Upload a speech recording", type=[e.lstrip(".") for e in sorted(config.SUPPORTED_EXTENSIONS)],
                              help="WAV and FLAC always work. MP3/OGG are decoded by libsndfile; M4A needs ffmpeg.")
    resample_on = st.toggle(f"Resample to {config.PROCESSING_SAMPLE_RATE // 1000} kHz for processing", value=True,
                            help="Speech processing standard. Required for the ML model, which was trained at 16 kHz.")

    st.markdown("**Experiment mode**")
    mode_label = st.radio("Experiment mode", ["Simulate noise (controlled experiment)", "Process recording as-is (real noise)"],
                          label_visibility="collapsed", disabled=upload is None)
    mode = "simulate" if mode_label.startswith("Simulate") else "real"

    ref_upload = None
    if mode == "real":
        ref_upload = st.file_uploader("Optional: sample-aligned clean reference",
                                      type=[e.lstrip(".") for e in sorted(config.SUPPORTED_EXTENSIONS)],
                                      help="Only if you have the exact clean source of this noisy recording. "
                                           "Enables SNR/RMSE/correlation.", disabled=upload is None)

    snr_label = st.selectbox("Noise level (target input SNR)", list(config.SNR_PRESETS_DB),
                             index=1, disabled=(upload is None or mode != "simulate"))
    noise_type = st.selectbox("Noise type", list(ns.NOISE_TYPES), format_func=lambda k: ns.NOISE_TYPES[k],
                              disabled=(upload is None or mode != "simulate"))
    seed = int(st.number_input("Random seed", 0, 2**31 - 1, config.DEFAULT_SEED, step=1,
                               disabled=(upload is None or mode != "simulate")))

    working_sr = config.PROCESSING_SAMPLE_RATE
    ml_possible = enhancer is not None and resample_on
    method_options = ["DSP filtering (Butterworth low-pass)", "Spectral subtraction"]
    if ml_possible:
        method_options.append("ML enhancement")
    method_options.append("Compare all available methods")
    method_choice = st.selectbox("Processing method", method_options, index=len(method_options) - 1,
                                 disabled=upload is None)
    if enhancer is None:
        st.caption("ML enhancement unavailable – no trained model (see the ML pipeline tab).")
    elif not resample_on:
        st.caption("ML enhancement needs 16 kHz processing – enable resampling.")

    st.markdown("**Butterworth filter**")
    # Exact Nyquist check happens after loading; slider max adapts once audio is known.
    cutoff = st.slider("Cut-off frequency (Hz)", 300, 7900, int(config.DEFAULT_CUTOFF_HZ), step=50,
                       disabled=upload is None, key="cutoff")
    order = st.slider("Filter order", config.MIN_FILTER_ORDER, config.MAX_FILTER_ORDER,
                      config.DEFAULT_FILTER_ORDER, disabled=upload is None)
    with st.expander("Spectral subtraction parameters"):
        alpha = st.slider("Over-subtraction α", 0.5, 4.0, config.SPECTRAL_SUB_ALPHA, 0.1, disabled=upload is None)
        beta = st.slider("Spectral floor β", 0.0, 0.2, config.SPECTRAL_SUB_BETA, 0.01, disabled=upload is None)
    log_axis = st.radio("Frequency axis", ["Logarithmic", "Linear"], horizontal=True) == "Logarithmic"

    run_clicked = st.button("▶ Process audio", type="primary", width="stretch", disabled=upload is None)
    download_slot = st.container()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Speech Signal Analysis & Noise Reduction")
st.markdown(f'<div class="subtitle">{config.PROJECT_TITLE}</div>', unsafe_allow_html=True)

if upload is None:
    st.info("Upload a speech recording in the sidebar to begin. Short (3–20 s) WAV files of a single speaker work best.")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Approach A – DSP", "Butterworth low-pass · spectral subtraction"), unsafe_allow_html=True)
    c2.markdown(card("Approach B – ML", "Random Forest spectral-mask estimator"), unsafe_allow_html=True)
    c3.markdown(card("ML model status", badge("Loaded", "ok") if enhancer else badge("Not trained", "off")),
                unsafe_allow_html=True)
    st.caption(model_msg)
    st.stop()

# --------------------------------------------------------------------------
# Load and validate audio
# --------------------------------------------------------------------------
raw = upload.getvalue()
target_sr = config.PROCESSING_SAMPLE_RATE if resample_on else None
try:
    audio = load_cached(raw, upload.name, target_sr)
except au.AudioLoadError as exc:
    st.error(f"Could not load **{Path(upload.name).name}**: {exc}")
    st.stop()

sr = audio.sr
signal = audio.signal
nyquist = sr / 2

reference = None
if mode == "real" and ref_upload is not None:
    try:
        # Load without normalisation, then apply exactly the noisy file's gain so the
        # two signals stay on the same amplitude scale (required for valid SNR).
        ref_audio = load_cached(ref_upload.getvalue(), ref_upload.name, target_sr, normalize=False)
        if ref_audio.sr != sr:
            st.sidebar.error("Reference sample rate differs from the recording; enable resampling.")
        else:
            reference = ref_audio.signal * audio.normalization_gain
            if abs(reference.size - signal.size) > 0.01 * signal.size:
                st.sidebar.warning("Reference and recording lengths differ by >1 %; both are trimmed to the shorter one. "
                                   "Metrics are only valid if the files are sample-aligned.")
    except au.AudioLoadError as exc:
        st.sidebar.error(f"Reference could not be loaded: {exc}")

cutoff_valid = cutoff < nyquist
if not cutoff_valid:
    st.sidebar.error(f"Cut-off {cutoff} Hz is not below the Nyquist frequency ({nyquist:.0f} Hz).")

if method_choice.startswith("DSP"):
    methods = (pl.METHOD_LOWPASS,)
elif method_choice.startswith("Spectral"):
    methods = (pl.METHOD_SPECSUB,)
elif method_choice.startswith("ML"):
    methods = (pl.METHOD_ML,)
else:
    methods = (pl.METHOD_LOWPASS, pl.METHOD_SPECSUB) + ((pl.METHOD_ML,) if ml_possible else ())
if not cutoff_valid:
    methods = tuple(m for m in methods if m != pl.METHOD_LOWPASS)

settings = pl.PipelineSettings(mode=mode, snr_db=config.SNR_PRESETS_DB[snr_label], noise_type=noise_type, seed=seed,
                               methods=methods, cutoff_hz=float(cutoff), order=int(order), alpha=float(alpha),
                               beta=float(beta))
ref_digest = hashlib.sha1(reference.tobytes()).hexdigest() if reference is not None else None
signature = (hashlib.sha1(raw).hexdigest(), resample_on, repr(settings), ref_digest, model_mtime())

if run_clicked:
    if au.is_silent(signal):
        st.error("The recording is silent, so noise simulation and enhancement are not meaningful.")
    elif not methods:
        st.error("No processing method can run with the current settings.")
    else:
        with st.spinner("Processing…"):
            try:
                st.session_state.pop("report", None)          # never offer a report from an earlier run
                st.session_state["result"] = pl.run(signal, sr, settings, enhancer if ml_possible else None, reference)
                st.session_state["signature"] = signature
            except ValueError as exc:
                st.error(f"Processing failed: {exc}")

result: pl.PipelineResult | None = st.session_state.get("result")
if result is not None and st.session_state.get("signature", (None,))[0] != signature[0]:
    result = None                                # a different file was uploaded
    st.session_state.pop("result", None)
    st.session_state.pop("report", None)
stale = result is not None and st.session_state.get("signature") != signature

# --------------------------------------------------------------------------
# Section A – Overview
# --------------------------------------------------------------------------
st.subheader("A · Overview")
cols = st.columns([1.5, 0.9, 1.2, 1, 1.2, 1.1, 1.3])
status = ("Processed" if result is not None and not stale else
          "Settings changed – press Process" if stale else "Ready – press Process")
items = [
    ("File", audio.filename), ("Duration", f"{audio.duration_s:.2f} s"),
    ("Sample rate", f"{audio.original_sr} Hz" + (f" → {sr} Hz" if sr != audio.original_sr else "")),
    ("Channels", f"{audio.n_channels} ({'stereo' if audio.n_channels == 2 else 'mono' if audio.n_channels == 1 else 'multi'})"),
    ("Method", {"DSP filtering": "Butterworth low-pass", "Compare all available methods": "Compare all"}
     .get(method_choice.split(" (")[0], method_choice.split(" (")[0])),
    ("ML model", badge("Loaded", "ok") + (" " + badge("synthetic", "warn") if enhancer and enhancer.trained_on_synthetic else "")
     if enhancer else badge("Not available", "off")),
    ("Status", status),
]
for c, (k, v) in zip(cols, items):
    c.markdown(card(k, v), unsafe_allow_html=True)
if audio.processing_log:
    with st.expander("Preprocessing log (what was changed and why)", expanded=False):
        for line in audio.processing_log:
            st.markdown(f"- {line}")
        st.caption("The uploaded file itself is never modified; all processing happens on an in-memory copy.")

tabs = st.tabs(["B · Original audio", "C · Frequency analysis", "D · Noise simulation", "E · DSP enhancement",
                "F · ML pipeline", "G · Results comparison", "H · Download & export", "Applications"])

# --------------------------------------------------------------------------
# Section B – Original audio
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Original recording")
    ext = Path(audio.filename).suffix.lower().lstrip(".")
    st.audio(raw, format={"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                          "m4a": "audio/mp4"}.get(ext, "audio/wav"))
    st.caption("Above: the untouched upload. Below: analysis of the mono working copy "
               f"({sr} Hz{', peak-normalised' if audio.normalization_gain != 1 else ''}).")
    stats = sa.time_domain_stats(signal, sr)
    m = st.columns(5)
    m[0].metric("Amplitude range", f"{stats['Min amplitude']:.3f} … {stats['Max amplitude']:.3f}")
    m[1].metric("RMS amplitude", f"{stats['RMS amplitude']:.4f}", f"{stats['RMS (dBFS)']:.1f} dBFS", delta_color="off")
    m[2].metric("Peak amplitude", f"{stats['Peak amplitude']:.4f}")
    m[3].metric("Crest factor", f"{stats['Crest factor (dB)']:.1f} dB" if np.isfinite(stats['Crest factor (dB)']) else "N/A")
    m[4].metric("Samples", f"{stats['Samples']:,}")
    chart(plots.waveform(signal, sr, "Original", "Time-domain waveform"), "wave_orig")
    st.dataframe(pd.DataFrame([audio.metadata()]).T.rename(columns={0: "Value"}).astype(str), width="stretch")

# --------------------------------------------------------------------------
# Section C – Frequency analysis
# --------------------------------------------------------------------------
with tabs[1]:
    freqs, mag = sa.compute_fft(signal, sr)
    f_dom = sa.dominant_frequency(freqs, mag)
    centroid = sa.spectral_centroid(freqs, mag)
    m = st.columns(4)
    m[0].metric("Dominant frequency", f"{f_dom:.1f} Hz" if np.isfinite(f_dom) else "N/A")
    m[1].metric("Spectral centroid", f"{centroid:.0f} Hz" if np.isfinite(centroid) else "N/A")
    m[2].metric("Frequency resolution", f"{sr / signal.size:.3f} Hz")
    m[3].metric("Nyquist frequency", f"{nyquist:.0f} Hz")
    chart(plots.spectra({"Original": (freqs, mag)}, log_x=log_axis, title="FFT magnitude spectrum (Hann window)",
                        marker_hz=f_dom), "fft_orig")
    st.caption("Dominant frequency = largest FFT peak above 50 Hz. For voiced speech it is often the pitch (F0) "
               "or a low harmonic/formant, not necessarily the perceived pitch.")
    c1, c2 = st.columns(2)
    with c1:
        pf, pp = sa.compute_psd(signal, sr)
        chart(plots.psd(pf, pp, log_x=log_axis, title="Power spectral density (Welch)"), "psd_orig")
    with c2:
        chart(plots.band_energy(sa.band_energy_distribution(freqs, mag)), "bands_orig")
    t, f, sdb = sa.spectrogram_db(signal, sr)
    chart(plots.spectrogram(t, f, sdb, f"Spectrogram (STFT, {config.N_FFT}-sample Hann window, "
                                       f"{config.HOP_LENGTH}-sample hop)"), "spec_orig")


def need_result() -> bool:
    if result is None:
        st.info("Press **▶ Process audio** in the sidebar to run this section.")
        return False
    if stale:
        st.warning("Settings changed since the last run – results below are for the previous settings. "
                   "Press **▶ Process audio** to update.")
    return True


# --------------------------------------------------------------------------
# Section D – Noise simulation
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Gaussian noise is generated from a seeded random generator and scaled from the **clean signal "
                "power** so the mixture has exactly the requested SNR: "
                r"$P_{noise} = P_{signal} / 10^{SNR/10}$.")
    st.caption("Simulated stationary noise is a controlled test condition. It does not represent every real "
               "environment (babble, music, traffic, reverberation, keyboard clicks behave differently).")
    if mode == "real":
        st.info("Experiment mode is **real recording**: no noise is added. The uploaded recording is treated as "
                "the noisy input. Switch to *Simulate noise* to run controlled experiments.")
    elif need_result():
        r = result
        in_snr = metrics.snr_db(r.reference, r.input_signal)
        m = st.columns(4)
        m[0].metric("Target SNR", f"{r.settings.snr_db:.1f} dB")
        m[1].metric("Measured input SNR", f"{in_snr:.2f} dB")
        m[2].metric("Noise RMS", f"{np.sqrt(np.mean(r.noise ** 2)):.4f}")
        m[3].metric("Noisy peak", f"{np.max(np.abs(r.input_signal)):.3f}")
        st.audio(wav(r.input_signal, sr), format="audio/wav")
        if np.max(np.abs(r.input_signal)) > 0.99:
            st.caption("The noisy mixture exceeds full scale; the audio player/export scales it down to avoid "
                       "clipping. Metrics use the unscaled signal.")
        chart(plots.stacked_waveforms({"Clean reference": r.reference, "Noisy": r.input_signal}, sr), "wave_noisy")
        t, f, sdb = sa.spectrogram_db(r.input_signal, sr)
        chart(plots.spectrogram(t, f, sdb, "Noisy spectrogram"), "spec_noisy")

    if mode == "simulate":
        with st.expander("Experiment 2 – compare 20 dB, 10 dB and 0 dB SNR", expanded=False):
            if st.button("Generate comparison", key="snr_cmp"):
                sigs, rows, specs = {"Clean reference": signal}, [], {}
                for lvl in (20.0, 10.0, 0.0):
                    noisy_l, noise_l = ns.add_noise_at_snr(signal, lvl, noise_type, seed)
                    sigs[f"{lvl:g} dB SNR"] = noisy_l
                    specs[f"{lvl:g} dB SNR"] = sa.compute_fft(noisy_l, sr)
                    rows.append({"Target SNR (dB)": lvl, "Measured SNR (dB)": round(metrics.snr_db(signal, noisy_l), 3),
                                 "Noise RMS": round(float(np.sqrt(np.mean(noise_l ** 2))), 5),
                                 "Correlation with clean": round(metrics.correlation(signal, noisy_l), 4)})
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                plots.COLORS.update({"20 dB SNR": "#199e70", "10 dB SNR": "#c98500", "0 dB SNR": "#d95926"})
                chart(plots.stacked_waveforms(sigs, sr), "wave_cmp")
                specs = {"Clean reference": sa.compute_fft(signal, sr), **specs}
                chart(plots.spectra(specs, log_x=log_axis, title="Spectra at different noise levels"), "fft_cmp")

# --------------------------------------------------------------------------
# Section E – DSP enhancement
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("**Approach A – traditional DSP.** No learning is involved: the filter and the spectral-subtraction "
                "rule are fixed mathematical operations.")
    st.caption("A low-pass filter removes noise *above* the cut-off but also removes genuine high-frequency speech "
               "(e.g. ‘s’, ‘f’, ‘sh’). It cannot remove noise inside the speech band and does not eliminate every "
               "type of noise.")
    if cutoff_valid:
        sos = dsp.design_butterworth_lowpass(cutoff, order, sr)
        fz, hdb = dsp.filter_frequency_response(sos, sr)
        chart(plots.filter_response(fz, hdb, cutoff, True, log_axis), "filt_resp")
    if need_result():
        r = result
        for name in (pl.METHOD_LOWPASS, pl.METHOD_SPECSUB):
            if name in r.errors:
                st.error(f"{name} failed: {r.errors[name]}")
        if pl.METHOD_LOWPASS in r.outputs:
            fr: dsp.FilterResult = r.details[pl.METHOD_LOWPASS]
            st.markdown("#### Butterworth low-pass output")
            st.dataframe(pd.DataFrame([{"Type": "Butterworth low-pass (SOS)", "Order": fr.order,
                                        "Cut-off (Hz)": fr.cutoff_hz, "Sample rate (Hz)": sr,
                                        "Zero-phase (filtfilt)": fr.zero_phase,
                                        "Effective order (forward-backward)": 2 * fr.order if fr.zero_phase else fr.order}]),
                         hide_index=True, width="stretch")
            if fr.note:
                st.warning(fr.note)
            st.audio(wav(fr.output, sr), format="audio/wav")
            chart(plots.waveform(fr.output, sr, pl.METHOD_LOWPASS, "Filtered waveform"), "wave_lp")
            chart(plots.spectra({r.input_name: sa.compute_fft(r.input_signal, sr),
                                 pl.METHOD_LOWPASS: sa.compute_fft(fr.output, sr)}, log_x=log_axis,
                                title="Spectrum before and after filtering", marker_hz=fr.cutoff_hz), "fft_lp")
        if pl.METHOD_SPECSUB in r.outputs:
            ss: dsp.SpectralSubtractionResult = r.details[pl.METHOD_SPECSUB]
            st.markdown("#### Spectral subtraction output")
            st.caption(f"Noise spectrum estimated from the quietest {ss.n_noise_frames} STFT frames "
                       f"({config.NOISE_ESTIMATE_PERCENTILE} %). |Ŝ| = max(|Y| − α|N̂|, β|Y|), α = {ss.alpha}, "
                       f"β = {ss.beta}; the noisy phase is reused.")
            st.audio(wav(ss.output, sr), format="audio/wav")
            chart(plots.waveform(ss.output, sr, pl.METHOD_SPECSUB, "Spectral-subtraction waveform"), "wave_ss")
            t, f, sdb = sa.spectrogram_db(ss.output, sr)
            chart(plots.spectrogram(t, f, sdb, "Spectral-subtraction spectrogram"), "spec_ss")

# --------------------------------------------------------------------------
# Section F – ML pipeline
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("**Approach B – supervised machine learning.** A Random Forest regressor is trained (by "
                "`train_model.py`) on pairs of (noisy STFT band features → clean/noisy band ratio mask). At "
                "inference it predicts the mask, the "
                "mask times the noisy magnitude is the *estimated clean spectral magnitude*, and the waveform is "
                "rebuilt with the inverse STFT using the noisy phase.")
    summary = load_training_summary() if enhancer else None
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Model name", enhancer.name if enhancer else "—"), unsafe_allow_html=True)
    c2.markdown(card("Model type", enhancer.model_type if enhancer else "—"), unsafe_allow_html=True)
    c3.markdown(card("Trained model loaded", badge("Yes", "ok") if enhancer else badge("No", "off")), unsafe_allow_html=True)
    c4.markdown(card("Training data (data/clean)", f"{n_training_files} file(s)" if n_training_files else badge("None", "off")),
                unsafe_allow_html=True)
    st.caption(model_msg)
    if enhancer and enhancer.trained_on_synthetic:
        st.warning("This model was trained on **synthetic speech-like signals**. It demonstrates that the ML "
                   "pipeline works end-to-end, but it does **not** establish real-world speech-enhancement "
                   "performance. Train on real clean speech for meaningful results.")

    if result is not None and pl.METHOD_ML in result.details:
        er: ml.EnhancementResult = result.details[pl.METHOD_ML]
        st.markdown("#### Pipeline stages")
        stage_html = "".join(
            f'<div class="stage">{badge("done", "ok") if er.stage_status.get(s) == "done" else badge("pending", "off")} '
            f'{i}. {s}</div>' for i, s in enumerate(ml.ML_STAGES, 1))
        st.markdown(stage_html, unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("Input feature shape", f"{er.input_feature_shape[0]} × {er.input_feature_shape[1]}")
        m[1].metric("Output (mask) shape", f"{er.output_feature_shape[0]} × {er.output_feature_shape[1]}")
        m[2].metric("STFT magnitude shape", f"{er.estimated_magnitude.shape[0]} × {er.estimated_magnitude.shape[1]}")
        m[3].metric("Processing time", f"{er.processing_time_s:.2f} s")
        for w in er.warnings:
            st.warning(w)
        st.audio(wav(er.output, sr), format="audio/wav")
        chart(plots.waveform(er.output, sr, pl.METHOD_ML, "ML-enhanced waveform"), "wave_ml")
        t, f, sdb = sa.spectrogram_db(er.output, sr)
        chart(plots.spectrogram(t, f, sdb, "ML-enhanced spectrogram"), "spec_ml")
        st.caption("Limitation: only the magnitude is estimated; reusing the noisy phase leaves some residual noise "
                   "and roughness, especially at low SNR. Output length equals input length exactly.")
    elif result is not None and pl.METHOD_ML in result.errors:
        st.error(f"ML enhancement failed: {result.errors[pl.METHOD_ML]}")
    elif enhancer:
        st.info("Select **ML enhancement** or **Compare all available methods** and press Process to run inference.")
    else:
        st.info("No trained model is available, so ML enhancement is disabled. The DSP pipeline is fully functional.")

    if summary:
        with st.expander("Training summary (measured on held-out recordings)", expanded=False):
            st.markdown(f"Trained {summary.get('trained_at', '?')} on **{summary.get('data_source')}** data — "
                        f"{summary.get('n_train_recordings')} training / {summary.get('n_validation_recordings')} "
                        f"validation recordings ({summary.get('split')}); {summary.get('n_training_frames')} frames, "
                        f"{summary.get('n_features')} features → {summary.get('n_outputs')} outputs.")
            val = summary.get("validation", {})
            st.markdown(f"Mask R² vs. mean predictor on validation frames: **{val.get('mask_r2_vs_mean_predictor')}**")
            if val.get("per_input_snr"):
                vdf = pd.DataFrame(val["per_input_snr"]).T
                st.dataframe(vdf, width="stretch")
                st.caption(f"Noise type in validation: {val.get('noise_type')}. Values are mean SNRs in dB.")

    with st.expander("Train or retrain the model", expanded=enhancer is None):
        st.markdown(
            "**Real model (recommended):** put clean speech WAV/FLAC files in `data/clean/` "
            "(e.g. LibriSpeech, VCTK, or your own quiet recordings; `data/clean/<speaker>/…` enables a speaker-wise "
            "split) and run:\n\n```bash\npython train_model.py --data-dir data/clean\n```\n"
            "**Pipeline test only:** `python train_model.py --synthetic` trains on generated speech-like signals.")
        b1, b2 = st.columns(2)
        quick = b1.button("Train quick synthetic demo model (~1 min)", help="Synthetic data – pipeline test only.")
        real = b2.button(f"Train on data/clean ({n_training_files} files)", disabled=n_training_files < 2)
        if quick or real:
            import train_model as tm
            with st.status("Training…", expanded=True) as box:
                try:
                    kwargs = (dict(synthetic=True, n_synthetic=24, max_frames=30_000, n_estimators=25) if quick
                              else dict(data_dir=config.CLEAN_SPEECH_DIR))
                    tm.train(log=box.write, **kwargs)
                    box.update(label="Training finished", state="complete")
                    get_enhancer.clear()
                    st.rerun()
                except (FileNotFoundError, ValueError) as exc:
                    box.update(label="Training failed", state="error")
                    st.error(str(exc))

# --------------------------------------------------------------------------
# Section G – Results comparison
# --------------------------------------------------------------------------
with tabs[5]:
    if need_result():
        r = result
        for name, err in r.errors.items():
            st.error(f"{name} failed: {err}")
        if r.has_reference:
            src = ("the uploaded recording (before noise was added)" if r.settings.mode == "simulate"
                   else "the clean reference file you supplied")
            st.success(f"Clean reference available: {src}. Reference-based metrics are ground-truth measurements "
                       "for this controlled setup.")
        else:
            st.info("No clean reference: only reference-free statistics are measurable. SNR, RMSE and correlation "
                    "are shown as N/A because they require the true clean signal.")
        st.dataframe(fmt_table(r.table), width="stretch", hide_index=True)
        if r.has_reference:
            st.caption("SNR = 10·log10(Σs² / Σ(y−s)²) with s = clean reference, y = signal under test. "
                       "SNR improvement = output SNR − input SNR. Segmental SNR averages 20 ms frames (clamped to "
                       "[−10, 35] dB, silent frames skipped).")
            c1, c2 = st.columns(2)
            with c1:
                chart(plots.metric_bars(r.table[r.table["Method"] != r.input_name], "SNR improvement (dB)",
                                        "SNR improvement over the noisy input"), "bar_snr")
            with c2:
                chart(plots.metric_bars(r.table, "RMSE vs reference", "RMSE against the clean reference"), "bar_rmse")
        sigs = {"Clean reference": r.reference} if r.has_reference else {}
        sigs[r.input_name] = r.input_signal
        sigs.update(r.outputs)
        st.markdown("#### Time domain (shared time and amplitude axes)")
        chart(plots.stacked_waveforms(sigs, sr), "wave_all")
        st.markdown("#### Frequency spectra")
        chart(plots.spectra({k: sa.compute_fft(v, sr) for k, v in sigs.items()}, log_x=log_axis), "fft_all")
        st.markdown("#### Spectrograms")
        grid = st.columns(2)
        for i, (k, v) in enumerate(sigs.items()):
            with grid[i % 2]:
                t, f, sdb = sa.spectrogram_db(v, sr)
                chart(plots.spectrogram(t, f, sdb, k, height=260), f"spec_all_{i}")
        st.markdown("#### Listen")
        lc = st.columns(len(sigs))
        for c, (k, v) in zip(lc, sigs.items()):
            c.caption(k)
            c.audio(wav(v, sr), format="audio/wav")

# --------------------------------------------------------------------------
# Section H – Downloads and report
# --------------------------------------------------------------------------
def build_downloads(r: pl.PipelineResult) -> dict[str, tuple[bytes, str, str]]:
    stem = Path(audio.filename).stem
    files = {}
    for name, out in r.outputs.items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        files[f"{name} (WAV)"] = (wav(out, sr), f"{stem}_{slug}.wav", "audio/wav")
    if r.settings.mode == "simulate":
        files["Noisy input (WAV)"] = (wav(r.input_signal, sr), f"{stem}_noisy_{r.settings.snr_db:g}dB.wav", "audio/wav")
    csv = r.table.to_csv(index=False, na_rep="N/A").encode()
    files["Results table (CSV)"] = (csv, f"{stem}_results.csv", "text/csv")
    return files


def build_report_html(r: pl.PipelineResult) -> bytes:
    sigs = ({"Clean reference": r.reference} if r.has_reference else {})
    sigs[r.input_name] = r.input_signal
    sigs.update(r.outputs)
    if enhancer:
        ml_info = {"Model": enhancer.name, "Type": enhancer.model_type,
                   "Training data": enhancer.metadata.get("data_source"),
                   "Training recordings": enhancer.metadata.get("n_train_recordings"),
                   "Validation recordings": enhancer.metadata.get("n_validation_recordings"),
                   "Features / outputs": f"{enhancer.metadata.get('n_features')} / {enhancer.metadata.get('n_outputs')}",
                   "Used in this run": pl.METHOD_ML in r.outputs}
        if enhancer.metadata.get("warning"):
            ml_info["Warning"] = enhancer.metadata["warning"]
    else:
        ml_info = {"Status": "No trained model available – ML enhancement not run."}
    html_doc = rg.build_report(metadata=audio.metadata(), settings=r.settings.as_dict(), results_df=r.table,
                               signals=sigs, sr=sr, ml_info=ml_info, has_reference=r.has_reference,
                               processing_log=audio.processing_log)
    return html_doc.encode("utf-8")


with tabs[6]:
    if need_result():
        files = build_downloads(result)
        st.markdown("All WAV files are 16-bit PCM at the processing sample rate "
                    f"(**{sr} Hz**). Any signal whose peak exceeds 0.99 is scaled down (not clipped) on export.")
        cols = st.columns(3)
        for i, (label, (data, fname, mime)) in enumerate(files.items()):
            cols[i % 3].download_button(label, data, file_name=fname, mime=mime, width="stretch", key=f"dl_{i}")
        st.markdown("#### Analysis report")
        if st.button("Generate HTML report"):
            with st.spinner("Rendering report…"):
                st.session_state["report"] = build_report_html(result)
        if "report" in st.session_state and not stale:
            st.download_button("Download report (HTML)", st.session_state["report"],
                               file_name=f"{Path(audio.filename).stem}_report.html", mime="text/html")
            st.caption("Open the HTML file in a browser; use Print → Save as PDF for a PDF copy.")

with download_slot:
    if result is not None:
        st.markdown("**Downloads**")
        for i, (label, (data, fname, mime)) in enumerate(build_downloads(result).items()):
            st.download_button(label, data, file_name=fname, mime=mime, width="stretch", key=f"sdl_{i}")

# --------------------------------------------------------------------------
# Practical applications (Experiment 6)
# --------------------------------------------------------------------------
with tabs[7]:
    st.markdown("""
#### Where this pipeline is useful
| Scenario | Typical noise | Suitable stage |
|---|---|---|
| **Lecture / classroom recordings** | projector fan, HVAC hum, room noise | High-pass/low-pass filtering for hum & hiss; spectral subtraction for steady fan noise |
| **Online meetings** | laptop fan, keyboard, background chatter | Real-time spectral or ML masks (as in commercial noise suppressors) |
| **Voice notes / interviews** | street or café noise | ML enhancement trained on matching noise types |
| **Speech-recognition preprocessing** | any | Enhancement before ASR can lower word error rate, but aggressive filtering can also *hurt* ASR, so it must be validated |

#### DSP vs. ML – the key difference
* **DSP** applies a fixed rule designed by an engineer (a filter shape, a subtraction formula). It needs no data,
  is predictable, but cannot adapt to noise that overlaps speech.
* **ML** learns the mapping from noisy to clean features from examples. It can adapt to complex noise, but only as
  well as its training data represents the real situation — and it must be evaluated on held-out recordings.

#### Honest scope of this project
The ML component is a genuine but small baseline (Random Forest on STFT band features). Production systems use deep
neural networks trained on hundreds of hours of speech and noise (e.g. RNNoise, DCCRN, Demucs). The `SpeechEnhancer`
interface in `ml_enhancement.py` is designed so such a model could be plugged in later.
""")
