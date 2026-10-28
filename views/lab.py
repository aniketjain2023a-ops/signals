"""Noise Reduction lab: settings, processing, before/after comparison, players and metrics."""

from __future__ import annotations

import streamlit as st

import app_state as state
import config
import dsp_enhancement as dsp
import noise_simulation as ns
import pipeline as pl
import plots
import ui
from views.common import default_kw, flash, require_audio


def _reference_block() -> None:
    """Optional clean reference for real-recording mode (outside the form so it uploads immediately)."""
    if st.session_state.ref_bytes:
        ui.file_card(st.session_state.ref_name, ["clean reference", state.human_size(len(st.session_state.ref_bytes))])
        st.button("Remove reference", icon=":material/close:", on_click=state.remove_reference, key="rm_ref")
    else:
        key = f"ref_uploader_{st.session_state.ref_uploader_gen}"
        st.file_uploader("Optional clean reference (sample-aligned)", type=sorted(e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS),
                         key=key, on_change=state.on_ref_upload, args=(key,),
                         help="Only if you have the exact clean source of this noisy recording. It enables SNR, RMSE "
                              "and correlation; without it those metrics are reported as N/A.")


def controls(audio) -> None:
    with st.container(key="card-controls"):
        ui.section("Controls", "Settings apply when you press Run.", "filter")
        mode = st.segmented_control("Experiment mode", state.MODES, key="mode", required=True, persist_state="session",
                                    width="stretch", format_func=lambda m: "Simulated" if m == "Simulated noise" else "Real",
                                    **default_kw("mode", "default"))
        if mode == "Real recording":
            st.caption("Your upload already contains real noise. It is processed as-is; reference-based metrics need a "
                       "clean reference.")
            _reference_block()
        else:
            st.caption("Your upload is the clean reference. Seeded noise is added at the target SNR, so every metric is "
                       "a ground-truth measurement.")

        options = [m for m in state.METHOD_OPTIONS if m != "ML enhancement" or state.ml_possible()]
        if st.session_state.get("method") not in (None, *options):
            st.session_state["method"] = "Compare all methods"      # stored choice is no longer available
        max_cut = int((audio.sr / 2 - 50) // 50 * 50)
        if st.session_state.get("cutoff", 0) > max_cut:
            st.session_state["cutoff"] = max_cut

        with st.form("process_form", border=False, enter_to_submit=False):
            st.selectbox("Enhancement method", options, key="method", persist_state="session",
                         **({} if "method" in st.session_state else {"index": 0}))
            if not state.ml_possible():
                st.caption("ML enhancement appears here once a trained model is loaded (see ML Enhancement)."
                           if state.enhancer()[0] is None else "ML enhancement needs 16 kHz resampling (Overview).")
            if mode == "Simulated noise":
                st.html('<div class="sn-label" style="margin-top:.4rem">Noise simulation</div>')
                st.selectbox("Target input SNR", list(config.SNR_PRESETS_DB), key="snr_label", persist_state="session",
                             **({} if "snr_label" in st.session_state else {"index": 1}))
                c1, c2 = st.columns([1.4, 1])
                c1.selectbox("Noise type", list(ns.NOISE_TYPES), key="noise_type", persist_state="session",
                             format_func=lambda k: k.capitalize() + " noise")
                c2.number_input("Seed", 0, 2**31 - 1, step=1, key="seed", persist_state="session", **default_kw("seed"))
            st.html('<div class="sn-label" style="margin-top:.4rem">Butterworth low-pass</div>')
            st.slider("Cut-off frequency (Hz)", 300, max_cut, step=50, key="cutoff", persist_state="session",
                      **({} if "cutoff" in st.session_state else {"value": min(state.DEFAULTS["cutoff"], max_cut)}),
                      help=f"Must be below the Nyquist frequency ({audio.sr // 2:,} Hz).")
            st.slider("Filter order", config.MIN_FILTER_ORDER, config.MAX_FILTER_ORDER, key="order",
                      persist_state="session", **default_kw("order"), help="Higher order = steeper roll-off.")
            with st.expander("Spectral subtraction"):
                st.slider("Over-subtraction α", 0.5, 4.0, step=0.1, key="alpha", persist_state="session", **default_kw("alpha"))
                st.slider("Spectral floor β", 0.0, 0.2, step=0.01, key="beta", persist_state="session", **default_kw("beta"))
            submitted = st.form_submit_button("Run enhancement", type="primary", icon=":material/play_arrow:",
                                              width="stretch")
        if submitted:
            ok, msg = state.run_processing(audio)
            if ok:
                flash(msg)
                st.rerun()
            ui.notice("bad", "Processing could not run", ui.esc(msg))


def comparison(audio, res: pl.PipelineResult) -> None:
    log_x = state.setting("freq_axis") != "Linear"
    names = list(res.outputs)
    with st.container(key="card-compare"):
        ui.section("Before / after", "Compare the input with an enhanced output. The input is drawn in neutral slate; "
                                     "each method keeps its own colour throughout the app.", "wave")
        if st.session_state.get("compare_output") not in names:
            st.session_state["compare_output"] = names[0]
        name = st.segmented_control("Output to compare", names, key="compare_output", required=True,
                                    label_visibility="collapsed")
        out = res.outputs[name]
        t_wave, t_spec, t_fft, t_filt = st.tabs(["Waveform", "Spectrogram", "Spectrum", "Filter response"])
        with t_wave:
            st.plotly_chart(plots.before_after_waveform(res.input_signal, out, audio.sr, res.input_name, name),
                            width="stretch", config=plots.CONFIG, key="lab_ba_wave")
        with t_spec:
            c1, c2 = st.columns(2, gap="small")
            with c1:
                t, f, sdb = state.cached_chart_data("spec", res.input_name, res.input_signal, audio.sr)
                st.plotly_chart(plots.spectrogram(t, f, sdb, height=300, title=f"Before · {res.input_name}",
                                                  show_scale=False), width="stretch", config=plots.CONFIG, key="lab_sp_b")
            with c2:
                t, f, sdb = state.cached_chart_data("spec", name, out, audio.sr)
                st.plotly_chart(plots.spectrogram(t, f, sdb, height=300, title=f"After · {name}"),
                                width="stretch", config=plots.CONFIG, key="lab_sp_a")
        with t_fft:
            marker = res.settings.cutoff_hz if name == pl.METHOD_LOWPASS else None
            st.plotly_chart(plots.before_after_spectrum(
                state.cached_chart_data("fft", res.input_name, res.input_signal, audio.sr),
                state.cached_chart_data("fft", name, out, audio.sr), res.input_name, name, log_x,
                marker_hz=marker, marker_label="fc"), width="stretch", config=plots.CONFIG, key="lab_fft")
        with t_filt:
            s = res.settings
            if s.cutoff_hz < audio.sr / 2:
                sos = dsp.design_butterworth_lowpass(s.cutoff_hz, s.order, audio.sr)
                fz, hdb = dsp.filter_frequency_response(sos, audio.sr)
                fr = res.details.get(pl.METHOD_LOWPASS)
                zero_phase = fr.zero_phase if fr is not None else True
                st.plotly_chart(plots.filter_response(fz, hdb, s.cutoff_hz, zero_phase, log_x), width="stretch",
                                config=plots.CONFIG, key="lab_filt")
                ui.stats([("Type", "Butterworth low-pass (SOS)"), ("Order", str(s.order)),
                          ("Cut-off", f"{s.cutoff_hz:.0f} Hz"),
                          ("Phase", "Zero-phase (forward-backward)" if zero_phase else "Causal (short-signal fallback)"),
                          ("Effective order", str(2 * s.order if zero_phase else s.order))], mono=False)
                if fr is not None and fr.note:
                    ui.notice("warn", "Short-signal fallback", ui.esc(fr.note))
            st.caption("A low-pass filter removes noise above the cut-off but also removes genuine high-frequency "
                       "speech (s, f, sh). It cannot remove noise inside the speech band.")
        if name == pl.METHOD_SPECSUB:
            ss = res.details[name]
            st.caption(f"Spectral subtraction: noise estimated from the quietest {ss.n_noise_frames} STFT frames; "
                       f"|Ŝ| = max(|Y| − α|N̂|, β|Y|) with α = {ss.alpha:g}, β = {ss.beta:g}; noisy phase reused.")
        elif name == pl.METHOD_ML:
            st.caption("ML enhancement: predicted band mask × noisy magnitude, inverse STFT with the noisy phase. "
                       "Details on the ML Enhancement page.")


def players(audio, res: pl.PipelineResult) -> None:
    items = []
    if res.has_reference:
        items.append(("Clean reference", res.reference))
    items.append((res.input_name + (" input" if res.input_name == "Noisy" else ""), res.input_signal))
    items += list(res.outputs.items())
    with st.container(key="card-players"):
        ui.section("Listen", "All players are 16-bit WAV at the processing rate. Signals above full scale are scaled "
                             "down for playback, never clipped.", "play", tone="cyan")
        per_row = 3 if len(items) >= 3 else len(items)
        for start in range(0, len(items), per_row):
            cols = st.columns(per_row, gap="medium")
            for col, (label, sig) in zip(cols, items[start:start + per_row]):
                with col:
                    color_key = "Noisy" if label.startswith("Noisy") else label
                    ui.player_label(label, plots.color_for(color_key))
                    st.audio(state.wav_for(label, sig, audio.sr), format="audio/wav")


def performance(res: pl.PipelineResult) -> None:
    with st.container(key="card-perf"):
        ui.section("Performance comparison", "Measured on this run. Reference-based columns need a known clean signal; "
                                             "otherwise they show N/A rather than an estimate.", "table",
                   right_html=ui.pill("Clean reference available", "ok", dot=True) if res.has_reference
                   else ui.pill("No clean reference", "warn", dot=True))
        colors = {**plots.COLORS, res.input_name: plots.color_for(res.input_name)}
        ui.results_table(res.table, colors, res.has_reference)
        if res.has_reference:
            st.caption("SNR = 10·log10(Σs² / Σ(y−s)²), s = clean reference, y = signal under test. "
                       "SNR improvement = output SNR − input SNR. BEST marks the top enhancement method per metric.")
        else:
            st.caption("Without the true clean signal, SNR, RMSE and correlation cannot be measured. Compare methods "
                       "by listening and by their spectrograms.")


def render() -> None:
    ui.page_header("Noise reduction", "Enhancement laboratory",
                   "Simulate controlled noise or process a real recording, then compare traditional DSP with machine "
                   "learning – by ear, by eye and by measurement.")
    audio = require_audio()
    if audio is None:
        return
    left, right = st.columns([1, 2.1], gap="medium")
    with left:
        controls(audio)
    with right:
        res = state.result()
        if res is None:
            with st.container(key="card-empty-results"):
                ui.empty_state("play", "No processed audio yet",
                               "Choose a method and press <b>Run enhancement</b>. Processing only runs when you ask "
                               "for it, so you can adjust settings freely.")
            return
        if state.is_stale(audio):
            ui.notice("warn", "Settings changed since the last run",
                      "The results below belong to the previous settings. Press <b>Run enhancement</b> to update them.")
        for method, err in res.errors.items():
            ui.notice("bad", f"{method} failed", ui.esc(err))
        if res.outputs:
            comparison(audio, res)
    if res.outputs:
        players(audio, res)
    performance(res)
