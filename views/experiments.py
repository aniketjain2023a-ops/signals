"""Experiments page: guided demo checklist, SNR sweep, quantitative evaluation and applications."""

from __future__ import annotations

import numpy as np
import streamlit as st

import app_state as state
import metrics
import noise_simulation as ns
import pipeline as pl
import plots
import ui

SWEEP_LEVELS = (20.0, 10.0, 0.0)
# Ordinal coral ramp for increasing noise (light = mild, dark = severe); used only in small multiples.
SWEEP_COLORS = {"20 dB SNR": "#F59A9B", "10 dB SNR": "#F05D5E", "0 dB SNR": "#B8383A"}


def checklist(audio, res) -> None:
    fresh = audio is not None and res is not None and not state.is_stale(audio)
    sweep_done = bool(st.session_state.get("snr_sweep")) and audio is not None
    dsp_done = fresh and any(m in res.outputs for m in (pl.METHOD_LOWPASS, pl.METHOD_SPECSUB))
    ml_done = fresh and pl.METHOD_ML in res.outputs
    quant_done = fresh and res.has_reference
    with st.container(key="card-exp-checklist"):
        ui.section("Demonstration plan", "Six experiments for a faculty demo. Each one turns green only when you have "
                                         "actually run it in this session.", "flask", tone="cyan")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.steps([
                ("1 · Original speech analysis", "Waveform, FFT, dominant frequency, spectrogram",
                 "done" if audio else "active"),
                ("2 · Noise simulation", "Compare 20, 10 and 0 dB SNR (below)", "done" if sweep_done else ""),
                ("3 · DSP noise reduction", "Butterworth filter and spectral subtraction", "done" if dsp_done else ""),
            ])
        with c2:
            ui.steps([
                ("4 · ML-based enhancement", "Run the trained model, compare with DSP", "done" if ml_done else ""),
                ("5 · Quantitative evaluation", "SNR, SNR improvement, RMSE with a clean reference",
                 "done" if quant_done else ""),
                ("6 · Practical application", "Lectures, meetings, voice notes, ASR (below)", ""),
            ], start=4)
        l1, l2, l3 = st.columns(3)
        l1.page_link(state.PAGES["analysis"], label="Audio analysis", icon=":material/graphic_eq:")
        l2.page_link(state.PAGES["lab"], label="Noise reduction lab", icon=":material/tune:")
        l3.page_link(state.PAGES["ml"], label="ML enhancement", icon=":material/neurology:")


def snr_sweep(audio) -> None:
    with st.container(key="card-exp-sweep"):
        ui.section("Experiment 2 · Noise levels", "The same recording at 20, 10 and 0 dB SNR, using the noise type and "
                                                  "seed from the Noise Reduction lab. Measured SNR should equal the target.",
                   "bars")
        if audio is None:
            ui.empty_state("upload", "Needs a recording", "Upload audio on the Overview page to run this experiment.",
                           flat_wave=False)
            return
        noise_type, seed = state.setting("noise_type"), int(state.setting("seed"))
        run = st.button("Run noise-level comparison", icon=":material/play_arrow:", type="primary", key="run_sweep")
        if run:
            if state.analysis(audio)["stats"]["RMS amplitude"] < 1e-5:
                ui.notice("bad", "The recording is silent", "A target SNR cannot be defined for silence.")
                return
            sigs, rows = {}, []
            for lvl in SWEEP_LEVELS:
                noisy, noise = ns.add_noise_at_snr(audio.signal, lvl, noise_type, seed)
                sigs[f"{lvl:g} dB SNR"] = noisy
                rows.append([f"{lvl:g} dB", f"{round(metrics.snr_db(audio.signal, noisy), 3) + 0.0:.3f} dB",  # no "-0.000"
                             f"{np.sqrt(np.mean(noise ** 2)):.5f}", f"{metrics.correlation(audio.signal, noisy):.4f}"])
            st.session_state.snr_sweep = {"digest": st.session_state.audio_digest, "sr": audio.sr,
                                          "noise_type": noise_type, "seed": seed, "signals": sigs, "rows": rows}
        sweep = st.session_state.get("snr_sweep")
        if not sweep or sweep["digest"] != st.session_state.audio_digest or sweep["sr"] != audio.sr:
            st.caption("Press the button to generate the three noisy versions.")
            return
        st.caption(f"{ns.NOISE_TYPES[sweep['noise_type']]} · seed {sweep['seed']}")
        ui.simple_table(["Target SNR", "Measured SNR", "Noise RMS", "Correlation with clean"], sweep["rows"])
        plots.COLORS.update(SWEEP_COLORS)
        st.plotly_chart(plots.stacked_waveforms({"Clean reference": audio.signal, **sweep["signals"]}, audio.sr, 110),
                        width="stretch", config=plots.CONFIG, key="exp_sweep_wave")


def quantitative(audio, res) -> None:
    with st.container(key="card-exp-quant"):
        ui.section("Experiment 5 · Quantitative evaluation", "Ground-truth metrics from the latest Noise Reduction run.",
                   "chart", tone="cyan")
        fresh = audio is not None and res is not None and not state.is_stale(audio)
        if not fresh or not res.has_reference:
            reason = ("Run a method in the Noise Reduction lab first." if not fresh else
                      "This run has no clean reference. Use <b>Simulated noise</b> mode, or upload a sample-aligned "
                      "clean reference in <b>Real recording</b> mode.")
            ui.empty_state("chart", "No reference-based metrics yet", reason, flat_wave=False)
            return
        enh = res.table[res.table["Method"] != res.input_name]
        best_snr = enh.loc[enh["SNR improvement (dB)"].idxmax(), "Method"] if len(enh) > 1 else None
        best_rmse = enh.loc[enh["RMSE vs reference"].idxmin(), "Method"] if len(enh) > 1 else None
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(plots.metric_bars(enh, "SNR improvement (dB)", best=best_snr), width="stretch",
                            config=plots.CONFIG, key="exp_bar_snr")
        with c2:
            st.plotly_chart(plots.metric_bars(res.table, "RMSE vs reference", best=best_rmse), width="stretch",
                            config=plots.CONFIG, key="exp_bar_rmse")
        input_snr = res.table.iloc[0]["Input SNR (dB)"]
        st.caption(f"Input SNR {input_snr:.2f} dB. ★ marks the best enhancement method for each metric (higher SNR "
                   "improvement, lower RMSE). SNR measures waveform error, not perceived quality.")


APPLICATIONS = [
    ("mic", "Lecture recordings", "Projector fans and room hum. Filtering removes hiss; spectral subtraction handles "
                                  "steady fan noise."),
    ("channels", "Online meetings", "Laptop fans and keyboard noise. Real-time spectral or ML masks, as used by "
                                    "commercial noise suppressors."),
    ("wave", "Voice notes & interviews", "Street or café noise. ML enhancement trained on matching noise types."),
    ("sparkles", "Speech recognition", "Enhancement before ASR can lower word error rate – but aggressive filtering can "
                                       "also hurt it, so validate first."),
]


def applications() -> None:
    with st.container(key="card-exp-apps"):
        ui.section("Experiment 6 · Practical applications", "Where this pipeline is useful, and which stage fits.",
                   "layers")
        cells = "".join(f'<div class="sn-feature"><b>{ui.icon(ic, "#A78BFA")}{ui.esc(t)}</b><span>{ui.esc(d)}</span></div>'
                        for ic, t, d in APPLICATIONS)
        st.html(f'<div class="sn-features" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr));'
                f'margin-top:.4rem">{cells}</div>')
        st.markdown(
            "**DSP vs ML.** DSP applies a fixed rule designed by an engineer – predictable and data-free, but blind to "
            "noise that overlaps speech. ML learns the noisy → clean mapping from examples, so it can adapt, but only "
            "as well as its training data matches reality, which is why it is evaluated on held-out recordings.")


def render() -> None:
    ui.page_header("Experiments", "Experiment lab",
                   "Guided experiments for your presentation – every number here is measured from your audio in this "
                   "session.")
    audio, _ = state.current_audio()
    res = state.result()
    checklist(audio, res)
    snr_sweep(audio)
    quantitative(audio, res)
    applications()
