"""Audio Analysis page: time-domain statistics, FFT, band energy, spectrogram and PSD."""

from __future__ import annotations

import numpy as np
import streamlit as st

import app_state as state
import config
import plots
import ui
from views.common import default_kw, require_audio


def render() -> None:
    ui.page_header("Audio analysis", "Time & frequency analysis",
                   "Everything on this page is computed from your recording: the waveform statistics, the FFT "
                   "spectrum, how energy is distributed across speech bands, and how the spectrum evolves over time.")
    audio = require_audio()
    if audio is None:
        return
    an = state.analysis(audio)
    s = an["stats"]
    f_dom, centroid = an["dominant"], an["centroid"]

    ui.metric_cards([
        dict(icon="target", label="Dominant frequency", value=f"{f_dom:.1f}" if np.isfinite(f_dom) else "N/A",
             unit="Hz" if np.isfinite(f_dom) else None, desc="Often the pitch (F0) or a low harmonic"),
        dict(icon="bars", label="Spectral centroid", tone="cyan", value=f"{centroid:.0f}" if np.isfinite(centroid) else "N/A",
             unit="Hz" if np.isfinite(centroid) else None, desc="Centre of mass of the spectrum"),
        dict(icon="wave", label="RMS level", value=f"{s['RMS (dBFS)']:.1f}", unit="dBFS",
             desc=f"RMS amplitude {s['RMS amplitude']:.4f}"),
        dict(icon="chart", label="Peak amplitude", tone="gray", value=f"{s['Peak amplitude']:.3f}",
             desc="After peak normalisation to 0.95" if audio.normalization_gain != 1 else "Full scale = 1.0"),
    ])

    with st.container(key="card-an-wave"):
        ui.section("Time domain", "Waveform and amplitude statistics of the mono working signal.", "wave")
        st.plotly_chart(plots.waveform(audio.signal, audio.sr, "Original", height=300), width="stretch",
                        config=plots.CONFIG, key="an_wave")
        crest = s["Crest factor (dB)"]
        ui.stats([
            ("Amplitude range", f"{s['Min amplitude']:.3f} … {s['Max amplitude']:.3f}"),
            ("Crest factor", f"{crest:.1f} dB" if np.isfinite(crest) else "N/A"),
            ("Zero-crossing rate", f"{s['Zero-crossing rate (/s)']:.0f} /s"),
            ("Samples", f"{s['Samples']:,}"),
            ("Frequency resolution", f"{audio.sr / audio.signal.size:.3f} Hz"),
            ("Nyquist frequency", f"{audio.sr // 2:,} Hz"),
        ])

    with st.container(key="card-an-fft"):
        head, ctrl = st.columns([3, 1], vertical_alignment="center")
        with head:
            ui.section("Frequency spectrum", "Hann-windowed FFT of the whole recording, magnitude in dB relative to "
                                             "full scale. The cyan marker is the dominant frequency.", "bars", tone="cyan")
        with ctrl:
            axis = st.segmented_control("Frequency axis", ["Log", "Linear"], key="freq_axis", required=True,
                                        persist_state="session", width="stretch", label_visibility="collapsed",
                                        **default_kw("freq_axis", "default"))
        log_x = axis != "Linear"
        c1, c2 = st.columns([1.65, 1], gap="medium")
        with c1:
            freqs, mag = an["fft"]
            st.plotly_chart(plots.spectrum(freqs, mag, "Original", log_x, marker_hz=f_dom, height=340),
                            width="stretch", config=plots.CONFIG, key="an_fft")
        with c2:
            st.plotly_chart(plots.band_energy(an["bands"], height=340), width="stretch", config=plots.CONFIG,
                            key="an_bands")

    with st.container(key="card-an-spec"):
        ui.section("Spectrogram", f"Short-time Fourier transform: {config.N_FFT}-sample Hann window "
                                  f"({1000 * config.N_FFT / audio.sr:.0f} ms), {config.HOP_LENGTH}-sample hop. "
                                  "Brighter = louder. Horizontal stripes are pitch harmonics; gaps are pauses.", "grid")
        t, f, sdb = an["spectrogram"]
        st.plotly_chart(plots.spectrogram(t, f, sdb, height=340), width="stretch", config=plots.CONFIG, key="an_spec")

    with st.container(key="card-an-psd"):
        ui.section("Power spectral density", "Welch's method: averaged periodograms of overlapping segments "
                                             "give a smoother estimate of power per hertz.", "chart", tone="cyan")
        pf, pp = an["psd"]
        st.plotly_chart(plots.psd(pf, pp, log_x, height=300), width="stretch", config=plots.CONFIG, key="an_psd")

    with st.expander("File metadata", icon=":material/info:"):
        ui.stats([(k, f"{v:,}" if isinstance(v, int) else str(v)) for k, v in audio.metadata().items()], mono=False)
