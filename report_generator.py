"""
Self-contained HTML analysis report.

Figures are rendered with Matplotlib and embedded as base64 PNGs, so the
report opens offline and can be printed to PDF from any browser. Only values
that were actually computed in the session are included; reference-based
metrics are marked "N/A" when no clean reference existed.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io

import matplotlib

matplotlib.use("Agg")  # headless backend, no GUI required
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import signal_analysis as sa  # noqa: E402

PLOT_COLORS = {
    "Clean reference": "#2a78d6", "Original": "#2a78d6", "Noisy": "#eb6834",
    "Butterworth low-pass": "#1baf7a", "Spectral subtraction": "#c98500", "ML enhancement": "#d55181",
}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _waveforms_png(signals: dict[str, np.ndarray], sr: int) -> str:
    fig, axes = plt.subplots(len(signals), 1, figsize=(9, 1.7 * len(signals)), sharex=True, squeeze=False)
    lim = max(float(np.max(np.abs(s))) for s in signals.values()) * 1.05 or 1.0
    for ax, (name, s) in zip(axes[:, 0], signals.items()):
        ax.plot(np.arange(s.size) / sr, s, lw=0.5, color=PLOT_COLORS.get(name, "#6250d6"))
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Amp.")
        ax.set_title(name, loc="left", fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _spectra_png(signals: dict[str, np.ndarray], sr: int) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for name, s in signals.items():
        f, m = sa.compute_fft(s, sr)
        ax.semilogx(f[1:], 20 * np.log10(m[1:] + config.EPS), lw=0.7, label=name,
                    color=PLOT_COLORS.get(name, "#6250d6"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _spectrograms_png(signals: dict[str, np.ndarray], sr: int) -> str:
    fig, axes = plt.subplots(len(signals), 1, figsize=(9, 2.0 * len(signals)), sharex=True, squeeze=False)
    for ax, (name, s) in zip(axes[:, 0], signals.items()):
        t, f, sdb = sa.spectrogram_db(s, sr)
        vmax = sdb.max()
        im = ax.pcolormesh(t, f, sdb, shading="auto", cmap="Blues_r", vmin=vmax - 80, vmax=vmax)
        ax.set_ylabel("Hz")
        ax.set_title(name, loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="dBFS", shrink=0.8)
    return _fig_to_b64(fig)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return html.escape(str(v))


def _table(rows: dict | pd.DataFrame) -> str:
    if isinstance(rows, dict):
        body = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>" for k, v in rows.items())
        return f"<table>{body}</table>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in rows.columns)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in r) + "</tr>" for r in rows.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def interpret(df: pd.DataFrame, has_reference: bool) -> list[str]:
    """Plain-language interpretation derived only from the computed metrics."""
    notes = []
    if not has_reference:
        notes.append("No clean reference was available, so SNR, RMSE and correlation could not be computed. "
                     "Differences between methods can only be judged by listening and by the spectra.")
        return notes
    col = "SNR improvement (dB)"
    sub = df[df["Method"] != "Noisy"].dropna(subset=[col])
    if sub.empty:
        return notes
    best = sub.loc[sub[col].idxmax()]
    notes.append(f"The highest SNR improvement was achieved by <b>{html.escape(best['Method'])}</b> "
                 f"({best[col]:+.2f} dB).")
    for _, r in sub.iterrows():
        if r[col] < 0:
            notes.append(f"{html.escape(r['Method'])} lowered the SNR ({r[col]:+.2f} dB): it removed or distorted "
                         "more speech than noise for this input.")
    notes.append("SNR improvement measures waveform error only. It does not directly measure perceived quality or "
                 "intelligibility; listening tests or metrics such as PESQ/STOI would be needed for that.")
    return notes


def build_report(*, metadata: dict, settings: dict, results_df: pd.DataFrame, signals: dict[str, np.ndarray],
                 sr: int, ml_info: dict, has_reference: bool, processing_log: list[str]) -> str:
    waves = _waveforms_png(signals, sr)
    specs = _spectra_png(signals, sr)
    grams = _spectrograms_png(signals, sr)
    interp = "".join(f"<li>{n}</li>" for n in interpret(results_df, has_reference))
    log_items = "".join(f"<li>{html.escape(x)}</li>" for x in processing_log) or "<li>No changes.</li>"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speech Enhancement Report</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1b2333;background:#fff;line-height:1.5}}
 h1{{font-size:22px;margin-bottom:4px}} h2{{font-size:17px;border-bottom:1px solid #d5dae3;padding-bottom:4px;margin-top:28px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}} th,td{{border:1px solid #d5dae3;padding:5px 8px;text-align:left}}
 thead th{{background:#eef1f6}} img{{max-width:100%}} .muted{{color:#5b6475;font-size:13px}}
 .warn{{background:#fff6e5;border-left:4px solid #c98500;padding:8px 12px}}
</style></head><body>
<h1>{html.escape(config.PROJECT_TITLE)}</h1>
<p class="muted">Analysis report generated {now}. All values below were computed from the processed audio in this session.</p>

<h2>1. Input file</h2>{_table(metadata)}
<h3>Preprocessing applied</h3><ul>{log_items}</ul>

<h2>2. Processing settings</h2>{_table(settings)}

<h2>3. Methods</h2>
<ul>
<li><b>FFT / STFT analysis</b> – Hann-windowed FFT of the whole signal; STFT with {config.N_FFT}-sample frames and
{config.HOP_LENGTH}-sample hop at {sr} Hz.</li>
<li><b>Butterworth low-pass (DSP)</b> – IIR filter in second-order sections, applied forward-backward (zero phase).</li>
<li><b>Spectral subtraction (DSP)</b> – noise magnitude estimated from the quietest
{config.NOISE_ESTIMATE_PERCENTILE}% of frames and subtracted with over-subtraction and a spectral floor; noisy phase kept.</li>
<li><b>ML enhancement</b> – supervised Random Forest regression that predicts a per-band ratio mask from noisy STFT
band features (with ±{config.ML_CONTEXT_FRAMES} frames of context); estimated clean magnitude is resynthesised with the noisy phase.</li>
</ul>

<h2>4. ML model</h2>{_table(ml_info)}

<h2>5. Performance metrics</h2>{_table(results_df)}
<p class="muted">Reference-based metrics use error = output − clean reference. "N/A" means the metric requires a
known clean reference that was not available for this input.</p>

<h2>6. Graphs</h2>
<h3>Waveforms</h3><img alt="Waveforms" src="data:image/png;base64,{waves}">
<h3>Frequency spectra</h3><img alt="Spectra" src="data:image/png;base64,{specs}">
<h3>Spectrograms</h3><img alt="Spectrograms" src="data:image/png;base64,{grams}">

<h2>7. Interpretation</h2><ul>{interp}</ul>

<h2>8. Limitations</h2>
<ul>
<li>Simulated noise is stationary (white/pink/brown Gaussian) and does not represent babble, music, reverberation or other real environments.</li>
<li>A low-pass filter also removes genuine high-frequency speech (fricatives such as “s” and “f”) and cannot remove in-band noise.</li>
<li>Spectral subtraction assumes stationary noise and can create “musical noise” artefacts.</li>
<li>The ML baseline estimates magnitudes only and reuses the noisy phase, and a small Random Forest is far less capable than modern deep-learning enhancers.</li>
<li>When noise is simulated, the uploaded recording is treated as the clean reference, even if it already contains some noise.</li>
</ul>
</body></html>"""
