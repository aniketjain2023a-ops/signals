"""
Plotly figure builders for the dashboard.

Colours follow the *entity*, never its position: the clean reference is
always blue, the noisy signal always orange, and so on, so a signal keeps
the same colour in every chart. The palette was checked for colour-vision-
deficiency separation and contrast on the dashboard's dark surface.

Long signals are reduced for display only (min/max envelope for waveforms,
max-pooling for spectra and spectrograms); all calculations use the
full-resolution data.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

SURFACE = "#141c2b"
TEXT = "#e6e9ef"
TEXT_MUTED = "#a7b0c0"
GRID = "#263147"

# Fixed colour per signal role (dark-mode categorical palette, validated).
COLORS = {
    "Clean reference": "#3987e5",
    "Original": "#3987e5",
    "Noisy": "#d95926",
    "Noisy recording": "#d95926",
    "Butterworth low-pass": "#199e70",
    "Spectral subtraction": "#c98500",
    "ML enhancement": "#d55181",
}
SPECTROGRAM_SCALE = [
    [0.0, "#0b1220"], [0.25, "#104281"], [0.5, "#256abf"],
    [0.75, "#6da7ec"], [1.0, "#e8f1fd"],
]

MAX_WAVE_POINTS = 4000
MAX_SPEC_POINTS = 3000
MAX_SPECTRO_FRAMES = 500


def color_for(name: str) -> str:
    return COLORS.get(name, "#9085e9")


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01) if title else None,
        height=height,
        margin=dict(l=60, r=20, t=40 if title else 16, b=46),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1f2a3f", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _envelope(signal: np.ndarray, sr: int, max_points: int = MAX_WAVE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation so peaks remain visible after downsampling for display."""
    n = signal.size
    t = np.arange(n) / sr
    if n <= max_points:
        return t, signal
    block = int(np.ceil(n / (max_points // 2)))
    n_blocks = n // block
    trimmed = signal[: n_blocks * block].reshape(n_blocks, block)
    mins, maxs = trimmed.min(axis=1), trimmed.max(axis=1)
    tt = t[: n_blocks * block].reshape(n_blocks, block)[:, 0]
    out_t = np.repeat(tt, 2)
    out_y = np.empty(2 * n_blocks)
    out_y[0::2], out_y[1::2] = mins, maxs
    return out_t, out_y


def waveform(signal: np.ndarray, sr: int, name: str = "Original", title: str | None = None, height: int = 260) -> go.Figure:
    t, y = _envelope(np.asarray(signal), sr)
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", name=name,
                                 line=dict(color=color_for(name), width=1.2),
                                 hovertemplate="t = %{x:.3f} s<br>amplitude = %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    return _base_layout(fig, title, height)


def stacked_waveforms(signals: dict[str, np.ndarray], sr: int, title: str | None = None) -> go.Figure:
    """Small multiples with a shared time axis and a shared amplitude range."""
    names = list(signals)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=names)
    lim = max(float(np.max(np.abs(s))) for s in signals.values() if s.size) * 1.05 or 1.0
    for i, name in enumerate(names, start=1):
        t, y = _envelope(np.asarray(signals[name]), sr)
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name, showlegend=False,
                                   line=dict(color=color_for(name), width=1.1),
                                   hovertemplate=f"{name}<br>t = %{{x:.3f}} s<br>amplitude = %{{y:.4f}}<extra></extra>"),
                      row=i, col=1)
        fig.update_yaxes(range=[-lim, lim], title_text="Amp.", row=i, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(names), col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=TEXT)
        ann.x, ann.xanchor = 0.0, "left"
    return _base_layout(fig, title, height=150 * len(names) + 60)


def _vmarker(fig: go.Figure, x: float, text: str, log_x: bool) -> None:
    """Dotted vertical marker with a label.

    On a log axis Plotly positions shapes in data units but annotations in
    log10 units, so the two are added separately with the right coordinates.
    """
    fig.add_vline(x=x, line=dict(color=TEXT_MUTED, dash="dot", width=1))
    fig.add_annotation(x=float(np.log10(x)) if log_x else float(x), y=1, yref="paper", text=text,
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=TEXT, size=11))


def _pool_spectrum(freqs: np.ndarray, mag: np.ndarray, log_x: bool) -> tuple[np.ndarray, np.ndarray]:
    """Max-pool a long spectrum into ≤ MAX_SPEC_POINTS bins (log-spaced when log_x)."""
    if freqs.size <= MAX_SPEC_POINTS:
        return freqs, mag
    lo = max(freqs[1], 1.0)
    edges = (np.geomspace(lo, freqs[-1], MAX_SPEC_POINTS + 1) if log_x
             else np.linspace(0, freqs[-1], MAX_SPEC_POINTS + 1))
    idx = np.clip(np.searchsorted(edges, freqs) - 1, 0, MAX_SPEC_POINTS - 1)
    pooled = np.full(MAX_SPEC_POINTS, np.nan)
    np.fmax.at(pooled, idx, mag)
    centres = np.sqrt(edges[:-1] * edges[1:]) if log_x else 0.5 * (edges[:-1] + edges[1:])
    keep = np.isfinite(pooled)
    return centres[keep], pooled[keep]


def spectra(spectra_by_name: dict[str, tuple[np.ndarray, np.ndarray]], log_x: bool = True, title: str | None = None,
            ylabel: str = "Magnitude (dBFS)", to_db: bool = True, marker_hz: float | None = None,
            height: int = 340) -> go.Figure:
    fig = go.Figure()
    for name, (f, m) in spectra_by_name.items():
        f2, m2 = _pool_spectrum(np.asarray(f), np.asarray(m), log_x)
        if log_x:
            keep = f2 > 0
            f2, m2 = f2[keep], m2[keep]
        y = 20 * np.log10(m2 + config.EPS) if to_db else m2
        fig.add_trace(go.Scatter(x=f2, y=y, mode="lines", name=name,
                                   line=dict(color=color_for(name), width=1.4),
                                   hovertemplate=f"{name}<br>%{{x:.1f}} Hz<br>%{{y:.2f}}<extra></extra>"))
    if marker_hz is not None and np.isfinite(marker_hz) and marker_hz > 0:
        _vmarker(fig, marker_hz, f"{marker_hz:.0f} Hz", log_x)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log" if log_x else "linear")
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(hovermode="x unified", showlegend=len(spectra_by_name) > 1)
    return _base_layout(fig, title, height)


def psd(freqs: np.ndarray, power: np.ndarray, name: str = "Original", log_x: bool = True,
        title: str | None = None) -> go.Figure:
    return spectra({name: (freqs, np.sqrt(power))}, log_x=log_x, title=title,
                   ylabel="PSD (dB re 1 FS²/Hz)", to_db=True)


def spectrogram(times: np.ndarray, freqs: np.ndarray, s_db: np.ndarray, title: str | None = None,
                dynamic_range_db: float = 80.0, height: int = 300) -> go.Figure:
    if s_db.shape[1] > MAX_SPECTRO_FRAMES:
        step = int(np.ceil(s_db.shape[1] / MAX_SPECTRO_FRAMES))
        n = (s_db.shape[1] // step) * step
        s_db = s_db[:, :n].reshape(s_db.shape[0], -1, step).max(axis=2)
        times = times[:n:step]
    vmax = float(np.max(s_db))
    fig = go.Figure(go.Heatmap(
        x=times, y=freqs, z=s_db, colorscale=SPECTROGRAM_SCALE, zmin=vmax - dynamic_range_db, zmax=vmax,
        colorbar=dict(title=dict(text="dBFS", font=dict(color=TEXT_MUTED)), tickfont=dict(color=TEXT_MUTED)),
        hovertemplate="t = %{x:.3f} s<br>f = %{y:.0f} Hz<br>%{z:.1f} dBFS<extra></extra>",
    ))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frequency (Hz)")
    return _base_layout(fig, title, height)


def filter_response(freqs: np.ndarray, single_db: np.ndarray, cutoff_hz: float, zero_phase: bool,
                    log_x: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs[1:], y=single_db[1:], mode="lines", name="Single pass |H(f)|",
                             line=dict(color="#199e70", width=2)))
    if zero_phase:
        fig.add_trace(go.Scatter(x=freqs[1:], y=2 * single_db[1:], mode="lines",
                                 name="Effective zero-phase |H(f)|²",
                                 line=dict(color="#9085e9", width=2, dash="dash")))
    _vmarker(fig, cutoff_hz, f"cut-off {cutoff_hz:.0f} Hz", log_x)
    fig.add_hline(y=-3, line=dict(color=GRID, width=1))
    fig.update_xaxes(title_text="Frequency (Hz)", type="log" if log_x else "linear")
    fig.update_yaxes(title_text="Gain (dB)", range=[-100, 5])
    fig.update_layout(hovermode="x unified")
    return _base_layout(fig, "Butterworth filter frequency response", 320)


def band_energy(rows: list[dict]) -> go.Figure:
    names = [r["Band"] for r in rows][::-1]
    vals = [r["Energy (%)"] for r in rows][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color="#3987e5"),
                           text=[f"{v:.1f}%" for v in vals], textposition="outside",
                           textfont=dict(color=TEXT),
                           hovertemplate="%{y}<br>%{x:.2f}% of energy<extra></extra>"))
    fig.update_xaxes(title_text="Share of spectral energy (%)", range=[0, max(vals + [1]) * 1.18])
    return _base_layout(fig, "Energy distribution across speech bands", 300)


def metric_bars(df, column: str, title: str) -> go.Figure:
    """Single-measure bar chart for one comparison-table column."""
    sub = df[["Method", column]].dropna()
    fig = go.Figure(go.Bar(x=sub["Method"], y=sub[column],
                           marker=dict(color=[color_for(m) for m in sub["Method"]]),
                           text=[f"{v:+.2f}" if "improvement" in column else f"{v:.2f}" if "dB" in column
                                 else f"{v:.4f}" for v in sub[column]],
                           textposition="outside", textfont=dict(color=TEXT),
                           hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text=column)
    fig.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1))
    return _base_layout(fig, title, 320)
