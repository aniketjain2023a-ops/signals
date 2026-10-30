"""
Plotly figure builders (SONORA visual language).

Rules
-----
* Colour follows the *signal*, never its position: the original/clean signal is
  always cyan, noisy always coral, Butterworth blue, spectral subtraction amber,
  ML violet. This palette passes colour-vision-deficiency and contrast checks
  for adjacent series on the card surface (#111827).
* Overlays never show more than two coloured series. Before/after views draw
  the "before" signal in neutral slate so any pair stays distinguishable;
  comparisons of many methods use small multiples or bars instead.
* Long signals are reduced for DISPLAY only (min/max envelope, max-pooling).
  All calculations elsewhere use the full-resolution data.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

TEXT = "#E5E7EB"
TEXT_MUTED = "#94A3B8"
GRID = "rgba(148,163,184,0.09)"
AXIS = "rgba(148,163,184,0.22)"
BEFORE = "#64748B"           # neutral slate used for the "before" signal in comparisons
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "JetBrains Mono, ui-monospace, Menlo, monospace"

COLORS = {
    "Clean reference": "#0EA5C6",
    "Original": "#0EA5C6",
    "Noisy": "#F05D5E",
    "Noisy recording": "#F05D5E",
    "Butterworth low-pass": "#3B82F6",
    "Spectral subtraction": "#C27C0E",
    "ML enhancement": "#8B5CF6",
}
# Sequential violet ramp for spectrograms: dark (quiet) -> light (loud).
SPECTROGRAM_SCALE = [
    [0.0, "#080B12"], [0.18, "#1A1238"], [0.38, "#3B1F86"], [0.58, "#6D3FE0"],
    [0.78, "#A78BFA"], [1.0, "#F5F3FF"],
]
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]}

MAX_WAVE_POINTS = 4000
MAX_SPEC_POINTS = 3000
MAX_SPECTRO_FRAMES = 500


def color_for(name: str) -> str:
    return COLORS.get(name, "#A78BFA")


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, height: int, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=TEXT, family=FONT), x=0, xanchor="left", y=0.98) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=36 if (title or legend) else 10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11.5, color=TEXT_MUTED),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=TEXT, size=11),
                    bgcolor="rgba(0,0,0,0)", itemclick="toggleothers"),
        hoverlabel=dict(bgcolor="#0D1220", bordercolor="#2A3550", font=dict(color=TEXT, family=MONO, size=11.5)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=4,
                     title_font=dict(size=11.5), automargin=True, spikecolor=TEXT_MUTED, spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, title_font=dict(size=11.5), automargin=True)
    return fig


# --------------------------------------------------------------------------
# Waveforms
# --------------------------------------------------------------------------
def _envelope(signal: np.ndarray, sr: int, max_points: int = MAX_WAVE_POINTS):
    """Return (t, lower, upper). For long signals: per-block min/max (keeps every peak visible)."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    if n <= max_points:
        t = np.arange(n) / sr
        return t, signal, signal
    block = int(np.ceil(n / max_points))
    n_blocks = n // block
    trimmed = signal[: n_blocks * block].reshape(n_blocks, block)
    t = (np.arange(n_blocks) * block + block / 2) / sr
    return t, trimmed.min(axis=1), trimmed.max(axis=1)


def _add_wave(fig: go.Figure, signal: np.ndarray, sr: int, name: str, color: str, row=None, col=None,
              showlegend: bool = True, fill_alpha: float = 0.45) -> None:
    """Draw a waveform as a filled min/max band (or a plain line for short signals)."""
    t, lo, hi = _envelope(signal, sr)
    kw = dict(row=row, col=col) if row else {}
    if lo is hi:   # short signal: exact samples
        fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", name=name, line=dict(color=color, width=1.3),
                                 showlegend=showlegend, hovertemplate="%{y:.4f}<extra>" + name + "</extra>"), **kw)
        return
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(color=color, width=0.8), name=name,
                             legendgroup=name, showlegend=showlegend,
                             hovertemplate="max %{y:.4f}<extra>" + name + "</extra>"), **kw)
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(color=color, width=0.8), name=name,
                             legendgroup=name, showlegend=False, fill="tonexty", fillcolor=_rgba(color, fill_alpha),
                             hovertemplate="min %{y:.4f}<extra>" + name + "</extra>"), **kw)


def waveform(signal: np.ndarray, sr: int, name: str = "Original", height: int = 300, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    _add_wave(fig, signal, sr, name, color_for(name))
    fig.update_xaxes(title_text="Time (s)", showspikes=True, spikemode="across")
    fig.update_yaxes(title_text="Amplitude (normalised)")
    return _layout(fig, height, title, legend=False)     # one series: the card title names it


def stacked_waveforms(signals: dict[str, np.ndarray], sr: int, row_height: int = 132) -> go.Figure:
    """Small multiples with shared time and amplitude axes (one colour per row, no overlap)."""
    names = list(signals)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=names)
    lim = max((float(np.max(np.abs(s))) for s in signals.values() if s.size), default=1.0) * 1.05 or 1.0
    for i, name in enumerate(names, start=1):
        _add_wave(fig, signals[name], sr, name, color_for(name), row=i, col=1, showlegend=False)
        fig.update_yaxes(range=[-lim, lim], title_text="Amp.", row=i, col=1, nticks=4)
    fig.update_xaxes(title_text="Time (s)", row=len(names), col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11.5, color=TEXT, family=FONT), x=0, xanchor="left")
    return _layout(fig, row_height * len(names) + 50, legend=False)


def before_after_waveform(before: np.ndarray, after: np.ndarray, sr: int, before_name: str, after_name: str,
                          height: int = 340) -> go.Figure:
    """Two stacked rows: input in neutral slate, output in its method colour."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"Before · {before_name}", f"After · {after_name}"))
    lim = max(float(np.max(np.abs(before))), float(np.max(np.abs(after)))) * 1.05 or 1.0
    _add_wave(fig, before, sr, before_name, BEFORE, row=1, col=1, showlegend=False, fill_alpha=0.35)
    _add_wave(fig, after, sr, after_name, color_for(after_name), row=2, col=1, showlegend=False)
    for r in (1, 2):
        fig.update_yaxes(range=[-lim, lim], title_text="Amp.", row=r, col=1, nticks=5)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11.5, color=TEXT, family=FONT), x=0, xanchor="left")
    return _layout(fig, height, legend=False)


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------
def _pool_spectrum(freqs: np.ndarray, mag: np.ndarray, log_x: bool):
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


def _vmarker(fig: go.Figure, x: float, text: str, log_x: bool, color: str = TEXT_MUTED) -> None:
    """Dotted vertical marker. On a log axis Plotly places shapes in data units but
    annotations in log10 units, so the two are added with different coordinates."""
    fig.add_vline(x=x, line=dict(color=color, dash="dot", width=1.2))
    fig.add_annotation(x=float(np.log10(x)) if log_x else float(x), y=1, yref="paper", text=text, showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=4, font=dict(color=TEXT, size=11, family=MONO),
                       bgcolor="rgba(13,18,32,0.85)", borderpad=3)


def _spectrum_trace(f, m, name, color, log_x, width=1.4):
    f2, m2 = _pool_spectrum(np.asarray(f), np.asarray(m), log_x)
    if log_x:
        keep = f2 > 0
        f2, m2 = f2[keep], m2[keep]
    y = 20 * np.log10(m2 + config.EPS)
    return go.Scatter(x=f2, y=y, mode="lines", name=name, line=dict(color=color, width=width),
                      hovertemplate="%{x:.1f} Hz · %{y:.1f} dB<extra>" + name + "</extra>")


def spectrum(freqs, mag, name: str = "Original", log_x: bool = True, marker_hz: float | None = None,
             marker_label: str = "Dominant", height: int = 340, floor_db: float | None = None) -> go.Figure:
    fig = go.Figure(_spectrum_trace(freqs, mag, name, color_for(name), log_x))
    if marker_hz is not None and np.isfinite(marker_hz) and marker_hz > 0:
        _vmarker(fig, marker_hz, f"{marker_label} {marker_hz:.0f} Hz", log_x, "#22D3EE")
    fig.update_xaxes(title_text="Frequency (Hz)", type="log" if log_x else "linear")
    fig.update_yaxes(title_text="Magnitude (dBFS)", range=[floor_db, None] if floor_db is not None else None)
    return _layout(fig, height, legend=False)


def before_after_spectrum(before: tuple, after: tuple, before_name: str, after_name: str, log_x: bool = True,
                          marker_hz: float | None = None, marker_label: str = "", height: int = 340) -> go.Figure:
    fig = go.Figure([
        _spectrum_trace(*before, f"Before · {before_name}", BEFORE, log_x, width=1.2),
        _spectrum_trace(*after, f"After · {after_name}", color_for(after_name), log_x, width=1.5),
    ])
    if marker_hz:
        _vmarker(fig, marker_hz, f"{marker_label} {marker_hz:.0f} Hz", log_x)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log" if log_x else "linear")
    fig.update_yaxes(title_text="Magnitude (dBFS)")
    return _layout(fig, height)


def psd(freqs: np.ndarray, power: np.ndarray, log_x: bool = True, height: int = 300) -> go.Figure:
    fig = go.Figure(_spectrum_trace(freqs, np.sqrt(power), "PSD", "#22D3EE", log_x))
    fig.update_xaxes(title_text="Frequency (Hz)", type="log" if log_x else "linear")
    fig.update_yaxes(title_text="PSD (dB re 1 FS²/Hz)")
    return _layout(fig, height, legend=False)


def spectrogram(times: np.ndarray, freqs: np.ndarray, s_db: np.ndarray, height: int = 320, title: str | None = None,
                dynamic_range_db: float = 80.0, show_scale: bool = True) -> go.Figure:
    if s_db.shape[1] > MAX_SPECTRO_FRAMES:
        step = int(np.ceil(s_db.shape[1] / MAX_SPECTRO_FRAMES))
        n = (s_db.shape[1] // step) * step
        s_db = s_db[:, :n].reshape(s_db.shape[0], -1, step).max(axis=2)
        times = times[:n:step]
    vmax = float(np.max(s_db))
    fig = go.Figure(go.Heatmap(
        x=times, y=freqs, z=s_db, colorscale=SPECTROGRAM_SCALE, zmin=vmax - dynamic_range_db, zmax=vmax,
        showscale=show_scale,
        colorbar=dict(title=dict(text="dBFS", font=dict(color=TEXT_MUTED, size=11)), tickfont=dict(color=TEXT_MUTED, size=10),
                      thickness=10, outlinewidth=0, len=0.9),
        hovertemplate="t %{x:.2f} s · %{y:.0f} Hz · %{z:.1f} dBFS<extra></extra>",
    ))
    fig.update_xaxes(title_text="Time (s)", showgrid=False)
    fig.update_yaxes(title_text="Frequency (Hz)", showgrid=False)
    fig = _layout(fig, height, title, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def filter_response(freqs: np.ndarray, single_db: np.ndarray, cutoff_hz: float, zero_phase: bool,
                    log_x: bool = True, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs[1:], y=single_db[1:], mode="lines", name="Single pass |H(f)|",
                             line=dict(color="#22D3EE", width=2), hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>"))
    if zero_phase:
        fig.add_trace(go.Scatter(x=freqs[1:], y=2 * single_db[1:], mode="lines", name="Zero-phase |H(f)|²",
                                 line=dict(color="#A78BFA", width=2, dash="dash"),
                                 hovertemplate="%{x:.0f} Hz · %{y:.1f} dB<extra></extra>"))
    _vmarker(fig, cutoff_hz, f"fc {cutoff_hz:.0f} Hz", log_x)
    fig.add_hline(y=-3, line=dict(color=AXIS, width=1, dash="dot"))
    fig.update_xaxes(title_text="Frequency (Hz)", type="log" if log_x else "linear")
    fig.update_yaxes(title_text="Gain (dB)", range=[-100, 6])
    fig = _layout(fig, height)
    # Pass band sits along the top, so the legend goes bottom-left, clear of the cut-off label.
    fig.update_layout(legend=dict(x=0.01, xanchor="left", y=0.04, yanchor="bottom", bgcolor="rgba(13,18,32,0.85)"))
    return fig


def band_energy(rows: list[dict], height: int = 340) -> go.Figure:
    names = [r["Band"] for r in rows][::-1]
    vals = [r["Energy (%)"] for r in rows][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color="#8B5CF6", cornerradius=4),
                           text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
                           textfont=dict(color=TEXT, family=MONO, size=11),
                           hovertemplate="%{y}<br>%{x:.2f}% of energy<extra></extra>"))
    fig.update_xaxes(title_text="Share of spectral energy (%)", range=[0, max(vals + [1]) * 1.25], showgrid=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=TEXT))
    fig = _layout(fig, height, legend=False)
    fig.update_layout(hovermode="closest", bargap=0.35)
    return fig


def metric_bars(df, column: str, height: int = 300, best: str | None = None) -> go.Figure:
    """One bar per method for a single metric. Text values use neutral ink; colour carries identity."""
    sub = df[["Method", column]].dropna()
    fmt = "{:+.2f}" if "improvement" in column else "{:.2f}" if "dB" in column else "{:.4f}"
    fig = go.Figure(go.Bar(
        x=sub["Method"], y=sub[column], marker=dict(color=[color_for(m) for m in sub["Method"]], cornerradius=4),
        text=[fmt.format(v) + ("  ★" if m == best else "") for m, v in zip(sub["Method"], sub[column])],
        textposition="outside", cliponaxis=False, textfont=dict(color=TEXT, family=MONO, size=11.5),
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    fig.update_yaxes(title_text=column)
    short = {"Butterworth low-pass": "Butterworth", "Spectral subtraction": "Spectral sub.", "ML enhancement": "ML",
             "Noisy recording": "Noisy"}
    fig.update_xaxes(showgrid=False, tickangle=0, tickvals=list(sub["Method"]),
                     ticktext=[short.get(m, m) for m in sub["Method"]], tickfont=dict(color=TEXT))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig = _layout(fig, height, legend=False)
    fig.update_layout(hovermode="closest", bargap=0.45, margin=dict(t=30))
    return fig
