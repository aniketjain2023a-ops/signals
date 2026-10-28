"""
Reusable presentation components for the SONORA dashboard.

Every function here only builds HTML or calls simple Streamlit APIs – no signal
processing happens in this file. Components are rendered with ``st.html``
(sanitised HTML, no JavaScript) and styled by ``assets/styles.css``.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import numpy as np
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

# Inline SVG icons (24×24, stroke-based). Kept here so no icon font or extra
# dependency is needed.
_ICONS = {
    "upload": '<path d="M12 16V4"/><path d="m6 10 6-6 6 6"/><path d="M4 20h16"/>',
    "download": '<path d="M12 4v12"/><path d="m6 10 6 6 6-6"/><path d="M4 20h16"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "wave": '<path d="M3 12h3l3-8 4 16 3-8h5"/>',
    "bars": '<path d="M4 20V10M9 20V4M14 20v-8M19 20v-5"/>',
    "rate": '<path d="M3 12h2M7 7v10M11 4v16M15 8v8M19 10v4"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M10 10h4v4h-4z"/>'
           '<path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    "check": '<path d="m5 12 5 5 9-10"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'
            '<path d="M9 17v-3M12 17v-5M15 17v-2"/>',
    "filter": '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    "sparkles": '<path d="M12 3l1.8 4.6L18 9.5l-4.2 1.9L12 16l-1.8-4.6L6 9.5l4.2-1.9z"/>'
                '<path d="M19 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>',
    "report": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>',
    "alert": '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "x": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
            '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "rebuild": '<path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v4h-4"/>',
    "chart": '<path d="M4 20V4M4 20h16"/><path d="m8 14 3-3 3 2 5-6"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    "flask": '<path d="M9 3h6M10 3v6L4.5 19A1.5 1.5 0 0 0 5.8 21h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3"/>',
    "channels": '<path d="M4 9v6M8 6v12M12 9v6M16 6v12M20 9v6"/>',
    "play": '<path d="M7 4v16l13-8z"/>',
}


TONE_COLORS = {"": "#C4B5FD", "violet": "#C4B5FD", "cyan": "#22D3EE", "green": "#34D399", "amber": "#F59E0B",
               "gray": "#94A3B8", "red": "#F87171", "faint": "#64748B", "text": "#CBD5E1"}


def icon(name: str, tone: str = "") -> str:
    """Icon as an <img> with an SVG data URI (st.html sanitises inline <svg>, but allows data-URI images)."""
    color = TONE_COLORS.get(tone, tone or TONE_COLORS[""])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_ICONS.get(name, _ICONS["info"])}</svg>')
    return f'<img class="sn-icon" alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


def icon_box(name: str, tone: str = "") -> str:
    return f'<span class="sn-icon-box {tone}">{icon(name, tone)}</span>'


def esc(value) -> str:
    return html.escape(str(value))


def inject_css() -> None:
    """Load the design-system stylesheet once per run (st.html with only CSS takes no space)."""
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)   # comments are for humans; they could confuse the sanitiser
    st.html(f"<style>{css}</style>")


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------
def pill(text: str, tone: str = "", dot: bool = False, live: bool = False) -> str:
    cls = " ".join(c for c in ("sn-pill", tone, "live" if live else "") if c)
    return f'<span class="{cls}">{"<span class=sn-dot></span>" if dot else ""}{esc(text)}</span>'


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.html(f'<div class="sn-page-head"><span class="sn-eyebrow">{esc(eyebrow)}</span>'
            f'<h1>{esc(title)}</h1><p>{esc(description)}</p></div>')


def section(title: str, description: str | None = None, icon_name: str | None = None,
            right_html: str = "", tone: str = "") -> None:
    """Card heading: optional icon, title, muted description and right-aligned extras (pills)."""
    ico = icon_box(icon_name, tone) if icon_name else ""
    desc = f"<p>{esc(description)}</p>" if description else ""
    st.html(f'<div class="sn-section"><div><div class="sn-section-title">{ico}<h3>{esc(title)}</h3></div>{desc}</div>'
            f'<div>{right_html}</div></div>')


def notice(kind: str, title: str, body: str = "") -> None:
    """Friendly inline message. kind ∈ {ok, warn, bad, info}. ``body`` may contain simple HTML."""
    ico = {"ok": "check", "warn": "alert", "bad": "x", "info": "info"}.get(kind, "info")
    tone = {"ok": "green", "warn": "amber", "bad": "red", "info": "cyan"}.get(kind, "cyan")
    st.html(f'<div class="sn-notice {kind}">{icon(ico, tone)}<div><b>{esc(title)}</b>'
            f'{"<br>" + body if body else ""}</div></div>')


def empty_state(icon_name: str, title: str, body: str, flat_wave: bool = True) -> None:
    wave = '<div class="sn-flat" aria-hidden="true"></div>' if flat_wave else ""
    st.html(f'<div class="sn-empty">{icon_box(icon_name)}<h4>{esc(title)}</h4><p>{body}</p>{wave}</div>')


def stats(items: list[tuple[str, str]], mono: bool = True) -> None:
    """Responsive grid of compact values (label + value). ``mono`` for numbers, False for prose-like values."""
    cells = "".join(f'<div class="sn-stat"><div class="sn-label">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                    for k, v in items)
    st.html(f'<div class="sn-stats{"" if mono else " text"}">{cells}</div>')


def metric_cards(cards: list[dict]) -> None:
    """Row of KPI cards. Each dict: icon, label, value, unit, desc, tone, placeholder."""
    out = []
    for c in cards:
        cls = "sn-metric-value placeholder" if c.get("placeholder") else "sn-metric-value"
        unit = f'<small>{esc(c["unit"])}</small>' if c.get("unit") else ""
        out.append(
            f'<div class="sn-metric"><div class="sn-metric-top"><span class="sn-label">{esc(c["label"])}</span>'
            f'{icon_box(c["icon"], c.get("tone", ""))}</div>'
            f'<div class="{cls}">{esc(c["value"])}{unit}</div>'
            f'<div class="sn-metric-desc">{esc(c.get("desc", ""))}</div></div>')
    st.html(f'<div class="sn-metrics">{"".join(out)}</div>')


def player_label(text: str, color: str) -> None:
    st.html(f'<div class="sn-player-label"><span class="sn-swatch" style="background:{color}"></span>{esc(text)}</div>')


# --------------------------------------------------------------------------
# Composite components
# --------------------------------------------------------------------------
def _spectrum_bars(n: int = 56) -> str:
    """Decorative animated spectrum for the hero (deterministic shape, not data)."""
    x = np.linspace(0, 1, n)
    shape = 0.25 + 0.75 * np.abs(np.sin(9 * x) * np.exp(-((x - 0.42) ** 2) / 0.09)
                                 + 0.35 * np.sin(23 * x + 1.3) * np.exp(-((x - 0.7) ** 2) / 0.05))
    shape = np.clip(shape / shape.max(), 0.08, 1.0)
    return "".join(f'<i style="--h:{h * 100:.0f}%;--d:{-(i % 9) * 0.19:.2f}s"></i>' for i, h in enumerate(shape))


def hero(status_html: str) -> None:
    st.html(f"""
<section class="sn-hero">
  <div>
    <span class="sn-eyebrow">AI Audio Intelligence</span>
    <h1>Hear the Signal.<br><span class="sn-grad">Remove the Noise.</span></h1>
    <p class="sn-lead">Analyze, enhance, and understand audio with intelligent signal processing and machine learning.</p>
    <div class="sn-hero-meta">{status_html}</div>
    <div class="sn-features">
      <div class="sn-feature"><b>{icon("bars", "#A78BFA")}FFT analysis</b><span>Spectrum, dominant frequency and STFT spectrogram.</span></div>
      <div class="sn-feature"><b>{icon("filter", "#A78BFA")}DSP enhancement</b><span>Butterworth filtering and spectral subtraction.</span></div>
      <div class="sn-feature"><b>{icon("sparkles", "#A78BFA")}ML processing</b><span>Trained spectral-mask model, measured honestly.</span></div>
    </div>
  </div>
  <div class="sn-visual" aria-hidden="true">
    <div class="sn-visual-head"><span class="sn-label">Spectral engine</span>{pill("STFT · 512 / 128", "violet")}</div>
    <div class="sn-eq">{_spectrum_bars()}</div>
    <div class="sn-visual-axis"><span>0 Hz</span><span>2 kHz</span><span>4 kHz</span><span>6 kHz</span><span>8 kHz</span></div>
  </div>
</section>""")


def file_card(name: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="sn-chip">{esc(c)}</span>' for c in chips)
    st.html(f'<div class="sn-file">{icon_box("file", "cyan")}<div style="min-width:0">'
            f'<div class="sn-file-name">{esc(name)}</div><div class="sn-chips">{chip_html}</div></div></div>')


def upload_header(max_mb: int, formats: list[str]) -> None:
    st.html(f'<div class="sn-upload-head">{icon_box("upload")}<div><h3>Drop your audio here</h3>'
            f'<p>{esc(", ".join(f.upper() for f in formats))} · up to {max_mb} MB · first 5 minutes analysed</p></div></div>')


def steps(items: list[tuple[str, str, str]], start: int = 1) -> None:
    """Workflow list. Each item: (title, description, state) with state ∈ {done, active, ''}."""
    rows = "".join(f'<div class="sn-step {state}"><div class="sn-step-n">{"✓" if state == "done" else i}</div>'
                   f'<div><b>{esc(t)}</b><span>{esc(d)}</span></div></div>'
                   for i, (t, d, state) in enumerate(items, start))
    st.html(f'<div class="sn-steps">{rows}</div>')


def pipeline(nodes: list[tuple[str, str, str, str]]) -> None:
    """ML pipeline diagram. Each node: (icon, title, caption, state) with state ∈ {done, ready, locked}."""
    tones = {"done": "violet", "ready": "text", "locked": "faint"}
    cells = "".join(f'<div class="sn-node {state}"><div class="sn-node-dot">{icon(ic, tones.get(state, "text"))}</div>'
                    f'<b>{esc(title)}</b><span>{esc(caption)}</span></div>' for ic, title, caption, state in nodes)
    st.html(f'<div class="sn-pipe">{cells}</div>')


def results_table(df, colors: dict[str, str], has_reference: bool) -> None:
    """Comparison table with right-aligned monospaced numbers.

    "Best" is highlighted only for reference-based metrics, only when a clean
    reference exists, and only among the enhancement methods (not the input).
    """
    # (data column, short header, number format). Correlation and segmental SNR stay in the CSV/report.
    cols = [("Method", "Method", None), ("Processing time (s)", "Time (s)", "{:.3f}"), ("Output RMS", "RMS", "{:.4f}"),
            ("RMSE vs reference", "RMSE", "{:.4f}"), ("Input SNR (dB)", "Input SNR (dB)", "{:.2f}"),
            ("Output SNR (dB)", "Output SNR (dB)", "{:.2f}"), ("SNR improvement (dB)", "Δ SNR (dB)", "{:+.2f}")]
    best = {}
    if has_reference and len(df) > 2:
        enh = df.iloc[1:]
        for col, better in (("RMSE vs reference", "min"), ("Output SNR (dB)", "max"), ("SNR improvement (dB)", "max")):
            vals = enh[col].dropna()
            if len(vals) > 1:
                best[col] = vals.idxmin() if better == "min" else vals.idxmax()
    head = "".join(f"<th>{esc(h)}</th>" for _, h, _ in cols)
    body = []
    for idx, row in df.iterrows():
        cells = []
        for c, _, fmt in cols:
            v = row.get(c)
            if c == "Method":
                cells.append(f'<td class="method"><span class="sn-swatch" style="background:{colors.get(v, "#94A3B8")}">'
                             f'</span>{esc(v)}</td>')
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append('<td class="na">N/A</td>')
            else:
                cls = ' class="best"' if best.get(c) == idx else ""
                cells.append(f"<td{cls}>{fmt.format(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.html(f'<div class="sn-table-wrap"><table class="sn-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def download_header(icon_name: str, title: str, filename: str, meta: str, available: bool) -> None:
    st.html(f'<div class="sn-dl{"" if available else " off"}">{icon_box(icon_name, "" if available else "gray")}'
            f'<div><b>{esc(title)}</b><code>{esc(filename).replace("_", "_<wbr>")}</code>'   # break after underscores
            f'<code>{esc(meta)}</code></div></div>')


def sidebar_status(rows: list[tuple[str, str]], footer: str) -> None:
    body = "".join(f'<div class="sn-side-row"><span>{esc(k)}</span>{v}</div>' for k, v in rows)
    st.html(f'<div class="sn-label" style="margin:.3rem 0 .1rem">System status</div>'
            f'<div class="sn-side-status">{body}</div><div class="sn-side-foot">{esc(footer)}</div>')


def simple_table(headers: list[str], rows: list[list], first_col_left: bool = True) -> None:
    """Compact table with right-aligned monospaced cells (values are pre-formatted strings)."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(
        f'<td class="{"method" if (i == 0 and first_col_left) else ("na" if str(v) == "N/A" else "")}">{esc(v)}</td>'
        for i, v in enumerate(r)) + "</tr>" for r in rows)
    st.html(f'<div class="sn-table-wrap"><table class="sn-table" style="min-width:520px"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')
