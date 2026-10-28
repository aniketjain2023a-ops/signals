"""
SONORA – Intelligent Audio Lab.
AI-Based Speech Signal Analysis and Noise Reduction Using FFT, Digital Filtering, and Machine Learning.

Run with:   streamlit run app.py

Structure
---------
app.py          page config, theme, brand, navigation, sidebar status
app_state.py    session state, cached analysis, running the pipeline
ui.py           reusable HTML components (cards, pills, tables, empty states)
plots.py        Plotly figures
views/*.py      one module per page

All signal processing lives in the tested modules (audio_utils, signal_analysis,
dsp_enhancement, ml_enhancement, metrics, pipeline). The app starts and the DSP
pipeline works even when no trained ML model exists.
"""

from __future__ import annotations

import streamlit as st

import app_state as state
import config
import ui
from views import analysis, common, experiments, lab, ml_page, overview, reports

st.set_page_config(page_title=f"{config.PROJECT_SHORT_NAME} · Intelligent Audio Lab", page_icon=":material/graphic_eq:",
                   layout="wide", initial_sidebar_state="auto")  # auto: open on desktop, collapsed on phones
state.init()
ui.inject_css()
st.logo(str(ui.ASSETS / "sonora_logo.svg"), size="large", icon_image=str(ui.ASSETS / "sonora_icon.svg"))

# Real navigation: every entry is a working page (st.navigation renders the sidebar menu
# with the active-page indicator and gives each page its own URL).
PAGES = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/space_dashboard:",
                        url_path="overview", default=True),
    "analysis": st.Page(analysis.render, title="Audio Analysis", icon=":material/graphic_eq:", url_path="analysis"),
    "lab": st.Page(lab.render, title="Noise Reduction", icon=":material/tune:", url_path="noise-reduction"),
    "ml": st.Page(ml_page.render, title="ML Enhancement", icon=":material/neurology:", url_path="ml-enhancement"),
    "experiments": st.Page(experiments.render, title="Experiments", icon=":material/science:", url_path="experiments"),
    "reports": st.Page(reports.render, title="Reports", icon=":material/description:", url_path="reports"),
}
state.PAGES.update(PAGES)
current = st.navigation(list(PAGES.values()), position="sidebar")


def sidebar_status() -> None:
    audio, error = state.current_audio()
    enh, _ = state.enhancer()
    res = state.result()
    if error:
        app = ui.pill("Invalid file", "bad", dot=True)
    elif audio is None:
        app = ui.pill("Ready", "ok", dot=True, live=True)
    elif res is not None and not state.is_stale(audio):
        app = ui.pill("Processed", "ok", dot=True)
    else:
        app = ui.pill("Audio loaded", "cyan", dot=True)
    ml_pill = (ui.pill("Synthetic model", "warn", dot=True) if enh and enh.trained_on_synthetic
               else ui.pill("Model loaded", "ok", dot=True) if enh else ui.pill("Not trained", "", dot=True))
    footer = (f"{audio.filename} · {audio.duration_s:.1f} s · {audio.sr // 1000} kHz" if audio
              else "Upload a recording on the Overview page to begin.")
    with st.sidebar:
        ui.sidebar_status([("Application", app), ("DSP engine", ui.pill("Online", "ok", dot=True)),
                           ("ML model", ml_pill)], footer)


sidebar_status()
common.show_flash()
current.run()
