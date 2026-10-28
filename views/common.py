"""Helpers shared by several pages."""

from __future__ import annotations

import streamlit as st

import app_state as state
import ui


def require_audio():
    """Return the loaded AudioData, or render a helpful empty/error state and return None."""
    audio, error = state.current_audio()
    if error:
        ui.notice("bad", "This file could not be read", ui.esc(error) + " Upload a different recording on the Overview page.")
        st.page_link(state.PAGES["overview"], label="Go to upload", icon=":material/upload:")
        return None
    if audio is None:
        ui.empty_state("upload", "No audio loaded yet",
                       "Upload a speech recording on the Overview page. Every chart and metric here is computed "
                       "from your real audio – nothing is shown until there is something to measure.")
        st.page_link(state.PAGES["overview"], label="Go to upload", icon=":material/upload:")
        return None
    return audio


def default_kw(key: str, param: str = "value") -> dict:
    """Pass a widget default only the first time; afterwards the persisted session value is used.

    (Passing both a default and a session-state value makes Streamlit warn.)
    """
    return {} if key in st.session_state else {param: state.DEFAULTS[key]}


def flash(message: str) -> None:
    st.session_state["flash"] = message


def show_flash() -> None:
    msg = st.session_state.pop("flash", None)
    if msg:
        st.toast(msg, icon=":material/check_circle:")
