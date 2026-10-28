"""Reports page: download centre for processed audio, the results table and the HTML report."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import app_state as state
import pipeline as pl
import report_generator as rg
import ui


def _build_report(audio, res: pl.PipelineResult) -> bytes:
    enh, _ = state.enhancer()
    sigs = {"Clean reference": res.reference} if res.has_reference else {}
    sigs[res.input_name] = res.input_signal
    sigs.update(res.outputs)
    if enh:
        md = enh.metadata
        ml_info = {"Model": enh.name, "Type": enh.model_type, "Training data": md.get("data_source"),
                   "Training recordings": md.get("n_train_recordings"),
                   "Validation recordings": md.get("n_validation_recordings"),
                   "Features / outputs": f"{md.get('n_features')} / {md.get('n_outputs')}",
                   "Used in this run": pl.METHOD_ML in res.outputs}
        if md.get("warning"):
            ml_info["Warning"] = md["warning"]
    else:
        ml_info = {"Status": "No trained model available – ML enhancement not run."}
    return rg.build_report(metadata=audio.metadata(), settings=res.settings.as_dict(), results_df=res.table,
                           signals=sigs, sr=audio.sr, ml_info=ml_info, has_reference=res.has_reference,
                           processing_log=audio.processing_log).encode("utf-8")


def _card(key: str, icon_name: str, title: str, filename: str, data: bytes | None, mime: str, reason: str) -> None:
    available = data is not None
    with st.container(key=f"card-dl-{key}"):
        meta = f"{mime.split('/')[-1].upper()} · {state.human_size(len(data))}" if available else reason
        ui.download_header(icon_name, title, filename if available else "—", meta, available)
        st.download_button("Download", data or b"", file_name=filename, mime=mime, disabled=not available,
                           icon=":material/download:", width="stretch", key=f"dl_{key}",
                           type="primary" if available else "secondary")


def render() -> None:
    ui.page_header("Reports", "Download center",
                   "Export the processed audio, the measured results and a self-contained analysis report. Files are "
                   "generated from the latest processing run.")
    audio, _ = state.current_audio()
    res = state.result()
    fresh = audio is not None and res is not None and not state.is_stale(audio)
    if audio is None:
        ui.notice("info", "Nothing to export yet", "Upload a recording on the Overview page, then run a method in the "
                                                   "Noise Reduction lab.")
    elif res is None:
        ui.notice("info", "No processing run yet", "Run a method in the Noise Reduction lab to generate outputs.")
        st.page_link(state.PAGES["lab"], label="Open noise reduction lab", icon=":material/tune:")
    elif not fresh:
        ui.notice("warn", "Settings changed since the last run",
                  "Downloads still reflect the previous run. Run again in the Noise Reduction lab to refresh them.")
    else:
        ui.notice("ok", f"{len(res.outputs)} output{'s' if len(res.outputs) != 1 else ''} ready",
                  f"WAV files are 16-bit PCM at {audio.sr:,} Hz. Signals above full scale are scaled down, never clipped.")

    have = audio is not None and res is not None
    stem = Path(audio.filename).stem if audio else "audio"
    outputs = res.outputs if have else {}

    def audio_data(name, signal):
        return state.wav_for(name, signal, audio.sr) if signal is not None else None

    def why(method: str) -> str:
        if not have:
            return "Run processing first"
        if method in res.errors:
            return "Failed in the last run"
        if method == pl.METHOD_ML and state.enhancer()[0] is None:
            return "No trained ML model"
        return "Not selected in the last run"

    st.html('<div class="sn-label" style="margin:.6rem 0 .6rem">Enhanced audio</div>')
    cols = st.columns(3, gap="medium")
    for col, (method, ic) in zip(cols, [(pl.METHOD_LOWPASS, "filter"), (pl.METHOD_SPECSUB, "bars"),
                                       (pl.METHOD_ML, "sparkles")]):
        with col:
            sig = outputs.get(method)
            _card(state.slug(method), ic, method, f"{stem}_{state.slug(method)}.wav", audio_data(method, sig),
                  "audio/wav", why(method))

    st.html('<div class="sn-label" style="margin:1rem 0 .6rem">Test signals & results</div>')
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        noisy = res.input_signal if have and res.settings.mode == "simulate" else None
        _card("noisy", "wave", "Noisy test audio",
              f"{stem}_noisy_{res.settings.snr_db:g}dB.wav" if noisy is not None else "noisy.wav",
              audio_data("Noisy input", noisy), "audio/wav",
              "Only in simulated-noise mode" if have else "Run processing first")
    with c2:
        csv = res.table.to_csv(index=False, na_rep="N/A").encode() if have else None
        _card("csv", "table", "Results table", f"{stem}_results.csv", csv, "text/csv", "Run processing first")
    with c3:
        report = st.session_state.get("report_html") if fresh else None
        with st.container(key="card-dl-report"):
            meta = (f"HTML · {state.human_size(len(report))}" if report else
                    "Generate it from the latest run" if fresh else "Needs an up-to-date run")
            ui.download_header("report", "Analysis report", f"{stem}_report.html" if report else "—", meta,
                               report is not None or fresh)
            if report:
                st.download_button("Download", report, file_name=f"{stem}_report.html", mime="text/html",
                                   icon=":material/download:", width="stretch", type="primary", key="dl_report")
            elif st.button("Generate report", icon=":material/auto_awesome:", width="stretch", disabled=not fresh,
                           key="gen_report"):
                with st.spinner("Rendering figures and tables…"):
                    st.session_state.report_html = _build_report(audio, res)
                st.rerun()
    st.caption("The report contains input metadata, settings, methods, ML model details, the measured metrics, "
               "figures, an interpretation and limitations. Open it in a browser and use Print → Save as PDF for a PDF.")
