"""ML Enhancement page: pipeline status, model details, output, validation results and training."""

from __future__ import annotations

import json

import streamlit as st

import app_state as state
import config
import ml_enhancement as ml
import pipeline as pl
import plots
import ui

# Six display stages, each backed by one or more real stage flags reported by SpeechEnhancer.enhance().
STAGES = [
    ("mic", "Audio input", ["Audio input"]),
    ("layers", "Preprocessing", ["Audio preprocessing"]),
    ("grid", "Feature extraction", ["STFT feature extraction", "Noisy-feature preparation"]),
    ("sparkles", "Model inference", ["Trained-model inference"]),
    ("rebuild", "Reconstruction", ["Estimated clean spectral magnitude", "Waveform reconstruction", "Output validation"]),
    ("chart", "Evaluation", ["Performance evaluation"]),
]


def _training_summary() -> dict | None:
    try:
        return json.loads(config.TRAINING_SUMMARY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def pipeline_card(enh, er) -> None:
    if er is not None:
        nodes = [(ic, title, "Done" if all(er.stage_status.get(f) == "done" for f in flags) else "Not reached",
                  "done" if all(er.stage_status.get(f) == "done" for f in flags) else "ready")
                 for ic, title, flags in STAGES]
        status = ui.pill("Last run completed", "ok", dot=True)
    elif enh is not None:
        nodes = [(ic, title, "Ready", "ready") for ic, title, _ in STAGES]
        status = ui.pill("Ready · not run yet", "violet", dot=True)
    else:
        nodes = [(ic, title, "Ready" if i < 3 else "Needs model", "ready" if i < 3 else "locked")
                 for i, (ic, title, _) in enumerate(STAGES)]
        status = ui.pill("Model required", "warn", dot=True)
    with st.container(key="card-ml-pipeline"):
        ui.section("Inference pipeline", "Noisy STFT band features → trained Random Forest → per-band gain mask → "
                                         "estimated clean magnitude → inverse STFT with the noisy phase.",
                   "sparkles", right_html=status)
        ui.pipeline(nodes)
        ui.metric_cards([
            dict(icon="cpu", label="Model", value="Random Forest" if enh else "—", placeholder=enh is None,
                 desc=enh.model_type if enh else "No trained model loaded"),
            dict(icon="check", label="Availability", tone="green" if enh else "gray",
                 value=("Synthetic" if enh.trained_on_synthetic else "Loaded") if enh else "Not trained",
                 placeholder=enh is None,
                 desc=("Pipeline test model" if enh.trained_on_synthetic else "Trained on clean speech") if enh
                 else "Train with train_model.py"),
            dict(icon="clock", label="Inference time", tone="cyan",
                 value=f"{er.processing_time_s:.2f}" if er else "—", unit="s" if er else None, placeholder=er is None,
                 desc="Features → mask → waveform" if er else "Run ML in Noise Reduction"),
            dict(icon="grid", label="Feature matrix", value=f"{er.input_feature_shape[0]:,} × {er.input_feature_shape[1]}"
                 if er else "—", placeholder=er is None,
                 desc=f"frames × features → mask {er.output_feature_shape[1]} bands" if er else "Shown after inference"),
        ])


def model_card(enh) -> None:
    with st.container(key="card-ml-model"):
        if enh is None:
            ui.section("Set up the model", "ML enhancement is optional – the DSP pipeline works without it.", "lock",
                       tone="amber")
            ui.steps([
                ("Collect clean speech", "e.g. LibriSpeech dev-clean, VCTK or your own quiet recordings", "active"),
                ("Train the model", "python train_model.py --data-dir <folder>", ""),
                ("Come back here", "The model is detected automatically", ""),
            ])
            st.code("python train_model.py --data-dir ~/Downloads/LibriSpeech/dev-clean --max-files 400", language="bash")
            return
        md = enh.metadata
        src = md.get("data_source")
        ui.section("Model details", "Read from the saved model bundle – nothing on this card is estimated.", "cpu",
                   right_html=ui.pill("Synthetic data", "warn", dot=True) if src == "synthetic"
                   else ui.pill("Real speech", "ok", dot=True))
        hp = md.get("hyperparameters", {})
        ui.stats([
            ("Model", md.get("model_name", enh.name)),
            ("Type", md.get("model_type", enh.model_type)),
            ("Trained", str(md.get("trained_at", "unknown")).replace("T", " ")),
            ("Recordings", f"{md.get('n_train_recordings', '?')} train / {md.get('n_validation_recordings', '?')} held out"),
            ("Split", md.get("split", "unknown")),
            ("Features → outputs", f"{md.get('n_features', '?')} → {md.get('n_outputs', '?')}"),
            ("Training frames", f"{md.get('n_training_frames', 0):,}"),
            ("Trees / depth", f"{hp.get('n_estimators', '?')} / {hp.get('max_depth', '?')}"),
        ], mono=False)
        if src == "synthetic":
            ui.notice("warn", "Trained on synthetic signals",
                      "This verifies that the ML pipeline runs end to end. It does not establish real-world "
                      "speech-enhancement performance – train on real clean speech for meaningful results.")


def output_card(audio, res, er, enh) -> None:
    with st.container(key="card-ml-output"):
        ui.section("Enhanced output", "The ML result for the current recording.", "play", tone="cyan")
        if er is None:
            if enh is None:
                ui.empty_state("lock", "Locked until a model is trained",
                               "Once a model exists, choose <b>ML enhancement</b> or <b>Compare all methods</b> in the "
                               "Noise Reduction lab.", flat_wave=False)
            elif audio is None:
                ui.empty_state("upload", "No audio loaded", "Upload a recording on the Overview page first.", flat_wave=False)
            else:
                ui.empty_state("play", "Not run yet", "Choose <b>ML enhancement</b> or <b>Compare all methods</b> in "
                                                      "the Noise Reduction lab and press Run.", flat_wave=False)
                st.page_link(state.PAGES["lab"], label="Open noise reduction lab", icon=":material/tune:")
            return
        ui.player_label("ML enhancement", plots.color_for(pl.METHOD_ML))
        st.audio(state.wav_for(pl.METHOD_ML, er.output, audio.sr), format="audio/wav")
        st.plotly_chart(plots.before_after_waveform(res.input_signal, er.output, audio.sr, res.input_name,
                                                    pl.METHOD_ML, height=300),
                        width="stretch", config=plots.CONFIG, key="ml_ba")
        row = res.table[res.table["Method"] == pl.METHOD_ML]
        if res.has_reference and not row.empty:
            r = row.iloc[0]
            ui.stats([("Input SNR", f"{r['Input SNR (dB)']:.2f} dB"), ("Output SNR", f"{r['Output SNR (dB)']:.2f} dB"),
                      ("SNR improvement", f"{r['SNR improvement (dB)']:+.2f} dB"), ("RMSE", f"{r['RMSE vs reference']:.4f}")])
        for w in er.warnings:
            ui.notice("warn", "Output check", ui.esc(w))


def validation_card(enh) -> None:
    summary = _training_summary() if enh else None
    val = (summary or {}).get("validation", {})
    if not val.get("per_input_snr"):
        return
    with st.container(key="card-ml-validation"):
        ui.section("Held-out validation", f"Measured by train_model.py on recordings never seen in training "
                                          f"({val.get('noise_type', 'white')} noise). Mask R² vs a mean predictor: "
                                          f"{val.get('mask_r2_vs_mean_predictor')}.", "chart")
        rows = [[f"{round(r['input_snr'], 1) + 0.0:.1f} dB",          # + 0.0 turns -0.0 into 0.0 f"{r['ml_out_snr']:.2f}", f"{r['ml_improvement']:+.2f}",
                 f"{r['specsub_out_snr']:.2f}", f"{r['specsub_improvement']:+.2f}",
                 f"{r['lowpass_out_snr']:.2f}", f"{r['lowpass_improvement']:+.2f}"]
                for r in val["per_input_snr"].values()]
        ui.simple_table(["Input SNR", "ML out (dB)", "ML Δ", "Spec. sub. out", "Spec. sub. Δ", "Low-pass out",
                         "Low-pass Δ"], rows)


def limitations_card() -> None:
    with st.container(key="card-ml-limits"):
        ui.section("Limitations", "What this baseline can and cannot do.", "info", tone="gray")
        st.markdown(
            "- Only the **magnitude** is estimated; the noisy **phase** is reused, so some roughness remains at low SNR.\n"
            "- A Random Forest on 32 mel bands is far smaller than modern deep-learning enhancers (DCCRN, Demucs).\n"
            "- It generalises only to speech and noise similar to its training data.\n"
            "- A Butterworth filter is **not** machine learning – it is listed separately as DSP.")


def training_actions(enh) -> None:
    n_files = ml.training_data_available()
    with st.expander("Train or retrain from the dashboard", icon=":material/model_training:", expanded=enh is None):
        st.caption("Training on a large dataset is faster from the terminal (see README). The quick synthetic model "
                   "only tests the pipeline.")
        c1, c2 = st.columns(2)
        quick = c1.button("Quick synthetic demo model (~1 min)", icon=":material/bolt:", width="stretch", key="train_quick")
        real = c2.button(f"Train on data/clean ({n_files} files)", icon=":material/library_music:", width="stretch",
                         disabled=n_files < 2, key="train_real")
        if quick or real:
            import train_model as tm
            with st.status("Training model…", expanded=True) as box:
                try:
                    kwargs = (dict(synthetic=True, n_synthetic=24, max_frames=30_000, n_estimators=25) if quick
                              else dict(data_dir=config.CLEAN_SPEECH_DIR))
                    tm.train(log=box.write, **kwargs)
                except (FileNotFoundError, ValueError) as exc:
                    box.update(label="Training failed", state="error")
                    ui.notice("bad", "Training failed", ui.esc(exc))
                    return
                box.update(label="Training finished", state="complete")
            state.clear_model_cache()
            state.clear_results()
            st.session_state["flash"] = "Model trained and loaded"
            st.rerun()


def render() -> None:
    ui.page_header("Approach B · Machine learning", "ML enhancement",
                   "A supervised Random Forest learns, from noisy and clean speech pairs, how much to attenuate each "
                   "frequency band. It is trained offline and evaluated on held-out recordings.")
    enh, _ = state.enhancer()
    audio, _ = state.current_audio()
    res = state.result()
    fresh = audio is not None and res is not None and not state.is_stale(audio)
    er = res.details.get(pl.METHOD_ML) if fresh else None
    pipeline_card(enh, er)
    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        model_card(enh)
    with c2:
        output_card(audio, res, er, enh)
    validation_card(enh)
    limitations_card()
    training_actions(enh)
