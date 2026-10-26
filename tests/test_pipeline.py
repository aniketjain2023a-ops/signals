"""End-to-end tests: pipeline, report and a headless run of the Streamlit app."""

import numpy as np
import pytest

import ml_enhancement as ml
import pipeline as pl
import report_generator as rg
from conftest import SR


def test_simulated_pipeline_all_dsp(speech):
    s = pl.PipelineSettings(mode="simulate", snr_db=5, methods=(pl.METHOD_LOWPASS, pl.METHOD_SPECSUB))
    r = pl.run(speech, SR, s)
    assert not r.errors
    assert r.has_reference
    assert set(r.outputs) == {pl.METHOD_LOWPASS, pl.METHOD_SPECSUB}
    for out in r.outputs.values():
        assert out.size == speech.size and np.all(np.isfinite(out))
    noisy_row = r.table[r.table["Method"] == "Noisy"].iloc[0]
    assert noisy_row["Input SNR (dB)"] == pytest.approx(5.0, abs=1e-6)
    assert noisy_row["SNR improvement (dB)"] == pytest.approx(0.0, abs=1e-9)


def test_ml_requested_without_model_reports_error(speech):
    r = pl.run(speech, SR, pl.PipelineSettings(methods=(pl.METHOD_ML, pl.METHOD_LOWPASS)), enhancer=None)
    assert pl.METHOD_ML in r.errors and pl.METHOD_LOWPASS in r.outputs


def test_real_mode_without_reference_has_no_snr(speech):
    r = pl.run(speech, SR, pl.PipelineSettings(mode="real", methods=(pl.METHOD_SPECSUB,)))
    assert not r.has_reference and r.noise is None
    assert r.table["Output SNR (dB)"].isna().all()


def test_real_mode_with_reference(speech):
    noisy = speech + 0.05 * np.random.default_rng(0).standard_normal(speech.size)
    r = pl.run(noisy, SR, pl.PipelineSettings(mode="real", methods=(pl.METHOD_SPECSUB,)), reference=speech)
    assert r.has_reference
    assert np.isfinite(r.table["SNR improvement (dB)"].iloc[1])


def test_pipeline_with_ml_and_report(speech, tiny_model):
    enh = ml.SpeechEnhancer.load(tiny_model)
    s = pl.PipelineSettings(methods=pl.ALL_METHODS)
    r = pl.run(speech, SR, s, enhancer=enh)
    assert not r.errors and pl.METHOD_ML in r.outputs
    assert r.details[pl.METHOD_ML].stage_status["Performance evaluation"] == "done"
    html = rg.build_report(metadata={"Filename": "x.wav"}, settings=s.as_dict(), results_df=r.table,
                           signals={"Clean reference": r.reference, "Noisy": r.input_signal, **r.outputs},
                           sr=SR, ml_info={"Model": enh.name}, has_reference=True, processing_log=[])
    assert "<html" in html and "data:image/png;base64" in html
    assert "SNR improvement" in html


def test_report_without_reference_says_na(speech):
    r = pl.run(speech, SR, pl.PipelineSettings(mode="real", methods=(pl.METHOD_LOWPASS,)))
    notes = rg.interpret(r.table, has_reference=False)
    assert "No clean reference" in notes[0]


def test_app_starts_without_model(monkeypatch, tmp_path):
    """The dashboard must start even when no ML model exists."""
    from streamlit.testing.v1 import AppTest
    import config
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "none.joblib")
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert any("Upload a speech recording" in i.value for i in at.info)
