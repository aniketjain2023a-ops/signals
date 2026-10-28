import numpy as np
import pytest

import config
import ml_enhancement as ml
import noise_simulation as ns
import signal_analysis as sa
from conftest import SR


def test_feature_dimensions_match_config(speech):
    cfg = ml.FeatureConfig()
    mag = sa.stft(speech, SR).magnitude
    X = ml.build_features(mag, cfg)
    assert X.shape == (mag.shape[1], cfg.n_features)
    assert cfg.n_features == config.ML_N_BANDS * (2 * config.ML_CONTEXT_FRAMES + 1) + config.ML_N_BANDS
    assert np.all(np.isfinite(X))


def test_context_window_edge_handling():
    cfg = ml.FeatureConfig()
    fb, _ = ml.mel_filterbank(cfg)
    mag = np.abs(np.random.default_rng(0).standard_normal((cfg.n_fft // 2 + 1, 10))) + 0.1
    X = ml.build_features(mag, cfg, fb)
    B, c = cfg.n_bands, cfg.context
    centre = X[:, c * B:(c + 1) * B]
    # the left-most context block of frame 0 must equal frame 0 itself (edge repeat)
    assert np.allclose(X[0, :B], centre[0])
    # the left context of frame t is frame t-c
    assert np.allclose(X[5, :B], centre[5 - c])
    # the right-most block of the last frame repeats the last frame
    assert np.allclose(X[-1, 2 * c * B:(2 * c + 1) * B], centre[-1])


def test_targets_bounded(speech):
    cfg = ml.FeatureConfig()
    fb, _ = ml.mel_filterbank(cfg)
    noisy, _ = ns.add_noise_at_snr(speech, 0, seed=0)
    T = ml.build_targets(sa.stft(speech, SR).magnitude, sa.stft(noisy, SR).magnitude, fb)
    assert T.shape[1] == cfg.n_bands
    assert T.min() >= 0 and T.max() <= 1


def test_unity_mask_reconstructs_input(speech):
    cfg = ml.FeatureConfig()
    y, info = ml.apply_mask(speech, SR, lambda X: np.ones((X.shape[0], cfg.n_bands)), cfg)
    assert y.size == speech.size
    assert np.max(np.abs(y - speech)) < 1e-8


def test_missing_model_reports_unavailable(tmp_path):
    enh, msg = ml.model_status(tmp_path / "does_not_exist.joblib")
    assert enh is None
    assert "No trained model" in msg
    with pytest.raises(ml.ModelUnavailableError):
        ml.SpeechEnhancer.load(tmp_path / "nope.joblib")


def test_corrupt_model_reports_unavailable(tmp_path):
    p = tmp_path / "bad.joblib"
    p.write_bytes(b"garbage")
    enh, msg = ml.model_status(p)
    assert enh is None and "could not be loaded" in msg


def test_mismatched_feature_config_rejected(tiny_model):
    import joblib
    bundle = joblib.load(tiny_model)
    bundle["feature_config"] = {**bundle["feature_config"], "n_bands": 16}
    with pytest.raises(ml.ModelUnavailableError, match="different"):
        ml.SpeechEnhancer(bundle)


def test_trained_model_inference(tiny_model, speech):
    enh = ml.SpeechEnhancer.load(tiny_model)
    assert enh.trained_on_synthetic
    noisy, _ = ns.add_noise_at_snr(speech, 5, seed=0)
    r = enh.enhance(noisy, SR)
    assert r.output.size == noisy.size
    assert np.all(np.isfinite(r.output))
    assert r.input_feature_shape[1] == enh.cfg.n_features
    assert r.output_feature_shape[1] == enh.cfg.n_bands
    assert all(v == "done" for k, v in r.stage_status.items() if k != "Performance evaluation")
    # mask ≤ 1 means the output can never be louder than the input spectrum
    assert np.sqrt(np.mean(r.output ** 2)) <= np.sqrt(np.mean(noisy ** 2)) * 1.01


def test_wrong_sample_rate_rejected(tiny_model, speech):
    enh = ml.SpeechEnhancer.load(tiny_model)
    with pytest.raises(ValueError, match="16000"):
        enh.enhance(speech, 22_050)


def test_training_summary_written(tiny_model):
    import json
    s = json.loads(tiny_model.with_name("training_summary.json").read_text())
    assert s["data_source"] == "synthetic" and "warning" in s
    assert s["n_validation_recordings"] >= 1
    assert "per_input_snr" in s["validation"]


def test_train_on_empty_dir_fails_clearly(tmp_path):
    import train_model as tm
    with pytest.raises(FileNotFoundError, match="No .wav or .flac"):
        tm.train(data_dir=tmp_path, output=tmp_path / "m.joblib", log=lambda *_: None)
    assert tm.main(["--data-dir", str(tmp_path / "missing")]) == 1


def test_train_on_wav_dataset_splits_by_speaker(tmp_path):
    import soundfile as sf
    import train_model as tm
    from conftest import speech_like
    for spk in ("spk1", "spk2", "spk3"):
        (tmp_path / spk).mkdir()
        for i in range(2):
            sf.write(tmp_path / spk / f"{i}.wav", speech_like(1.0, seed=i), SR)
    meta = tm.train(data_dir=tmp_path, max_frames=3000, n_estimators=3, max_depth=4, snrs=(5.0,),
                    noise_types=("white",), output=tmp_path / "m.joblib",
                    summary_path=tmp_path / "s.json", log=lambda *_: None)
    assert meta["data_source"] == "dataset"
    assert meta["split"] == "split by speaker folder"
    assert meta["n_validation_recordings"] == 2 and meta["n_train_recordings"] == 4


def test_discover_files_spreads_across_speakers(tmp_path):
    import train_model as tm
    for spk in ("a", "b", "c"):
        (tmp_path / spk / "chapter").mkdir(parents=True)
        for i in range(10):
            (tmp_path / spk / "chapter" / f"{i}.flac").write_bytes(b"x")
    picked = tm.discover_files(tmp_path, 6)
    assert len(picked) == 6
    assert sorted({tm.group_key(p, tmp_path) for p in picked}) == ["a", "b", "c"]
    assert len(tm.discover_files(tmp_path, 100)) == 30
