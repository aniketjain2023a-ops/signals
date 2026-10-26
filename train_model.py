"""
Train the baseline ML speech-enhancement model.

Usage
-----
Real speech (recommended):
    python train_model.py --data-dir data/clean

Pipeline test with synthetic speech-like signals (NOT a real-world model):
    python train_model.py --synthetic

Workflow
--------
1. Find clean WAV/FLAC files in --data-dir (searched recursively).
2. Validate, convert to mono, resample to 16 kHz, peak-normalise.
3. Split into TRAINING and VALIDATION sets *by recording* (and by speaker if
   the files are organised as data/clean/<speaker>/<file>.wav), so frames
   from one recording never appear in both sets.
4. For every training recording, create noisy copies at several SNRs and
   noise colours (seeded, repeatable).
5. Extract STFT band features (ml_enhancement.build_features – exactly the
   same code used at inference) and band ratio-mask targets.
6. Fit a RandomForestRegressor on a bounded random subsample of frames
   (--max-frames) to keep memory and time manageable on a laptop.
7. Evaluate on the held-out recordings: frame-level mask R² plus full
   waveform reconstruction SNR, compared with the DSP baselines.
8. Save the model bundle (models/speech_enhancer.joblib) and a JSON summary
   (models/training_summary.json).

Memory: features are computed one recording at a time and only a random
subset of frames per recording is kept, so peak memory is roughly
``max_frames * n_features * 4 bytes`` (≈ 46 MB for the defaults) plus one
recording.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.signal import butter, sosfilt
from sklearn.ensemble import RandomForestRegressor

import audio_utils as au
import config
import dsp_enhancement as dsp
import metrics
import ml_enhancement as ml
import noise_simulation as ns
import signal_analysis as sa

EVAL_SNRS_DB = (0.0, 5.0, 10.0)


# --------------------------------------------------------------------------
# Data discovery and loading
# --------------------------------------------------------------------------
def discover_files(data_dir: Path, max_files: int) -> list[Path]:
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"})
    return files[:max_files]


def group_key(path: Path, data_dir: Path) -> str:
    """Speaker/group id = first sub-folder under data_dir, else the file itself."""
    rel = path.relative_to(data_dir)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def load_clean(path: Path, max_seconds: float) -> np.ndarray | None:
    try:
        audio = au.load_audio_file(path, target_sr=config.PROCESSING_SAMPLE_RATE)
    except au.AudioLoadError as exc:
        print(f"  [skip] {path.name}: {exc}")
        return None
    sig = audio.signal[: int(max_seconds * audio.sr)]
    if sig.size < config.N_FFT * 4 or au.is_silent(sig):
        print(f"  [skip] {path.name}: too short or silent")
        return None
    return sig


# --------------------------------------------------------------------------
# Synthetic speech-like signals (pipeline testing only)
# --------------------------------------------------------------------------
def synth_utterance(rng: np.random.Generator, sr: int, duration: float) -> np.ndarray:
    """Voiced 'syllables' (harmonic source shaped by 3 formants) + fricative bursts + pauses.

    This imitates the *structure* of speech (pitch harmonics, formants,
    syllable envelopes, silences) but it is not speech. A model trained on it
    only proves that the pipeline works end-to-end.
    """
    n = int(duration * sr)
    out = np.zeros(n)
    pos = int(rng.uniform(0.05, 0.25) * sr)
    sos_fric = butter(4, [3000, 7000], btype="bandpass", fs=sr, output="sos")
    while pos < n:
        seg = int(rng.uniform(0.12, 0.40) * sr)
        seg = min(seg, n - pos)
        if seg < 64:
            break
        t = np.arange(seg) / sr
        env = np.hanning(seg)
        if rng.random() < 0.2:          # unvoiced fricative
            burst = sosfilt(sos_fric, rng.standard_normal(seg)) * env * rng.uniform(0.2, 0.5)
            out[pos:pos + seg] += burst
        else:                           # voiced syllable
            f0 = np.linspace(rng.uniform(90, 250), rng.uniform(90, 250), seg)
            f0 *= 1 + 0.01 * np.sin(2 * np.pi * 5 * t)      # vibrato
            phase = 2 * np.pi * np.cumsum(f0) / sr
            formants = [(rng.uniform(300, 900), 90), (rng.uniform(900, 2500), 140), (rng.uniform(2500, 3500), 200)]
            voiced = np.zeros(seg)
            for k in range(1, int(4000 / f0.max()) + 1):
                fk = k * f0
                amp = sum(np.exp(-0.5 * ((fk - F) / bw) ** 2) for F, bw in formants) + 0.02
                voiced += amp * np.sin(k * phase) / np.sqrt(k)
            out[pos:pos + seg] += voiced * env
        pos += seg + int(rng.uniform(0.03, 0.30) * sr)
    return au.peak_normalize(au.remove_dc(out))


# --------------------------------------------------------------------------
# Feature/target extraction
# --------------------------------------------------------------------------
def make_pair(clean: np.ndarray, snr: float, noise_type: str, seed: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    noisy, _ = ns.add_noise_at_snr(clean, snr, noise_type, seed)
    # Random overall level: the mask target is level-invariant, so this
    # teaches the model to cope with quiet and loud recordings.
    g = rng.uniform(0.25, 1.0) / (np.max(np.abs(noisy)) + config.EPS)
    return clean * g, noisy * g


def extract_xy(clean: np.ndarray, noisy: np.ndarray, cfg: ml.FeatureConfig, fb: np.ndarray):
    sr = cfg.sr
    S = sa.stft(clean, sr, cfg.n_fft, cfg.hop).magnitude
    Y = sa.stft(noisy, sr, cfg.n_fft, cfg.hop).magnitude
    return ml.build_features(Y, cfg, fb), ml.build_targets(S, Y, fb)


def collect_training_frames(signals: list[np.ndarray], snrs, noise_types, cfg, max_frames, seed):
    fb, _ = ml.mel_filterbank(cfg)
    rng = np.random.default_rng(seed)
    n_conditions = max(1, len(signals) * len(snrs) * len(noise_types))
    quota = max(50, max_frames // n_conditions)
    Xs, Ys = [], []
    cond = 0
    for i, clean in enumerate(signals):
        for snr in snrs:
            for nt in noise_types:
                c, y = make_pair(clean, snr, nt, seed + 1000 * i + cond, rng)
                X, T = extract_xy(c, y, cfg, fb)
                if X.shape[0] > quota:
                    idx = rng.choice(X.shape[0], quota, replace=False)
                    X, T = X[idx], T[idx]
                Xs.append(X)
                Ys.append(T)
                cond += 1
    return np.vstack(Xs), np.vstack(Ys)


# --------------------------------------------------------------------------
# Evaluation on held-out recordings
# --------------------------------------------------------------------------
def evaluate(model, val_signals: list[np.ndarray], cfg: ml.FeatureConfig, seed: int) -> dict:
    """Waveform-level evaluation on recordings never seen during training."""
    fb, _ = ml.mel_filterbank(cfg)
    sr = cfg.sr
    rng = np.random.default_rng(seed + 7)
    per_snr = {}
    Xv, Tv = [], []
    for snr in EVAL_SNRS_DB:
        rows = []
        for i, clean in enumerate(val_signals):
            c, noisy = make_pair(clean, snr, "white", seed + 50_000 + i, rng)
            X, T = extract_xy(c, noisy, cfg, fb)
            Xv.append(X)
            Tv.append(T)
            y_ml, _ = ml.apply_mask(noisy, sr, model.predict, cfg)
            y_ss = dsp.spectral_subtraction(noisy, sr).output
            y_lp = dsp.butterworth_lowpass(noisy, sr).output
            in_snr = metrics.snr_db(c, noisy)
            rows.append({
                "input_snr": in_snr,
                "ml_out_snr": metrics.snr_db(c, y_ml),
                "specsub_out_snr": metrics.snr_db(c, y_ss),
                "lowpass_out_snr": metrics.snr_db(c, y_lp),
            })
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        mean["ml_improvement"] = mean["ml_out_snr"] - mean["input_snr"]
        mean["specsub_improvement"] = mean["specsub_out_snr"] - mean["input_snr"]
        mean["lowpass_improvement"] = mean["lowpass_out_snr"] - mean["input_snr"]
        per_snr[f"{snr:g} dB"] = {k: round(v, 3) for k, v in mean.items()}
    Xv, Tv = np.vstack(Xv), np.vstack(Tv)
    if Xv.shape[0] > 40_000:
        idx = np.random.default_rng(seed).choice(Xv.shape[0], 40_000, replace=False)
        Xv, Tv = Xv[idx], Tv[idx]
    pred = model.predict(Xv)
    mse = float(np.mean((pred - Tv) ** 2))
    baseline_mse = float(np.mean((Tv - Tv.mean(axis=0)) ** 2))
    return {
        "noise_type": "white Gaussian",
        "per_input_snr": per_snr,
        "mask_mse": round(mse, 5),
        "mask_r2_vs_mean_predictor": round(1 - mse / (baseline_mse + config.EPS), 4),
    }


# --------------------------------------------------------------------------
# Main training routine (also callable from the dashboard)
# --------------------------------------------------------------------------
def train(
    data_dir: Path | None = None,
    synthetic: bool = False,
    n_synthetic: int = 40,
    max_files: int = config.TRAIN_MAX_FILES,
    max_seconds: float = config.TRAIN_MAX_SECONDS_PER_FILE,
    max_frames: int = config.TRAIN_MAX_FRAMES,
    snrs=config.TRAIN_SNRS_DB,
    noise_types=config.TRAIN_NOISE_TYPES,
    n_estimators: int = config.RF_N_ESTIMATORS,
    max_depth: int = config.RF_MAX_DEPTH,
    val_fraction: float = config.TRAIN_VALIDATION_FRACTION,
    seed: int = config.DEFAULT_SEED,
    output: Path = config.MODEL_PATH,
    summary_path: Path = config.TRAINING_SUMMARY_PATH,
    log=print,
) -> dict:
    cfg = ml.FeatureConfig()
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    if synthetic:
        log(f"Generating {n_synthetic} SYNTHETIC speech-like utterances (pipeline test only)…")
        signals = [synth_utterance(rng, cfg.sr, rng.uniform(2.0, 4.0)) for _ in range(n_synthetic)]
        groups = [f"synthetic_{i}" for i in range(n_synthetic)]
        source = "synthetic"
    else:
        data_dir = Path(data_dir or config.CLEAN_SPEECH_DIR)
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Training directory '{data_dir}' does not exist. Create it and add clean speech "
                "WAV/FLAC files (see data/README.md), or run with --synthetic to test the pipeline."
            )
        files = discover_files(data_dir, max_files)
        if not files:
            raise FileNotFoundError(
                f"No .wav or .flac files found in '{data_dir}'. Add clean speech recordings "
                "(see data/README.md), or run with --synthetic to test the pipeline."
            )
        log(f"Loading {len(files)} clean recordings from {data_dir.name}/ …")
        signals, groups = [], []
        for f in files:
            sig = load_clean(f, max_seconds)
            if sig is not None:
                signals.append(sig)
                groups.append(group_key(f, data_dir))
        source = "dataset"

    if len(signals) < 2:
        raise ValueError("At least two valid recordings are needed so one can be held out for validation.")

    # ---- split by group (speaker / recording), never by frame -------------
    unique = sorted(set(groups))
    rng.shuffle(unique)
    if len(unique) >= 2:
        n_val = max(1, int(round(len(unique) * val_fraction)))
        val_groups = set(unique[:n_val])
        split_note = f"split by {'speaker folder' if len(unique) < len(signals) else 'recording'}"
    else:
        # Single speaker folder: fall back to a per-recording split.
        val_groups = None
        split_note = "single group found – split by recording"
    if val_groups is None:
        order = rng.permutation(len(signals))
        n_val = max(1, int(round(len(signals) * val_fraction)))
        val_idx = set(order[:n_val].tolist())
    else:
        val_idx = {i for i, g in enumerate(groups) if g in val_groups}
    train_sigs = [s for i, s in enumerate(signals) if i not in val_idx]
    val_sigs = [s for i, s in enumerate(signals) if i in val_idx]
    log(f"{len(train_sigs)} training / {len(val_sigs)} validation recordings ({split_note}).")

    # ---- features ----------------------------------------------------------
    log("Extracting STFT band features and ratio-mask targets…")
    X, Y = collect_training_frames(train_sigs, snrs, noise_types, cfg, max_frames, seed)
    log(f"Training matrix: X {X.shape}, Y {Y.shape}")

    # ---- fit ---------------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        max_features=0.3, n_jobs=-1, random_state=seed,
    )
    log(f"Fitting RandomForestRegressor ({n_estimators} trees, max depth {max_depth})…")
    t_fit = time.perf_counter()
    model.fit(X, Y)
    fit_time = time.perf_counter() - t_fit
    log(f"Fitted in {fit_time:.1f} s.")

    # ---- evaluate on held-out recordings ----------------------------------
    log("Evaluating on held-out recordings…")
    evaluation = evaluate(model, val_sigs, cfg, seed)

    metadata = {
        "model_name": "Random Forest spectral ratio-mask estimator",
        "model_type": type(model).__name__,
        "data_source": source,
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train_recordings": len(train_sigs),
        "n_validation_recordings": len(val_sigs),
        "split": split_note,
        "train_snrs_db": list(snrs),
        "train_noise_types": list(noise_types),
        "n_training_frames": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_outputs": int(Y.shape[1]),
        "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth,
                            "min_samples_leaf": config.RF_MIN_SAMPLES_LEAF, "max_features": 0.3},
        "fit_time_s": round(fit_time, 2),
        "total_time_s": round(time.perf_counter() - t_start, 2),
        "validation": evaluation,
    }
    if source == "synthetic":
        metadata["warning"] = ("Trained on synthetic speech-like signals. This only verifies the ML pipeline; "
                               "it does not establish real-world speech-enhancement performance.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_config": cfg.to_dict(), "metadata": metadata}, output, compress=3)
    Path(summary_path).write_text(json.dumps(metadata, indent=2))
    log(f"Saved model to {output.name} and summary to {Path(summary_path).name}.")
    return metadata


def print_summary(meta: dict) -> None:
    print("\nHeld-out validation (white Gaussian noise, recordings never seen in training):")
    print(f"  Mask R² vs. mean predictor: {meta['validation']['mask_r2_vs_mean_predictor']}")
    print(f"  {'Input SNR':>10} | {'ML out':>8} {'Δ':>6} | {'SpecSub':>8} {'Δ':>6} | {'Low-pass':>8} {'Δ':>6}")
    for _, r in meta["validation"]["per_input_snr"].items():
        print(f"  {r['input_snr']:10.2f} | {r['ml_out_snr']:8.2f} {r['ml_improvement']:+6.2f} | "
              f"{r['specsub_out_snr']:8.2f} {r['specsub_improvement']:+6.2f} | "
              f"{r['lowpass_out_snr']:8.2f} {r['lowpass_improvement']:+6.2f}")
    if meta.get("warning"):
        print("\nWARNING: " + meta["warning"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data-dir", type=Path, default=config.CLEAN_SPEECH_DIR,
                     help="Folder with clean speech WAV/FLAC files (default: data/clean)")
    src.add_argument("--synthetic", action="store_true",
                     help="Train on generated speech-like signals (pipeline test only)")
    p.add_argument("--n-synthetic", type=int, default=40, help="Number of synthetic utterances")
    p.add_argument("--max-files", type=int, default=config.TRAIN_MAX_FILES)
    p.add_argument("--max-seconds", type=float, default=config.TRAIN_MAX_SECONDS_PER_FILE,
                   help="Use at most this many seconds of each recording")
    p.add_argument("--max-frames", type=int, default=config.TRAIN_MAX_FRAMES,
                   help="Maximum STFT frames used for fitting (memory/time limit)")
    p.add_argument("--snrs", type=float, nargs="+", default=list(config.TRAIN_SNRS_DB))
    p.add_argument("--noise-types", nargs="+", default=list(config.TRAIN_NOISE_TYPES),
                   choices=list(ns.NOISE_TYPES))
    p.add_argument("--n-estimators", type=int, default=config.RF_N_ESTIMATORS)
    p.add_argument("--max-depth", type=int, default=config.RF_MAX_DEPTH)
    p.add_argument("--val-fraction", type=float, default=config.TRAIN_VALIDATION_FRACTION)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=config.MODEL_PATH)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        meta = train(
            data_dir=None if args.synthetic else args.data_dir, synthetic=args.synthetic,
            n_synthetic=args.n_synthetic, max_files=args.max_files, max_seconds=args.max_seconds,
            max_frames=args.max_frames, snrs=tuple(args.snrs), noise_types=tuple(args.noise_types),
            n_estimators=args.n_estimators, max_depth=args.max_depth, val_fraction=args.val_fraction,
            seed=args.seed, output=args.output,
            summary_path=args.output.with_name("training_summary.json"),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}\n")
        print("How to supply training data:")
        print("  1. Download a clean speech corpus, e.g. LibriSpeech 'dev-clean' (FLAC is fine),")
        print("     VCTK, or record 20+ short clean sentences yourself in a quiet room.")
        print("  2. Copy the WAV/FLAC files to data/clean/ (optionally data/clean/<speaker>/...).")
        print("  3. Run:  python train_model.py --data-dir data/clean")
        print("To only test the pipeline:  python train_model.py --synthetic")
        return 1
    print_summary(meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
