# AI-Based Speech Signal Analysis and Noise Reduction

**Using FFT, Digital Filtering, and Machine Learning** – a Signals and Systems project.

An interactive Streamlit dashboard that loads a real speech recording and lets you:

1. analyse it in the time domain (waveform, RMS, peak, crest factor);
2. analyse its spectrum (FFT, dominant frequency, band energy, Welch PSD, STFT spectrogram);
3. add reproducible noise at a chosen SNR (20 / 10 / 5 / 0 dB; white, pink or brown);
4. remove noise with **traditional DSP** (Butterworth low-pass filter, spectral subtraction);
5. remove noise with a **genuinely trained machine-learning model** (Random Forest spectral-mask estimator);
6. compare all outputs by listening, plotting and objective metrics (SNR, SNR improvement, RMSE, correlation);
7. download the processed audio, a results CSV and a self-contained HTML report.

Everything runs locally on a normal laptop CPU. No GPU and no large model downloads are required.

---

## Contents

- [Quick start](#quick-start)
- [System architecture](#system-architecture)
- [Background theory](#background-theory)
- [The ML model](#the-ml-model)
- [DSP vs ML](#dsp-vs-ml)
- [Dataset setup and model training](#dataset-setup-and-model-training)
- [Using the dashboard](#using-the-dashboard)
- [Metrics: what is and is not measurable](#metrics-what-is-and-is-not-measurable)
- [Demonstration plan for faculty](#demonstration-plan-for-faculty)
- [Running the tests](#running-the-tests)
- [Limitations and future work](#limitations-and-future-work)

---

## Quick start

Requires Python 3.11 or newer (tested on 3.11 and 3.13).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) train an ML model – see "Dataset setup" below
python train_model.py --data-dir data/clean     # real clean speech (recommended)
python train_model.py --synthetic               # pipeline test only (~2 min)

# 4. Run the tests
python -m pytest

# 5. Start the dashboard
streamlit run app.py
```

The browser opens at <http://localhost:8501>. The dashboard works **without** a trained model: the DSP
methods are always available, and the ML panel explains how to train a model.

> On Windows use `py -3.11 -m venv .venv` if several Python versions are installed. For `.m4a` uploads,
> install [ffmpeg](https://ffmpeg.org/download.html) and make sure it is on your `PATH`. WAV, FLAC, MP3 and
> OGG work without ffmpeg.

---

## System architecture

```
                ┌─────────────────────────── app.py (Streamlit UI) ──────────────────────────┐
 upload ──►  audio_utils ──► signal_analysis ──► noise_simulation ──► pipeline ──► metrics ──► report_generator
 (bytes)     decode/validate   FFT, PSD, STFT      SNR-controlled        │            SNR, RMSE,     HTML report
             mono, resample                        Gaussian noise        │            correlation
             DC, normalise                                               ├─► dsp_enhancement  (Approach A)
                                                                         │     Butterworth, spectral subtraction
                                                                         └─► ml_enhancement   (Approach B)
                                                                               features ─► trained model ─► mask ─► iSTFT
                                          train_model.py ─────────────────────────┘ (same build_features())
```

| File | Responsibility |
|---|---|
| `config.py` | All constants: sample rate, STFT size, feature layout, defaults |
| `audio_utils.py` | Safe decoding (libsndfile / ffmpeg), validation, mono mixdown, resampling, DC removal, normalisation, WAV export |
| `signal_analysis.py` | Time statistics, FFT, dominant frequency, band energy, Welch PSD, STFT / inverse STFT, spectrogram |
| `noise_simulation.py` | Seeded white/pink/brown noise scaled to an exact SNR |
| `dsp_enhancement.py` | Butterworth low-pass (SOS, zero-phase) and spectral subtraction |
| `ml_enhancement.py` | Feature extraction (shared by training and inference), model loading and validation, inference |
| `train_model.py` | Dataset loading, noisy-example generation, speaker/recording-wise split, training, held-out evaluation |
| `metrics.py` | SNR, segmental SNR, RMSE, correlation, comparison-table rows |
| `pipeline.py` | One processing run (UI-independent, tested) |
| `plots.py` | Plotly figures |
| `report_generator.py` | Self-contained HTML report with Matplotlib figures |
| `app.py` | The dashboard |
| `tests/` | 86 unit and integration tests |

The uploaded file is processed **in memory**. The original bytes are never modified or saved, and
M4A decoding uses a temporary directory that is deleted immediately.

---

## Background theory

### FFT – what frequencies are present?

The Discrete Fourier Transform rewrites a sampled signal `x[n]` as a sum of sinusoids:

```
X[k] = Σ_n x[n] · e^(−j2πkn/N),     f_k = k · fs / N
```

The **FFT** is a fast algorithm for computing it. `|X[k]|` tells you how strongly the frequency `f_k` is
present across the whole recording. The dashboard applies a Hann window first, which reduces *spectral
leakage* (energy smearing into neighbouring bins). It also scales the result so a sinusoid of amplitude
A shows a peak of about A.

* **Frequency resolution** = `fs / N` (for example 0.125 Hz for 8 s at 16 kHz).
* **Nyquist frequency** = `fs / 2`. Nothing above it can be represented, which is why every filter
  cut-off is checked against it.
* For speech, most energy lies between about 100 and 4000 Hz. The pitch (F0) is roughly 85–180 Hz for
  male voices and 165–255 Hz for female voices, and the formants (vocal-tract resonances) shape the
  higher frequencies.

### STFT and spectrograms – how does the spectrum change over time?

Speech is non-stationary, so one FFT of the whole file hides *when* each sound happens. The
**Short-Time Fourier Transform** splits the signal into short, overlapping, windowed frames and takes
an FFT of each one:

* frame = 512 samples (32 ms at 16 kHz), hop = 128 samples (8 ms, 75 % overlap), Hann window;
* the result is a matrix of complex values, frequency × time;
* the **spectrogram** shows `20·log10|X|` as an image. Horizontal stripes are pitch harmonics, dark
  vertical gaps are pauses, and bright high-frequency patches are fricatives (s, sh, f).

The STFT is **invertible**: the inverse STFT (overlap-add with a dual window) rebuilds the waveform.
The tests verify reconstruction to within 1e-10. This invertibility is what lets spectral subtraction
and the ML model change magnitudes and then produce audio again.

### Digital filtering – Butterworth low-pass

A Butterworth filter has a maximally flat pass-band:

```
|H(f)|² = 1 / (1 + (f / fc)^(2N))
```

* `fc` is the cut-off frequency (−3 dB point) and `N` is the order. A higher order gives a steeper roll-off.
* The filter is implemented as **second-order sections** for numerical stability.
* It is applied **forward and backward** (`sosfiltfilt`). This is *zero-phase* (no delay), and it squares
  the magnitude response, so the attenuation in dB doubles. The dashboard plots both curves.
* For recordings too short for zero-phase filtering, the code falls back to single-pass filtering and
  says so.

**Limitation:** a low-pass filter only removes noise *above* `fc`. It also removes genuine
high-frequency speech (s, f, sh) and cannot touch noise inside the speech band.

### Spectral subtraction

Noise is estimated from the quietest 10 % of STFT frames (assumed to be pauses), then subtracted from
each frame's magnitude:

```
|Ŝ| = max(|Y| − α·|N̂|,  β·|Y|),       phase(Ŝ) = phase(Y)
```

The over-subtraction factor α removes more noise, and the spectral floor β prevents negative magnitudes
and limits "musical noise" (random tonal blips). This is still DSP: a fixed rule, with no learning.

---

## The ML model

### Idea: supervised spectral-mask estimation

For each STFT frame of noisy speech, the model predicts a **gain (mask) per frequency band**.
Multiplying the noisy magnitude by this mask gives the **estimated clean spectral magnitude**.

| Step | What happens |
|---|---|
| 1. Clean speech | Read from `data/clean/` (or generated synthetically for a pipeline test) |
| 2. Noisy versions | Seeded noise added at −5, 0, 5, 10, 15 and 20 dB SNR, in white, pink and brown colours, with random overall level |
| 3. STFT | 512-point frames, 128-sample hop, giving 257 frequency bins |
| 4. Features | 257 bins compressed into **32 mel-spaced bands** (log energy) × **5 frames of context** (±2, edge frames repeated), plus a 32-value per-band **noise-floor estimate** (10th percentile over the utterance), for **192 features per frame** |
| 5. Targets | Band ratio mask `clip(|S|_band / |Y|_band, 0, 1)`, giving 32 outputs per frame |
| 6. Model | `RandomForestRegressor` (40 trees, depth 14), multi-output |
| 7. Inference | Predict the mask, interpolate it from bands to bins, compute `|Ŝ| = mask · |Y|` |
| 8. Reconstruction | Inverse STFT using `|Ŝ|` and the **noisy phase**; output length equals input length exactly |

Training and inference both call the same function, `ml_enhancement.build_features()`, so feature
dimensions, context windows and edge handling cannot diverge. The saved model stores its feature
configuration, and the app refuses to load a model whose configuration differs from the current one.

### Honest evaluation

* Training and validation sets are split **by speaker folder** (or by recording), never by frame, so
  overlapping frames of one recording cannot leak into validation.
* After training, the script reports, for the held-out recordings only:
  * mask R² compared with a predictor that always outputs the mean mask;
  * input SNR, ML output SNR, and the spectral-subtraction and low-pass output SNRs at 0, 5 and 10 dB.
* These numbers are stored in `models/training_summary.json` and shown in the dashboard.

### Synthetic training mode

`python train_model.py --synthetic` generates speech-*like* signals: harmonic "syllables" with pitch
glides and three formants, fricative noise bursts, and pauses. It exists **only to test that the
pipeline runs end to end**, and the app labels such a model as *synthetic* everywhere.

For example, during development the synthetic model reached the following on *held-out synthetic*
signals (white noise):

| Input SNR | ML output | Spectral subtraction | Low-pass (3.4 kHz, order 6) |
|---|---|---|---|
| 0 dB | 8.96 dB | 8.52 dB | 3.35 dB |
| 5 dB | 12.66 dB | 11.77 dB | 7.38 dB |
| 10 dB | 14.48 dB | 15.38 dB | 10.47 dB |

On a **real** LibriSpeech clip at 10 dB input SNR, the same synthetic model produced 7.32 dB, which is
*worse* than the noisy input (−2.68 dB). Spectral subtraction reached 13.90 dB (+3.90 dB) and the
low-pass filter 11.17 dB (+1.17 dB). This shows the limitation clearly: **a model only generalises to
data like its training data**. Train on real clean speech for meaningful results.

### Swapping in a stronger model later

The dashboard only calls `SpeechEnhancer.enhance(signal, sr) -> EnhancementResult`. A pretrained neural
enhancer (for example DCCRN, Demucs or MetricGAN+) can replace the Random Forest by implementing the
same method, without touching the UI, metrics or report.

---

## DSP vs ML

| | Traditional DSP (Approach A) | Machine learning (Approach B) |
|---|---|---|
| How it works | A fixed rule designed by an engineer | A mapping learned from noisy/clean examples |
| Needs data? | No | Yes – clean speech for training |
| Adapts to noise type? | Only through manual parameters (cut-off, α, β) | Learns patterns seen in training |
| Predictability | Fully predictable, easy to analyse mathematically | Depends on training data, must be validated on held-out data |
| Typical failure | Removes speech along with noise; fails for in-band noise | Fails on noise or speech unlike the training data |
| In this project | Butterworth low-pass, spectral subtraction | Random Forest band-mask estimator |

The Butterworth filter is **not** machine learning, and the project never labels it as such.

---

## Dataset setup and model training

1. Get clean speech, for example LibriSpeech `dev-clean` from <https://www.openslr.org/12> (CC BY 4.0),
   VCTK, or 20+ sentences that you and your classmates record in a quiet room.
2. Copy the files to `data/clean/`. A layout like `data/clean/<speaker>/<file>.wav` enables a
   speaker-wise split. See [`data/README.md`](data/README.md).
3. Train:

```bash
python train_model.py --data-dir data/clean
# a larger corpus, with limits to keep it laptop-friendly:
python train_model.py --data-dir path/to/LibriSpeech/dev-clean --max-files 300 --max-seconds 10
```

Useful options (`python train_model.py --help` lists all of them):

| Option | Default | Meaning |
|---|---|---|
| `--max-files` | 200 | Use at most this many recordings |
| `--max-seconds` | 15 | Use at most this many seconds per recording |
| `--max-frames` | 60 000 | Frames used for fitting (memory ≈ 60 000 × 192 × 4 B ≈ 46 MB) |
| `--snrs` | −5 0 5 10 15 20 | Training SNRs in dB |
| `--noise-types` | white pink brown | Noise colours used for training |
| `--n-estimators`, `--max-depth` | 40, 14 | Random Forest size (larger = slower, bigger file) |
| `--seed` | 42 | Makes the noise, split and model reproducible |

Training processes recordings one at a time and keeps only a random subset of frames from each, so
memory stays bounded. With the defaults it takes about 2 minutes on a 4-core laptop and produces a
model of about 30 MB. If the folder is empty or missing, the script prints setup instructions and
exits with an error code.

You can also train from the dashboard (**F · ML pipeline → Train or retrain the model**).

---

## Using the dashboard

1. **Upload** a recording in the sidebar (WAV/FLAC/MP3/OGG; M4A with ffmpeg; max 50 MB, first 5 minutes analysed).
2. Choose the **experiment mode**:
   * *Simulate noise (controlled experiment)*: your recording is the clean reference and noise is added at the chosen SNR. All metrics are ground truth.
   * *Process recording as-is (real noise)*: your recording is already noisy. SNR metrics are unavailable unless you also upload a **sample-aligned clean reference**.
3. Choose the noise level, noise type and seed (simulate mode), the processing method, and the filter cut-off and order.
4. Press **▶ Process audio**.
5. Explore the tabs:

| Tab | Shows |
|---|---|
| A · Overview (top) | File, duration, sample rate, channels, method, ML model status, processing status, preprocessing log |
| B · Original audio | Player for the untouched upload, waveform, amplitude range, RMS, peak, crest factor, metadata |
| C · Frequency analysis | FFT spectrum with dominant frequency, spectral centroid, Welch PSD, band-energy distribution, spectrogram (log or linear frequency axis) |
| D · Noise simulation | Noisy player, waveforms, spectrogram, measured SNR; **Experiment 2** compares 20 / 10 / 0 dB side by side |
| E · DSP enhancement | Filter frequency response, filter settings, filtered waveform, spectrum and player; spectral subtraction results |
| F · ML pipeline | Model details, the nine pipeline stages, feature shapes, timing, ML waveform, spectrogram and player; training summary; training buttons |
| G · Results comparison | Metrics table, SNR-improvement and RMSE charts, stacked waveforms, overlaid spectra, spectrograms, players |
| H · Download & export | WAVs for every output (and the noisy input), results CSV, HTML report |
| Applications | Lecture recordings, meetings, voice notes, ASR preprocessing |

Exported WAVs are 16-bit PCM at the processing rate (16 kHz by default). A signal whose peak exceeds
0.99 is **scaled down**, never hard-clipped. Open the HTML report in a browser and use *Print → Save as
PDF* to get a PDF.

---

## Metrics: what is and is not measurable

With a known clean reference `s` and a signal under test `y`, the error is `e = y − s`:

```
Input SNR       = 10·log10( Σ s² / Σ (noisy − s)² )
Output SNR      = 10·log10( Σ s² / Σ (y − s)² )
SNR improvement = Output SNR − Input SNR
RMSE            = sqrt( mean((y − s)²) )
Correlation     = Pearson correlation of s and y
Segmental SNR   = mean of 20 ms frame SNRs, clamped to [−10, 35] dB, silent frames skipped
```

A tiny epsilon (1e-12) only prevents division by zero.

* In **simulate** mode the reference is your recording before noise was added. The metrics are
  correct for that controlled setup, even though your "clean" recording may itself contain some noise.
* In **real** mode without a reference, these metrics are shown as **N/A**. The dashboard never
  estimates an SNR against an unknown clean signal and presents it as ground truth. Duration, RMS,
  peak and processing time are still reported.
* SNR measures waveform error, not perceived quality or intelligibility. Perceptual metrics such as
  PESQ or STOI would be needed for that.

---

## Demonstration plan for faculty

Use a 5–15 s clean recording of one speaker (WAV, quiet room). Train a model on real speech beforehand
if possible.

| # | Experiment | Steps | What to point out |
|---|---|---|---|
| 1 | Original speech analysis | Upload → tabs B and C | Waveform pauses vs. syllables; FFT peak (pitch or formant); energy concentrated below 4 kHz; harmonics in the spectrogram |
| 2 | Noise simulation | Tab D → *Experiment 2* | Measured SNR equals the target; at 0 dB noise power equals speech power; the noise fills the whole spectrogram |
| 3 | DSP noise reduction | Method *Compare all*, 10 dB → Process → tab E | Filter response and −3 dB point; low-pass removes the hiss above fc but dulls "s" sounds; spectral subtraction keeps the full band but adds musical noise |
| 4 | ML enhancement | Tab F | The nine stages, feature shape (frames × 192) and mask shape (frames × 32); listen and compare with DSP; mention the training data and the held-out validation table |
| 5 | Quantitative evaluation | Tab G (try 20, 10 and 0 dB) | Input SNR, output SNR, improvement, RMSE, correlation; which method wins depends on the SNR; download the CSV and report |
| 6 | Practical application | Applications tab; real mode with a noisy phone recording | Without a clean reference only statistics are measurable, so judge by ear and by spectrogram |

---

## Running the tests

```bash
python -m pytest            # 86 tests, about 20 s
python -m pytest -v tests/test_metrics.py
```

The tests cover: WAV, FLAC, MP3 and M4A loading; empty, invalid, unsupported, too-short, silent and
clipped files; stereo-to-mono conversion; resampling; FFT amplitude and frequency; STFT/iSTFT
reconstruction; noise at an exact SNR and its reproducibility; Butterworth pass-band, stop-band, −3 dB
point, zero phase, invalid parameters and the short-signal fallback; spectral subtraction; SNR, RMSE
and correlation definitions; feature dimensions and context-window edges; a missing, corrupt or
mismatched model; real (tiny) model training and inference; the speaker-wise split; WAV export
without clipping; the HTML report; and that the app starts without a model.

---

## Limitations and future work

**Limitations**

* Simulated noise is stationary Gaussian (white, pink or brown). Babble, music, traffic and
  reverberation behave differently.
* The low-pass filter removes speech above the cut-off and cannot remove in-band noise.
* Spectral subtraction assumes the noise is stationary and that the quietest frames contain no speech.
* The ML baseline estimates magnitudes only and reuses the noisy phase, which leaves residual noise at low SNR.
* A Random Forest on 32-band features is far less capable than deep networks trained on hundreds of hours of audio.
* The quality of the ML model depends entirely on the training data. A synthetic model does not generalise to real speech.
* SNR-based metrics do not measure perceived quality or intelligibility.

**Future improvements**

* Train on real noise recordings (for example MUSAN, DEMAND) in addition to synthetic noise.
* Add PESQ and STOI metrics, and word-error-rate tests with a speech recogniser.
* Plug in a pretrained neural enhancer through the `SpeechEnhancer` interface.
* Add Wiener filtering, band-pass and notch filters (e.g. 50/60 Hz hum removal).
* Add real-time processing from a microphone.

---

*Example data: the development tests used a LibriSpeech clip (CC BY 4.0) obtained from the librosa
example-data repository. No audio is included in this repository.*
