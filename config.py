"""
Central configuration for the Speech Signal Analysis and Noise Reduction project.

Every module imports its constants from here so that the dashboard, the
training script and the tests always agree on sample rate, STFT settings and
ML feature layout. If you change the STFT or feature settings you MUST retrain
the ML model; the model bundle stores the settings it was trained with and the
app refuses to use a model whose settings do not match.
"""

from pathlib import Path

PROJECT_TITLE = (
    "AI-Based Speech Signal Analysis and Noise Reduction "
    "Using FFT, Digital Filtering, and Machine Learning"
)
PROJECT_SHORT_NAME = "Speech Enhancement Dashboard"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CLEAN_SPEECH_DIR = DATA_DIR / "clean"          # user-supplied clean speech WAV/FLAC files
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODELS_DIR / "speech_enhancer.joblib"
TRAINING_SUMMARY_PATH = MODELS_DIR / "training_summary.json"

# --------------------------------------------------------------------------
# Audio input
# --------------------------------------------------------------------------
PROCESSING_SAMPLE_RATE = 16_000        # Hz – standard rate for speech processing
MAX_UPLOAD_MB = 50                     # reject larger uploads
MAX_DURATION_S = 300.0                 # analyse at most 5 minutes of audio
MIN_SAMPLES = 32                       # anything shorter is not analysable
PEAK_NORMALIZE_TARGET = 0.95           # peak level after normalisation
CLIPPING_THRESHOLD = 0.999             # |x| >= this counts as a clipped sample
SILENCE_RMS_THRESHOLD = 1e-5           # RMS below this is treated as silence
EPS = 1e-12                            # numerical guard against division by zero

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
# Formats that libsndfile (via soundfile) can decode directly.
SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
# Formats that need an external ffmpeg binary.
FFMPEG_EXTENSIONS = {".m4a"}

# --------------------------------------------------------------------------
# STFT analysis / synthesis (shared by DSP spectral subtraction and ML)
# --------------------------------------------------------------------------
N_FFT = 512            # 32 ms window at 16 kHz -> 257 frequency bins (31.25 Hz spacing)
HOP_LENGTH = 128       # 8 ms hop (75 % overlap)
WINDOW = "hann"

# --------------------------------------------------------------------------
# Noise simulation
# --------------------------------------------------------------------------
DEFAULT_SEED = 42
SNR_PRESETS_DB = {
    "20 dB (mild noise)": 20.0,
    "10 dB (moderate noise)": 10.0,
    "5 dB (strong noise)": 5.0,
    "0 dB (severe noise)": 0.0,
}

# --------------------------------------------------------------------------
# DSP defaults
# --------------------------------------------------------------------------
DEFAULT_CUTOFF_HZ = 3_400.0     # telephone-band upper edge, keeps most speech intelligibility
DEFAULT_FILTER_ORDER = 6
MIN_FILTER_ORDER = 1
MAX_FILTER_ORDER = 12
SPECTRAL_SUB_ALPHA = 2.0        # over-subtraction factor
SPECTRAL_SUB_BETA = 0.02        # spectral floor (fraction of noisy magnitude kept)
NOISE_ESTIMATE_PERCENTILE = 10  # quietest 10 % of frames are used as noise estimate

# --------------------------------------------------------------------------
# ML feature layout – changing any of these requires retraining
# --------------------------------------------------------------------------
ML_N_BANDS = 32           # mel-spaced bands that compress the 257 STFT bins
ML_CONTEXT_FRAMES = 2     # frames of context on each side (window = 2*2+1 = 5 frames)
ML_NOISE_FLOOR_PERCENTILE = 10
ML_MASK_FLOOR = 0.05      # minimum gain applied at inference (limits musical noise)
ML_FEATURE_VERSION = 1    # bump when build_features() changes

# Training defaults (sized for a normal laptop, CPU only)
TRAIN_SNRS_DB = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
TRAIN_NOISE_TYPES = ("white", "pink", "brown")
TRAIN_MAX_FILES = 200
TRAIN_MAX_SECONDS_PER_FILE = 15.0
TRAIN_MAX_FRAMES = 60_000            # random subsample of frames kept for fitting
TRAIN_VALIDATION_FRACTION = 0.2
RF_N_ESTIMATORS = 40
RF_MAX_DEPTH = 14
RF_MIN_SAMPLES_LEAF = 8
