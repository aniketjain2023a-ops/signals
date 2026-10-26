import numpy as np
import pytest
import soundfile as sf

import audio_utils as au
import config
from conftest import SR, speech_like, wav_bytes


def test_load_wav_metadata(speech):
    a = au.load_audio(wav_bytes(speech), "clip.wav")
    assert a.sr == SR and a.original_sr == SR
    assert a.n_channels == 1
    assert a.original_samples == speech.size
    assert a.signal.size == speech.size
    assert np.isclose(np.max(np.abs(a.signal)), config.PEAK_NORMALIZE_TARGET)
    assert abs(a.signal.mean()) < 1e-9


def test_filename_path_is_stripped(speech):
    a = au.load_audio(wav_bytes(speech), "/home/student/private/clip.wav")
    assert a.filename == "clip.wav"


def test_original_upload_is_preserved(speech):
    data = wav_bytes(speech)
    before = bytes(data)
    a = au.load_audio(data, "clip.wav")
    assert data == before
    # 'original' keeps the decoded samples unnormalised
    assert np.allclose(a.original[:, 0], sf.read(__import__("io").BytesIO(data))[0])


def test_stereo_to_mono(speech):
    stereo = np.stack([speech, 0.5 * speech], axis=1)
    a = au.load_audio(wav_bytes(stereo), "st.wav", normalize=False, remove_dc_offset=False)
    assert a.n_channels == 2
    assert a.signal.ndim == 1
    assert np.allclose(a.signal, 0.75 * speech, atol=2e-4)  # 16-bit quantisation tolerance
    assert any("mono" in m for m in a.processing_log)


def test_to_mono_passthrough():
    x = np.arange(5.0)
    assert np.array_equal(au.to_mono(x), x)
    with pytest.raises(ValueError):
        au.to_mono(np.zeros((2, 2, 2)))


def test_resampling_length_and_frequency():
    sr_in = 44_100
    t = np.arange(sr_in) / sr_in
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    a = au.load_audio(wav_bytes(x, sr_in), "tone.wav")
    assert a.original_sr == 44_100 and a.sr == SR
    assert abs(a.signal.size - SR) <= 1
    spec = np.abs(np.fft.rfft(a.signal))
    f = np.fft.rfftfreq(a.signal.size, 1 / SR)
    assert abs(f[np.argmax(spec)] - 440) < 2
    assert any("Resampled" in m for m in a.processing_log)


def test_no_resampling_when_disabled():
    x = speech_like(sr=22_050)
    a = au.load_audio(wav_bytes(x, 22_050), "a.wav", target_sr=None)
    assert a.sr == 22_050


def test_flac_supported(speech):
    import io
    buf = io.BytesIO()
    sf.write(buf, speech, SR, format="FLAC")
    a = au.load_audio(buf.getvalue(), "a.flac")
    assert a.signal.size == speech.size


def test_empty_file():
    with pytest.raises(au.AudioLoadError, match="empty"):
        au.load_audio(b"", "a.wav")


def test_unsupported_extension(speech):
    with pytest.raises(au.AudioLoadError, match="Unsupported"):
        au.load_audio(wav_bytes(speech), "notes.txt")


def test_invalid_audio_bytes():
    with pytest.raises(au.AudioLoadError):
        au.load_audio(b"this is definitely not audio" * 100, "fake.wav")


def test_too_short():
    with pytest.raises(au.AudioLoadError, match="too short"):
        au.load_audio(wav_bytes(np.zeros(10)), "short.wav")


def test_silent_recording_no_division_by_zero():
    a = au.load_audio(wav_bytes(np.zeros(4000)), "silence.wav")
    assert np.all(np.isfinite(a.signal))
    assert np.all(a.signal == 0)
    assert au.is_silent(a.signal)
    assert any("silent" in m for m in a.processing_log)


def test_clipping_detected():
    x = np.clip(3 * speech_like(), -1, 1)
    a = au.load_audio(wav_bytes(x, subtype="FLOAT"), "clip.wav")
    assert any("clipped" in m for m in a.processing_log)


def test_dc_removal():
    a = au.load_audio(wav_bytes(speech_like() * 0.5 + 0.2), "dc.wav")
    assert abs(a.signal.mean()) < 1e-9
    assert any("DC" in m for m in a.processing_log)


def test_peak_normalize_silence_safe():
    assert np.all(au.peak_normalize(np.zeros(100)) == 0)


def test_output_wav_creation_and_no_clipping(speech):
    loud = 3.0 * speech
    data = au.to_wav_bytes(loud, SR)
    y, sr = sf.read(__import__("io").BytesIO(data))
    assert sr == SR and y.size == loud.size
    assert np.max(np.abs(y)) <= 0.99 + 1e-4
    assert np.all(np.isfinite(y))
    # Shape is preserved (scaled, not clipped)
    assert np.corrcoef(y, loud)[0, 1] > 0.9999


def test_export_replaces_non_finite():
    out, _ = au.prepare_for_export(np.array([0.1, np.nan, np.inf, -0.2]))
    assert np.all(np.isfinite(out))


@pytest.mark.skipif(not au.ffmpeg_available(), reason="ffmpeg not installed")
def test_m4a_via_ffmpeg(tmp_path, speech):
    import subprocess
    src, dst = tmp_path / "a.wav", tmp_path / "a.m4a"
    sf.write(src, speech, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", str(dst)], check=True)
    a = au.load_audio(dst.read_bytes(), "a.m4a")
    assert a.sr == SR
    assert abs(a.duration_s - 1.0) < 0.1      # AAC adds encoder priming samples


def test_mp3_decoding(tmp_path, speech):
    import io
    buf = io.BytesIO()
    try:
        sf.write(buf, speech, SR, format="MP3")
    except (sf.LibsndfileError, TypeError, ValueError):
        pytest.skip("libsndfile without MP3 support")
    a = au.load_audio(buf.getvalue(), "a.mp3")
    assert abs(a.duration_s - 1.0) < 0.1
