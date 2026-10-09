"""Reverberation, noise at an exact SNR and two-channel measurement files."""

from __future__ import annotations

import numpy as np
import pytest
import speechdsp
from scipy.io import wavfile

from room_geometry_inference import (
    DatasetError,
    add_noise,
    measured_snr_db,
    read_recording,
    reverberate,
    write_recording,
)
from room_geometry_inference.recording import NOISE_KINDS, load_excitation


def test_reverberation_is_a_causal_convolution_cut_to_length(rng):
    signal = rng.standard_normal(500)
    response = np.array([0.0, 0.0, 1.0, 0.5])
    expected = np.convolve(signal, response)[:500]
    assert np.allclose(reverberate(signal, response), expected)


@pytest.mark.parametrize("kind", NOISE_KINDS)
@pytest.mark.parametrize("snr", [-5.0, 0.0, 20.0])
def test_noise_is_added_at_exactly_the_requested_snr(kind, snr, rng):
    clean = np.sin(np.arange(8000) / 9.0)
    noisy = add_noise(clean, snr, kind=kind, rng=rng)
    assert measured_snr_db(clean, noisy) == pytest.approx(snr, abs=1e-9)


def test_no_snr_means_no_noise():
    clean = np.ones(10)
    assert np.array_equal(add_noise(clean, None), clean)
    assert measured_snr_db(clean, clean) == float("inf")


def test_noise_rejects_silence_and_unknown_kinds():
    with pytest.raises(ValueError, match="silent"):
        add_noise(np.zeros(10), 10.0)
    with pytest.raises(ValueError, match="kind="):
        add_noise(np.ones(10), 10.0, kind="hum")


def test_measurements_round_trip_through_two_channel_wav(tmp_path, rng):
    excitation = rng.standard_normal(4000)
    microphone = np.convolve(excitation, [0.0, 0.6, 0.2])[:4000]
    path = write_recording(tmp_path / "sub" / "take.wav", excitation, microphone, 16_000)
    rate, data = wavfile.read(path)
    assert rate == 16_000 and data.shape == (4000, 2) and data.dtype == np.int16
    loaded = read_recording(path)
    ratio = loaded.excitation / excitation
    assert np.allclose(ratio[np.abs(excitation) > 0.5], ratio[0], rtol=1e-3)
    assert np.corrcoef(loaded.microphone, microphone)[0, 1] > 0.9999


def test_reading_resamples_both_channels(tmp_path, rng):
    path = write_recording(
        tmp_path / "take.wav", rng.standard_normal(4410), rng.standard_normal(4410), 44_100
    )
    loaded = read_recording(path, sample_rate=16_000)
    assert loaded.sample_rate == 16_000 and loaded.excitation.size == 1600
    assert loaded.duration_s == pytest.approx(0.1)


def test_float_wav_files_are_accepted(tmp_path):
    data = np.stack([np.linspace(-1, 1, 100), np.zeros(100)], axis=1).astype(np.float32)
    wavfile.write(tmp_path / "float.wav", 8000, data)
    loaded = read_recording(tmp_path / "float.wav")
    assert loaded.excitation[0] == pytest.approx(-1.0)


def test_mono_files_and_silent_excitations_are_rejected(tmp_path):
    speechdsp.write_wav(tmp_path / "mono.wav", np.full(100, 0.1), 8000)
    with pytest.raises(DatasetError, match="1 channel"):
        read_recording(tmp_path / "mono.wav")
    wavfile.write(tmp_path / "silent.wav", 8000, np.zeros((100, 2), dtype=np.int16))
    with pytest.raises(DatasetError, match="channel 0"):
        read_recording(tmp_path / "silent.wav")
    (tmp_path / "broken.wav").write_bytes(b"not a wav file")
    with pytest.raises(DatasetError, match="not a readable WAV"):
        read_recording(tmp_path / "broken.wav")
    with pytest.raises(FileNotFoundError):
        read_recording(tmp_path / "absent.wav")


def test_a_mono_excitation_file_is_read_and_resampled(tmp_path):
    speechdsp.write_wav(tmp_path / "speech.wav", 0.5 * np.sin(np.arange(800) / 5.0), 8000)
    signal = load_excitation(tmp_path / "speech.wav", 16_000)
    assert signal.shape == (1600,)
    with pytest.raises(FileNotFoundError, match="excitation file not found"):
        load_excitation(tmp_path / "absent.wav", 16_000)
